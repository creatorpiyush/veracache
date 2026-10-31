use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::SystemTime;

use async_trait::async_trait;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use veracache_core::{
    CacheEntry, Candidate, DeleteFilter, EntryId, Error, Feedback, Namespace, Result, VectorStore,
};

/// Source of the current time. Tests inject one to control expiry.
pub type Clock = Arc<dyn Fn() -> SystemTime + Send + Sync>;

type EntryKey = (Namespace, [u8; 32]);

/// Entries per namespace at which search moves from brute force to the index.
const INDEX_THRESHOLD: usize = 5_000;

/// One namespace's entries, with its own index once it is large enough.
#[derive(Default)]
struct Bucket {
    entries: HashMap<[u8; 32], CacheEntry>,
    index: Option<Index>,
    slots: HashMap<[u8; 32], u64>,
    keys: HashMap<u64, [u8; 32]>,
    next_slot: u64,
}

impl Bucket {
    fn insert(&mut self, entry: CacheEntry, threshold: usize) -> Result<Option<CacheEntry>> {
        check_finite(&entry.embedding)?;
        if let Some(index) = &self.index
            && entry.embedding.len() != index.dimensions()
        {
            return Err(Error::Store("vector dimension mismatch".into()));
        }

        let replaced = self.remove(&entry.key_hash)?;
        if self.index.is_some() {
            self.add_to_index(&entry)?;
        }
        self.entries.insert(entry.key_hash, entry);
        if self.index.is_none() && self.entries.len() >= threshold {
            self.build_index();
        }
        Ok(replaced)
    }

    fn remove(&mut self, key: &[u8; 32]) -> Result<Option<CacheEntry>> {
        if let Some(slot) = self.slots.remove(key) {
            self.keys.remove(&slot);
            if let Some(index) = &self.index {
                index.remove(slot).map_err(store_error)?;
            }
        }
        Ok(self.entries.remove(key))
    }

    fn add_to_index(&mut self, entry: &CacheEntry) -> Result<()> {
        let Some(index) = &self.index else {
            return Ok(());
        };
        if index.size() >= index.capacity() {
            index
                .reserve((index.capacity() * 2).max(16))
                .map_err(store_error)?;
        }
        let slot = self.next_slot;
        index.add(slot, &entry.embedding).map_err(store_error)?;
        self.next_slot += 1;
        self.slots.insert(entry.key_hash, slot);
        self.keys.insert(slot, entry.key_hash);
        Ok(())
    }

    // Brute force stays correct, so a failed build only costs speed.
    fn build_index(&mut self) {
        let Some(dimensions) = self.entries.values().next().map(|e| e.embedding.len()) else {
            return;
        };
        if self
            .entries
            .values()
            .any(|e| e.embedding.len() != dimensions)
        {
            return;
        }
        let Ok(index) = new_index(dimensions, self.entries.len() * 2) else {
            return;
        };

        let (mut slots, mut keys) = (HashMap::new(), HashMap::new());
        let mut next_slot = 0;
        for (key, entry) in &self.entries {
            if index.add(next_slot, &entry.embedding).is_err() {
                return;
            }
            slots.insert(*key, next_slot);
            keys.insert(next_slot, *key);
            next_slot += 1;
        }
        self.index = Some(index);
        self.slots = slots;
        self.keys = keys;
        self.next_slot = next_slot;
    }

    fn search(&self, vector: &[f32], k: usize, now: SystemTime) -> Result<Vec<Candidate>> {
        check_finite(vector)?;
        let mut candidates = match &self.index {
            Some(index) => self.search_index(index, vector, k, now)?,
            None => self.search_all(vector, now)?,
        };
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        candidates.truncate(k);
        Ok(candidates)
    }

    fn search_all(&self, vector: &[f32], now: SystemTime) -> Result<Vec<Candidate>> {
        let mut candidates = Vec::new();
        for entry in self.entries.values().filter(|e| !is_expired(e, now)) {
            if entry.embedding.len() != vector.len() {
                return Err(Error::Store("vector dimension mismatch".into()));
            }
            candidates.extend(scored(entry, vector));
        }
        Ok(candidates)
    }

    // The index holds expired entries, so widen the search until k live ones
    // are found. Scores are recomputed because the index is approximate.
    // ponytail: a namespace of mostly expired entries degrades toward a full
    // scan; evict expired entries in the background if that shows up.
    fn search_index(
        &self,
        index: &Index,
        vector: &[f32],
        k: usize,
        now: SystemTime,
    ) -> Result<Vec<Candidate>> {
        if vector.len() != index.dimensions() {
            return Err(Error::Store("vector dimension mismatch".into()));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut fetch = k * 2;
        loop {
            let matches = index.search(vector, fetch).map_err(store_error)?;
            let candidates: Vec<_> = matches
                .keys
                .iter()
                .filter_map(|slot| self.entries.get(self.keys.get(slot)?))
                .filter(|entry| !is_expired(entry, now))
                .filter_map(|entry| scored(entry, vector))
                .collect();
            if candidates.len() >= k || fetch >= index.size() {
                return Ok(candidates);
            }
            fetch *= 2;
        }
    }
}

// usearch never returns from `add` when a vector contains NaN.
fn check_finite(vector: &[f32]) -> Result<()> {
    if vector.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::Store("vector is not finite".into()))
    }
}

fn new_index(dimensions: usize, capacity: usize) -> Result<Index> {
    let index = Index::new(&IndexOptions {
        dimensions,
        metric: MetricKind::IP,
        quantization: ScalarKind::F32,
        connectivity: 16,
        expansion_add: 200,
        expansion_search: 64,
        multi: false,
    })
    .map_err(store_error)?;
    index.reserve(capacity).map_err(store_error)?;
    Ok(index)
}

fn scored(entry: &CacheEntry, vector: &[f32]) -> Option<Candidate> {
    let score: f32 = entry.embedding.iter().zip(vector).map(|(a, b)| a * b).sum();
    score.is_finite().then(|| Candidate {
        entry: entry.clone(),
        score,
    })
}

#[derive(Default)]
struct State {
    namespaces: HashMap<Namespace, Bucket>,
    ids: HashMap<EntryId, EntryKey>,
}

/// An in-process vector store. Search is brute force within a small namespace
/// and uses an HNSW index within a large one.
pub struct MemoryStore {
    state: RwLock<State>,
    clock: Clock,
    index_threshold: usize,
}

impl MemoryStore {
    /// Creates an empty store that reads the system clock.
    pub fn new() -> Self {
        Self::with_clock(Arc::new(SystemTime::now))
    }

    /// Creates an empty store that reads `clock`.
    pub fn with_clock(clock: Clock) -> Self {
        Self {
            state: RwLock::default(),
            clock,
            index_threshold: INDEX_THRESHOLD,
        }
    }

    /// Sets the entries per namespace at which search moves to the index.
    pub fn with_index_threshold(mut self, threshold: usize) -> Self {
        self.index_threshold = threshold;
        self
    }

    fn read(&self) -> Result<RwLockReadGuard<'_, State>> {
        self.state
            .read()
            .map_err(|_| Error::Store("lock poisoned".into()))
    }

    fn write(&self) -> Result<RwLockWriteGuard<'_, State>> {
        self.state
            .write()
            .map_err(|_| Error::Store("lock poisoned".into()))
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for MemoryStore {
    async fn upsert(&self, entry: CacheEntry) -> Result<()> {
        let mut state = self.write()?;
        let key = (entry.namespace, entry.key_hash);
        let id = entry.id;

        let bucket = state.namespaces.entry(entry.namespace).or_default();
        let replaced = bucket.insert(entry, self.index_threshold)?;
        if let Some(old) = replaced
            && old.id != id
        {
            state.ids.remove(&old.id);
        }

        if let Some(previous) = state.ids.insert(id, key)
            && previous != key
            && let Some(bucket) = state.namespaces.get_mut(&previous.0)
        {
            bucket.remove(&previous.1)?;
        }
        Ok(())
    }

    async fn search(&self, ns: &Namespace, vector: &[f32], k: usize) -> Result<Vec<Candidate>> {
        let now = (self.clock)();
        let state = self.read()?;
        match state.namespaces.get(ns) {
            Some(bucket) => bucket.search(vector, k, now),
            None => Ok(Vec::new()),
        }
    }

    async fn get_exact(&self, ns: &Namespace, key_hash: &[u8; 32]) -> Result<Option<CacheEntry>> {
        let now = (self.clock)();
        let state = self.read()?;
        Ok(state
            .namespaces
            .get(ns)
            .and_then(|bucket| bucket.entries.get(key_hash))
            .filter(|entry| !is_expired(entry, now))
            .cloned())
    }

    async fn delete(&self, filter: DeleteFilter) -> Result<u64> {
        let mut guard = self.write()?;
        let State { namespaces, ids } = &mut *guard;
        let mut deleted = 0;
        for bucket in namespaces.values_mut() {
            let matched: Vec<_> = bucket
                .entries
                .iter()
                .filter(|(_, entry)| matches(&filter, entry))
                .map(|(key, entry)| (*key, entry.id))
                .collect();
            for (key, id) in matched {
                bucket.remove(&key)?;
                ids.remove(&id);
                deleted += 1;
            }
        }
        namespaces.retain(|_, bucket| !bucket.entries.is_empty());
        Ok(deleted)
    }

    async fn record_feedback(&self, id: EntryId, fb: Feedback) -> Result<()> {
        let mut state = self.write()?;
        let Some((ns, key)) = state.ids.get(&id).copied() else {
            return Err(Error::Store("entry not found".into()));
        };
        let entry = state
            .namespaces
            .get_mut(&ns)
            .and_then(|bucket| bucket.entries.get_mut(&key))
            .ok_or_else(|| Error::Store("entry not found".into()))?;
        if !fb.correct {
            entry.stats.false_hit_reports += 1;
        }
        Ok(())
    }
}

fn store_error(error: impl std::fmt::Display) -> Error {
    Error::Store(error.to_string())
}

fn is_expired(entry: &CacheEntry, now: SystemTime) -> bool {
    entry.ttl.is_some_and(|ttl| entry.created_at + ttl <= now)
}

fn matches(filter: &DeleteFilter, entry: &CacheEntry) -> bool {
    if *filter == DeleteFilter::default() {
        return false;
    }
    filter.namespace.is_none_or(|ns| ns == entry.namespace)
        && filter
            .tag
            .as_ref()
            .is_none_or(|tag| entry.tags.contains(tag))
        && (filter.version.is_none() || filter.version == entry.version)
        && filter
            .created_before
            .is_none_or(|time| entry.created_at < time)
}

#[cfg(test)]
mod tests {
    use veracache_core::{EntryStats, ResponseMeta};

    use super::*;

    const NS: Namespace = Namespace([1; 32]);

    fn vector_for(id: u32) -> Vec<f32> {
        let angle = f64::from(id + 1) * 0.7;
        let mut vector: Vec<f32> = (1..=8u32)
            .map(|axis| (angle * f64::from(axis)).sin() as f32)
            .collect();
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        vector.iter_mut().for_each(|x| *x /= norm);
        vector
    }

    fn entry(id: u32, embedding: Vec<f32>) -> CacheEntry {
        let mut key_hash = [0; 32];
        key_hash[..4].copy_from_slice(&id.to_le_bytes());
        CacheEntry {
            id: EntryId(id.into()),
            namespace: NS,
            key_hash,
            prompt_text: None,
            analyzed: None,
            embedding,
            response: Vec::new(),
            response_meta: ResponseMeta::default(),
            created_at: SystemTime::now(),
            ttl: None,
            tags: Vec::new(),
            version: None,
            stats: EntryStats::default(),
            threshold_override: None,
        }
    }

    async fn filled(threshold: usize, count: u32) -> MemoryStore {
        let store = MemoryStore::new().with_index_threshold(threshold);
        for id in 0..count {
            store.upsert(entry(id, vector_for(id))).await.unwrap();
        }
        store
    }

    fn is_indexed(store: &MemoryStore) -> bool {
        store.read().unwrap().namespaces[&NS].index.is_some()
    }

    async fn top_id(store: &MemoryStore, vector: &[f32]) -> Option<u128> {
        let found = store.search(&NS, vector, 1).await.unwrap();
        found.first().map(|candidate| candidate.entry.id.0)
    }

    #[tokio::test]
    async fn passes_conformance_with_brute_force() {
        crate::conformance::run(MemoryStore::with_clock).await;
    }

    #[tokio::test]
    async fn passes_conformance_with_index() {
        crate::conformance::run(|clock| MemoryStore::with_clock(clock).with_index_threshold(1))
            .await;
    }

    #[tokio::test]
    async fn switches_to_the_index_at_the_threshold() {
        let store = filled(100, 99).await;
        assert!(!is_indexed(&store));

        store.upsert(entry(99, vector_for(99))).await.unwrap();
        assert!(is_indexed(&store));

        for id in [0, 57, 99] {
            assert_eq!(top_id(&store, &vector_for(id)).await, Some(id.into()));
        }
    }

    #[tokio::test]
    async fn entries_added_after_the_switch_are_found() {
        let store = filled(10, 300).await;
        assert_eq!(top_id(&store, &vector_for(299)).await, Some(299));
    }

    #[tokio::test]
    async fn scores_are_exact_dot_products() {
        let store = filled(10, 50).await;
        let query = vector_for(7);
        let found = store.search(&NS, &query, 1).await.unwrap();
        assert!((found[0].score - 1.0).abs() < 1e-5, "{}", found[0].score);
    }

    #[tokio::test]
    async fn deleted_entries_leave_the_index() {
        let store = filled(10, 50).await;
        let query = vector_for(7);

        let mut tagged = entry(7, query.clone());
        tagged.tags = vec!["gone".into()];
        store.upsert(tagged).await.unwrap();
        let filter = DeleteFilter {
            tag: Some("gone".into()),
            ..DeleteFilter::default()
        };
        assert_eq!(store.delete(filter).await.unwrap(), 1);

        assert_ne!(top_id(&store, &query).await, Some(7));
    }

    #[tokio::test]
    async fn replaced_entries_use_their_new_vector() {
        let store = filled(10, 50).await;
        let (old, new) = (vector_for(7), vector_for(1000));

        store.upsert(entry(7, new.clone())).await.unwrap();

        assert_eq!(top_id(&store, &new).await, Some(7));
        assert_ne!(top_id(&store, &old).await, Some(7));
    }

    #[tokio::test]
    async fn indexed_namespace_rejects_wrong_dimensions() {
        let store = filled(10, 50).await;
        assert!(store.upsert(entry(7, vec![1.0, 0.0])).await.is_err());
        assert_eq!(top_id(&store, &vector_for(7)).await, Some(7));
    }

    #[tokio::test]
    async fn mixed_dimensions_stay_on_brute_force() {
        let store = MemoryStore::new().with_index_threshold(2);
        store.upsert(entry(1, vec![1.0, 0.0])).await.unwrap();
        store.upsert(entry(2, vec![1.0])).await.unwrap();
        assert!(!is_indexed(&store));
    }

    #[tokio::test]
    async fn namespaces_get_separate_indexes() {
        let store = filled(10, 50).await;
        let mut other = entry(1_000, vector_for(7));
        other.namespace = Namespace([2; 32]);
        store.upsert(other).await.unwrap();

        let found = store
            .search(&Namespace([2; 32]), &vector_for(7), 5)
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].entry.id.0, 1_000);
    }

    #[tokio::test]
    async fn rejects_non_finite_vectors() {
        for threshold in [1, 1_000] {
            let store = filled(threshold, 50).await;
            for bad in [f32::NAN, f32::INFINITY] {
                let mut vector = vector_for(1);
                vector[0] = bad;
                assert!(store.upsert(entry(1_000, vector.clone())).await.is_err());
                assert!(store.search(&NS, &vector, 5).await.is_err());
            }
        }
    }

    fn at_key(mut item: CacheEntry, key: u8, namespace: Namespace) -> CacheEntry {
        item.key_hash = [key; 32];
        item.namespace = namespace;
        item
    }

    #[tokio::test]
    async fn default_store_is_empty_and_usable() {
        let store = MemoryStore::default();
        assert!(store.search(&NS, &[1.0], 5).await.unwrap().is_empty());
        store.upsert(entry(1, vec![1.0])).await.unwrap();
        assert_eq!(store.search(&NS, &[1.0], 5).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn same_id_at_a_new_key_moves_the_entry() {
        for threshold in [1, 1_000] {
            let store = MemoryStore::new().with_index_threshold(threshold);
            let vector = vector_for(1);
            let first = at_key(entry(1, vector.clone()), 1, NS);
            let moved = at_key(entry(1, vector.clone()), 2, NS);
            store.upsert(first).await.unwrap();
            store.upsert(moved).await.unwrap();

            assert_eq!(store.get_exact(&NS, &[1; 32]).await.unwrap(), None);
            assert!(store.get_exact(&NS, &[2; 32]).await.unwrap().is_some());
            assert_eq!(store.search(&NS, &vector, 5).await.unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn same_id_in_a_new_namespace_moves_the_entry() {
        let store = MemoryStore::new();
        let other = Namespace([2; 32]);
        store
            .upsert(at_key(entry(1, vec![1.0]), 1, NS))
            .await
            .unwrap();
        store
            .upsert(at_key(entry(1, vec![1.0]), 1, other))
            .await
            .unwrap();

        assert_eq!(store.get_exact(&NS, &[1; 32]).await.unwrap(), None);
        assert!(store.get_exact(&other, &[1; 32]).await.unwrap().is_some());

        let correct = Feedback { correct: false };
        store.record_feedback(EntryId(1), correct).await.unwrap();
        let found = store.get_exact(&other, &[1; 32]).await.unwrap().unwrap();
        assert_eq!(found.stats.false_hit_reports, 1);
    }

    #[tokio::test]
    async fn correct_feedback_changes_no_counter() {
        let store = MemoryStore::new();
        let stored = entry(1, vec![1.0]);
        store.upsert(stored.clone()).await.unwrap();

        let fb = Feedback { correct: true };
        store.record_feedback(stored.id, fb).await.unwrap();

        let found = store.get_exact(&NS, &stored.key_hash).await.unwrap();
        assert_eq!(found.unwrap().stats, EntryStats::default());
    }

    #[tokio::test]
    async fn k_larger_than_the_namespace_returns_everything() {
        for threshold in [1, 1_000] {
            let store = filled(threshold, 3).await;
            let found = store.search(&NS, &vector_for(0), 50).await.unwrap();
            assert_eq!(found.len(), 3);
        }
    }

    #[tokio::test]
    async fn delete_spans_namespaces_and_combines_fields() {
        let store = MemoryStore::new();
        let other = Namespace([2; 32]);
        let tagged = |id, namespace, version: &str| {
            let mut item = at_key(entry(id, vec![1.0]), id as u8, namespace);
            item.tags = vec!["tag".into()];
            item.version = Some(version.into());
            item
        };
        store.upsert(tagged(1, NS, "v1")).await.unwrap();
        store.upsert(tagged(2, other, "v1")).await.unwrap();
        store.upsert(tagged(3, other, "v2")).await.unwrap();

        let both = DeleteFilter {
            tag: Some("tag".into()),
            version: Some("v1".into()),
            ..DeleteFilter::default()
        };
        assert_eq!(store.delete(both).await.unwrap(), 2);

        assert_eq!(store.get_exact(&NS, &[1; 32]).await.unwrap(), None);
        assert_eq!(store.get_exact(&other, &[2; 32]).await.unwrap(), None);
        assert!(store.get_exact(&other, &[3; 32]).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn deleted_ids_reject_feedback() {
        let store = filled(10, 20).await;
        let filter = DeleteFilter {
            namespace: Some(NS),
            ..DeleteFilter::default()
        };
        assert_eq!(store.delete(filter).await.unwrap(), 20);

        let fb = Feedback { correct: false };
        assert!(store.record_feedback(EntryId(1), fb).await.is_err());
    }

    #[tokio::test]
    async fn emptied_namespace_can_be_filled_again() {
        for threshold in [1, 1_000] {
            let store = filled(threshold, 20).await;
            let filter = DeleteFilter {
                namespace: Some(NS),
                ..DeleteFilter::default()
            };
            store.delete(filter).await.unwrap();
            assert!(
                store
                    .search(&NS, &vector_for(1), 5)
                    .await
                    .unwrap()
                    .is_empty()
            );

            store.upsert(entry(1, vector_for(1))).await.unwrap();
            assert_eq!(top_id(&store, &vector_for(1)).await, Some(1));
        }
    }

    #[tokio::test]
    async fn index_grows_past_its_reserved_capacity() {
        let store = filled(4, 4).await;
        let reserved = {
            let state = store.read().unwrap();
            state.namespaces[&NS].index.as_ref().unwrap().capacity()
        };

        let count = u32::try_from(reserved).unwrap() * 3;
        for id in 4..count {
            store.upsert(entry(id, vector_for(id))).await.unwrap();
        }

        for id in [0, count / 2, count - 1] {
            assert_eq!(top_id(&store, &vector_for(id)).await, Some(id.into()));
        }
    }

    #[tokio::test]
    async fn rejected_upsert_leaves_the_old_entry() {
        for threshold in [1, 1_000] {
            let store = filled(threshold, 10).await;
            let mut bad = vector_for(3);
            bad[0] = f32::NAN;

            assert!(store.upsert(entry(3, bad)).await.is_err());

            assert_eq!(top_id(&store, &vector_for(3)).await, Some(3));
        }
    }

    #[tokio::test]
    async fn repeated_replacement_keeps_one_entry() {
        let store = filled(10, 20).await;
        for _ in 0..50 {
            store.upsert(entry(5, vector_for(5))).await.unwrap();
        }
        let found = store.search(&NS, &vector_for(5), 25).await.unwrap();
        assert_eq!(found.len(), 20);
    }

    #[tokio::test]
    async fn zero_k_returns_nothing() {
        let store = filled(10, 50).await;
        assert!(
            store
                .search(&NS, &vector_for(1), 0)
                .await
                .unwrap()
                .is_empty()
        );
    }
}
