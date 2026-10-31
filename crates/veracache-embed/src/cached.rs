use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use veracache_core::{Embedder, Error, Result};

#[derive(Default)]
struct Lru {
    entries: HashMap<[u8; 32], (Vec<f32>, u64)>,
    tick: u64,
}

impl Lru {
    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    fn get(&mut self, key: &[u8; 32]) -> Option<Vec<f32>> {
        let tick = self.next_tick();
        let (vector, used) = self.entries.get_mut(key)?;
        *used = tick;
        Some(vector.clone())
    }

    // ponytail: eviction scans every entry, O(n). Embedding costs milliseconds,
    // so this is cheap below ~100k entries; use a linked LRU if that changes.
    fn insert(&mut self, key: [u8; 32], vector: Vec<f32>, capacity: usize) {
        if capacity == 0 {
            return;
        }
        if self.entries.len() >= capacity && !self.entries.contains_key(&key) {
            let oldest = self.entries.iter().min_by_key(|(_, (_, used))| *used);
            if let Some(oldest) = oldest.map(|(key, _)| *key) {
                self.entries.remove(&oldest);
            }
        }
        let tick = self.next_tick();
        self.entries.insert(key, (vector, tick));
    }
}

/// Wraps an embedder with an LRU keyed by text hash.
///
/// `Cache::get` and `Cache::set` both embed the prompt, so this makes a miss
/// cost one real embedding, not two.
pub struct CachedEmbedder<E> {
    inner: E,
    capacity: usize,
    lru: Mutex<Lru>,
}

impl<E: Embedder> CachedEmbedder<E> {
    /// Wraps `inner`, keeping up to `capacity` recent vectors.
    pub fn new(inner: E, capacity: usize) -> Self {
        Self {
            inner,
            capacity,
            lru: Mutex::default(),
        }
    }

    // The LRU holds only finished vectors, so a poisoned lock is safe to reuse.
    fn lru(&self) -> MutexGuard<'_, Lru> {
        self.lru.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl<E: Embedder> Embedder for CachedEmbedder<E> {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let keys: Vec<_> = texts
            .iter()
            .map(|text| *blake3::hash(text.as_bytes()).as_bytes())
            .collect();

        let mut vectors: Vec<_> = {
            let mut lru = self.lru();
            keys.iter().map(|key| lru.get(key)).collect()
        };
        let missing: Vec<usize> = (0..texts.len()).filter(|&i| vectors[i].is_none()).collect();
        if missing.is_empty() {
            return Ok(vectors.into_iter().flatten().collect());
        }

        let batch: Vec<String> = missing.iter().map(|&i| texts[i].clone()).collect();
        let embedded = self.inner.embed(&batch).await?;
        if embedded.len() != batch.len() {
            return Err(Error::Embed("wrong number of vectors".into()));
        }

        let mut lru = self.lru();
        for (i, vector) in missing.into_iter().zip(embedded) {
            lru.insert(keys[i], vector.clone(), self.capacity);
            vectors[i] = Some(vector);
        }
        Ok(vectors.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

    /// Embeds a text as its length, and counts the texts it sees.
    #[derive(Default)]
    struct Counting {
        calls: AtomicUsize,
        failing: AtomicBool,
        short: AtomicBool,
    }

    #[async_trait]
    impl Embedder for &Counting {
        fn id(&self) -> &str {
            "counting"
        }

        fn dim(&self) -> usize {
            1
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if self.failing.load(Ordering::SeqCst) {
                return Err(Error::Embed("down".into()));
            }
            self.calls.fetch_add(texts.len(), Ordering::SeqCst);
            let take = texts.len() - usize::from(self.short.load(Ordering::SeqCst));
            Ok(texts[..take]
                .iter()
                .map(|text| vec![text.len() as f32])
                .collect())
        }
    }

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|text| text.to_string()).collect()
    }

    #[tokio::test]
    async fn repeated_text_embeds_once() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 10);

        let first = cached.embed(&texts(&["a"])).await.unwrap();
        let second = cached.embed(&texts(&["a"])).await.unwrap();

        assert_eq!(first, second);
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn partial_hits_keep_input_order() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 10);
        cached.embed(&texts(&["bb"])).await.unwrap();

        let vectors = cached.embed(&texts(&["a", "bb", "ccc"])).await.unwrap();

        assert_eq!(vectors, [vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(inner.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn evicts_the_least_recently_used() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 2);
        cached.embed(&texts(&["a", "bb"])).await.unwrap();
        cached.embed(&texts(&["a"])).await.unwrap();
        cached.embed(&texts(&["ccc"])).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::SeqCst), 3);

        cached.embed(&texts(&["a"])).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::SeqCst), 3);

        cached.embed(&texts(&["bb"])).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn zero_capacity_stores_nothing() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 0);
        cached.embed(&texts(&["a"])).await.unwrap();
        cached.embed(&texts(&["a"])).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn failures_are_not_cached() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 10);

        inner.failing.store(true, Ordering::SeqCst);
        assert!(cached.embed(&texts(&["a"])).await.is_err());

        inner.failing.store(false, Ordering::SeqCst);
        assert!(cached.embed(&texts(&["a"])).await.is_ok());
    }

    #[tokio::test]
    async fn wrong_vector_count_is_an_error() {
        let inner = Counting::default();
        inner.short.store(true, Ordering::SeqCst);
        let cached = CachedEmbedder::new(&inner, 10);
        assert!(cached.embed(&texts(&["a", "bb"])).await.is_err());
    }

    #[tokio::test]
    async fn forwards_id_and_dim() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 10);
        assert_eq!(cached.id(), "counting");
        assert_eq!(cached.dim(), 1);
    }

    #[tokio::test]
    async fn empty_input_calls_nothing() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 10);
        assert!(cached.embed(&[]).await.unwrap().is_empty());
        assert_eq!(inner.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn duplicates_in_one_batch_both_get_vectors() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 10);

        let vectors = cached.embed(&texts(&["aa", "aa"])).await.unwrap();

        assert_eq!(vectors, [vec![2.0], vec![2.0]]);
    }

    #[tokio::test]
    async fn different_texts_do_not_share_a_vector() {
        let inner = Counting::default();
        let cached = CachedEmbedder::new(&inner, 10);
        cached.embed(&texts(&["a"])).await.unwrap();

        let vectors = cached.embed(&texts(&["bb"])).await.unwrap();

        assert_eq!(vectors, [vec![2.0]]);
    }

    #[test]
    fn updating_a_key_at_capacity_evicts_nothing() {
        let mut lru = Lru::default();
        lru.insert([1; 32], vec![1.0], 2);
        lru.insert([2; 32], vec![2.0], 2);

        lru.insert([1; 32], vec![3.0], 2);

        assert_eq!(lru.get(&[1; 32]), Some(vec![3.0]));
        assert_eq!(lru.get(&[2; 32]), Some(vec![2.0]));
    }

    #[test]
    fn missing_key_returns_none() {
        assert_eq!(Lru::default().get(&[1; 32]), None);
    }
}
