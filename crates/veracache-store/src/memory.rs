use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::SystemTime;

use async_trait::async_trait;
use veracache_core::{
    CacheEntry, Candidate, DeleteFilter, EntryId, Error, Feedback, Namespace, Result, VectorStore,
};

/// Source of the current time. Tests inject one to control expiry.
pub type Clock = Arc<dyn Fn() -> SystemTime + Send + Sync>;

type EntryKey = (Namespace, [u8; 32]);

#[derive(Default)]
struct State {
    namespaces: HashMap<Namespace, HashMap<[u8; 32], CacheEntry>>,
    ids: HashMap<EntryId, EntryKey>,
}

/// An in-process vector store. Search is brute force within a namespace.
pub struct MemoryStore {
    state: RwLock<State>,
    clock: Clock,
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
        }
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

        if let Some(previous) = state.ids.insert(entry.id, key) {
            if previous != key {
                if let Some(entries) = state.namespaces.get_mut(&previous.0) {
                    entries.remove(&previous.1);
                }
            }
        }

        let id = entry.id;
        let replaced = state
            .namespaces
            .entry(entry.namespace)
            .or_default()
            .insert(entry.key_hash, entry);
        if let Some(old) = replaced {
            if old.id != id {
                state.ids.remove(&old.id);
            }
        }
        Ok(())
    }

    async fn search(&self, ns: &Namespace, vector: &[f32], k: usize) -> Result<Vec<Candidate>> {
        let now = (self.clock)();
        let state = self.read()?;
        let Some(entries) = state.namespaces.get(ns) else {
            return Ok(Vec::new());
        };

        let mut candidates = Vec::new();
        for entry in entries.values().filter(|e| !is_expired(e, now)) {
            if entry.embedding.len() != vector.len() {
                return Err(Error::Store("vector dimension mismatch".into()));
            }
            let score: f32 = entry.embedding.iter().zip(vector).map(|(a, b)| a * b).sum();
            if score.is_finite() {
                candidates.push(Candidate {
                    entry: entry.clone(),
                    score,
                });
            }
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        candidates.truncate(k);
        Ok(candidates)
    }

    async fn get_exact(&self, ns: &Namespace, key_hash: &[u8; 32]) -> Result<Option<CacheEntry>> {
        let now = (self.clock)();
        let state = self.read()?;
        Ok(state
            .namespaces
            .get(ns)
            .and_then(|entries| entries.get(key_hash))
            .filter(|entry| !is_expired(entry, now))
            .cloned())
    }

    async fn delete(&self, filter: DeleteFilter) -> Result<u64> {
        let mut guard = self.write()?;
        let State { namespaces, ids } = &mut *guard;
        let mut deleted = 0;
        for entries in namespaces.values_mut() {
            entries.retain(|_, entry| {
                let matched = matches(&filter, entry);
                if matched {
                    ids.remove(&entry.id);
                    deleted += 1;
                }
                !matched
            });
        }
        namespaces.retain(|_, entries| !entries.is_empty());
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
            .and_then(|entries| entries.get_mut(&key))
            .ok_or_else(|| Error::Store("entry not found".into()))?;
        if !fb.correct {
            entry.stats.false_hit_reports += 1;
        }
        Ok(())
    }
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
    use super::*;

    #[tokio::test]
    async fn passes_conformance() {
        crate::conformance::run(MemoryStore::with_clock).await;
    }
}
