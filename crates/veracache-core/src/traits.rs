use async_trait::async_trait;

use crate::{
    Analyzed, CacheEntry, Candidate, DeleteFilter, EntryId, Feedback, Namespace, RejectReason,
    Result,
};

/// Turns text into vectors.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Stable model identifier. Part of the namespace.
    fn id(&self) -> &str;
    /// Length of each vector.
    fn dim(&self) -> usize;
    /// Embeds a batch, returning one vector per input, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Stores cache entries and searches them within a namespace.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Inserts or replaces an entry.
    async fn upsert(&self, entry: CacheEntry) -> Result<()>;
    /// Returns up to `k` unexpired entries from `ns`, best score first.
    async fn search(&self, ns: &Namespace, vector: &[f32], k: usize) -> Result<Vec<Candidate>>;
    /// Returns the unexpired entry in `ns` with this key hash.
    async fn get_exact(&self, ns: &Namespace, key_hash: &[u8; 32]) -> Result<Option<CacheEntry>>;
    /// Deletes matching entries and returns how many were removed.
    async fn delete(&self, filter: DeleteFilter) -> Result<u64>;
    /// Records feedback against an entry.
    async fn record_feedback(&self, id: EntryId, fb: Feedback) -> Result<()>;
}

/// Extracts guard features from a prompt.
pub trait Analyzer: Send + Sync {
    /// Version of the extraction rules. Bump it when features change, so
    /// entries analyzed by older rules are rejected instead of misread.
    fn feature_version(&self) -> u32;
    /// Analyzes one prompt.
    fn analyze(&self, text: &str) -> Result<Analyzed>;
}

/// A rule that can veto a candidate.
pub trait Guard: Send + Sync {
    /// Guard name, used in traces and metrics.
    fn name(&self) -> &'static str;
    /// `Ok(())` means no objection. `Err` rejects this candidate.
    fn check(
        &self,
        query: &Analyzed,
        candidate: &Analyzed,
    ) -> std::result::Result<(), RejectReason>;
}

/// Scores a query and candidate more precisely than vector similarity.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Returns a relevance score.
    async fn score(&self, query: &str, candidate: &str) -> Result<f32>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // The pipeline holds these as trait objects, so each must stay dyn compatible.
    #[test]
    fn traits_are_dyn_compatible() {
        fn accepts(
            _: Option<&dyn Embedder>,
            _: Option<&dyn VectorStore>,
            _: Option<&dyn Analyzer>,
            _: Option<&dyn Guard>,
            _: Option<&dyn Reranker>,
        ) {
        }
        accepts(None, None, None, None, None);
    }
}
