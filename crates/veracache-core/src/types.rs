use std::fmt;
use std::time::{Duration, SystemTime};

/// The isolation scope for cache entries. A hit never crosses a namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Namespace(pub [u8; 32]);

/// Identifies a cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryId(pub u128);

/// A prompt plus the features guards compare, extracted once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Analyzed {
    /// The prompt text.
    pub text: String,
    /// Version of the extraction rules that produced these features.
    pub feature_version: u32,
    /// Numbers, sorted so that equal multisets compare equal.
    pub numbers: Vec<String>,
    /// Normalized dates and relative-date words, sorted.
    pub dates: Vec<String>,
    /// Entity spans, sorted.
    pub entities: Vec<String>,
    /// Negation words, sorted.
    pub negations: Vec<String>,
    /// Directional and relational words, in prompt order.
    pub directions: Vec<String>,
    /// Quoted strings and code spans, sorted.
    pub quotes: Vec<String>,
}

/// Why a guard rejected a candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectReason {
    /// Name of the guard that rejected.
    pub guard: &'static str,
    /// What differed.
    pub detail: String,
}

impl RejectReason {
    /// Creates a reject reason.
    pub fn new(guard: &'static str, detail: impl Into<String>) -> Self {
        Self {
            guard,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.guard, self.detail)
    }
}

/// Metadata about a cached upstream response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResponseMeta {
    /// Model that produced the response.
    pub model: String,
    /// Finish reason reported by upstream.
    pub finish_reason: String,
    /// Prompt tokens, when upstream reports them.
    pub prompt_tokens: Option<u32>,
    /// Completion tokens, when upstream reports them.
    pub completion_tokens: Option<u32>,
}

/// Usage counters for a cache entry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EntryStats {
    /// Times this entry was served.
    pub hits: u64,
    /// Times this entry was reported as a false hit.
    pub false_hit_reports: u64,
    /// When this entry was last served.
    pub last_hit_at: Option<SystemTime>,
}

/// A stored prompt and response.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheEntry {
    /// Entry identifier.
    pub id: EntryId,
    /// Namespace this entry belongs to.
    pub namespace: Namespace,
    /// Hash of the normalized prompt.
    pub key_hash: [u8; 32],
    /// Prompt text. Absent in no-text mode.
    pub prompt_text: Option<String>,
    /// Extracted features. Absent in no-text mode.
    pub analyzed: Option<Analyzed>,
    /// Normalized embedding vector.
    pub embedding: Vec<f32>,
    /// Original upstream response bytes.
    pub response: Vec<u8>,
    /// Response metadata.
    pub response_meta: ResponseMeta,
    /// When this entry was stored.
    pub created_at: SystemTime,
    /// Time to live. `None` means no expiry.
    pub ttl: Option<Duration>,
    /// Invalidation tags.
    pub tags: Vec<String>,
    /// Invalidation version.
    pub version: Option<String>,
    /// Usage counters.
    pub stats: EntryStats,
    /// Learned threshold for this entry, when one exists.
    pub threshold_override: Option<f32>,
}

/// A search result: an entry that is not yet accepted as a hit.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The stored entry.
    pub entry: CacheEntry,
    /// Dot-product similarity to the query.
    pub score: f32,
}

/// Selects entries to delete.
///
/// All set fields must match. An empty filter matches nothing, so a default
/// value can never clear the store.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeleteFilter {
    /// Match entries in this namespace.
    pub namespace: Option<Namespace>,
    /// Match entries with this tag.
    pub tag: Option<String>,
    /// Match entries with this version.
    pub version: Option<String>,
    /// Match entries created before this time.
    pub created_before: Option<SystemTime>,
}

/// A report on whether a served entry was correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Feedback {
    /// Whether the served response was correct.
    pub correct: bool,
}
