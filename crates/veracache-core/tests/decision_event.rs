//! Lives in its own binary because `tracing` caches callsite interest for the
//! whole process, so a parallel test could hide the event from this subscriber.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use serde_json::Value;
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};
use veracache_core::{
    Analyzed, Analyzer, Cache, CacheConfig, CacheEntry, CacheRequest, Candidate, ChatMessage,
    DeleteFilter, Embedder, EntryId, Error, Feedback, Namespace, Result, VectorStore,
};

#[derive(Clone, Default)]
struct DecisionEvents(Arc<AtomicU64>);

impl Subscriber for DecisionEvents {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if event.metadata().target() == "veracache::decision" {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

struct DownEmbedder;

#[async_trait]
impl Embedder for DownEmbedder {
    fn id(&self) -> &str {
        "down"
    }

    fn dim(&self) -> usize {
        2
    }

    async fn embed(&self, _: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(Error::Embed("down".into()))
    }
}

struct EmptyStore;

#[async_trait]
impl VectorStore for EmptyStore {
    async fn upsert(&self, _: CacheEntry) -> Result<()> {
        Ok(())
    }

    async fn search(&self, _: &Namespace, _: &[f32], _: usize) -> Result<Vec<Candidate>> {
        Ok(Vec::new())
    }

    async fn get_exact(&self, _: &Namespace, _: &[u8; 32]) -> Result<Option<CacheEntry>> {
        Ok(None)
    }

    async fn delete(&self, _: DeleteFilter) -> Result<u64> {
        Ok(0)
    }

    async fn record_feedback(&self, _: EntryId, _: Feedback) -> Result<()> {
        Ok(())
    }
}

struct PlainAnalyzer;

impl Analyzer for PlainAnalyzer {
    fn feature_version(&self) -> u32 {
        0
    }

    fn analyze(&self, _: &str) -> Result<Analyzed> {
        Ok(Analyzed::default())
    }
}

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
    }
}

fn request(messages: &[ChatMessage]) -> CacheRequest<'_> {
    CacheRequest {
        tenant: "tenant",
        model: "model",
        messages,
        tools: &Value::Null,
        response_format: &Value::Null,
        version_tag: "",
        route: "",
    }
}

#[tokio::test]
async fn every_lookup_emits_one_decision_event() {
    let events = DecisionEvents::default();
    let _guard = tracing::subscriber::set_default(events.clone());
    let cache = Cache::new(
        Arc::new(DownEmbedder),
        Arc::new(EmptyStore),
        Arc::new(PlainAnalyzer),
        Vec::new(),
        CacheConfig::default(),
    );

    let single = [msg("user", "query")];
    cache.get(&request(&single)).await;
    assert_eq!(events.0.load(Ordering::SeqCst), 1);

    let multi = [msg("user", "a"), msg("assistant", "b"), msg("user", "c")];
    cache.get(&request(&multi)).await;
    assert_eq!(events.0.load(Ordering::SeqCst), 2);
}
