//! Behavior every [`VectorStore`] backend must satisfy.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use veracache_core::{
    CacheEntry, DeleteFilter, EntryId, EntryStats, Feedback, Namespace, ResponseMeta, VectorStore,
};

use crate::Clock;

const NS_A: Namespace = Namespace([1; 32]);
const NS_B: Namespace = Namespace([2; 32]);

/// A clock tests can move forward.
struct TestClock(Arc<AtomicU64>);

impl TestClock {
    fn new() -> (Self, Clock) {
        let seconds = Arc::new(AtomicU64::new(1_000));
        let reader = seconds.clone();
        let clock: Clock =
            Arc::new(move || UNIX_EPOCH + Duration::from_secs(reader.load(Ordering::SeqCst)));
        (Self(seconds), clock)
    }

    fn set(&self, seconds: u64) {
        self.0.store(seconds, Ordering::SeqCst);
    }
}

fn at(seconds: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(seconds)
}

fn entry(id: u8, namespace: Namespace, embedding: Vec<f32>) -> CacheEntry {
    CacheEntry {
        id: EntryId(id.into()),
        namespace,
        key_hash: [id; 32],
        prompt_text: Some(format!("prompt {id}")),
        analyzed: None,
        embedding,
        response: vec![id],
        response_meta: ResponseMeta::default(),
        created_at: at(1_000),
        ttl: None,
        tags: Vec::new(),
        version: None,
        stats: EntryStats::default(),
        threshold_override: None,
    }
}

/// Runs every check, giving each a fresh store from `make`.
pub async fn run<S: VectorStore>(make: impl Fn(Clock) -> S) {
    let fresh = || TestClock::new().1;

    exact_is_scoped_to_namespace(&make(fresh())).await;
    upsert_replaces_by_key_hash(&make(fresh())).await;
    search_is_scoped_ordered_and_limited(&make(fresh())).await;
    search_rejects_dimension_mismatch(&make(fresh())).await;
    delete_by_each_field(&make).await;
    empty_delete_filter_matches_nothing(&make(fresh())).await;
    feedback_requires_a_known_entry(&make(fresh())).await;

    let (time, clock) = TestClock::new();
    expired_entries_are_never_returned(&make(clock), &time).await;
}

async fn exact_is_scoped_to_namespace(store: &impl VectorStore) {
    let stored = entry(1, NS_A, vec![1.0, 0.0]);
    store.upsert(stored.clone()).await.unwrap();

    assert_eq!(
        store.get_exact(&NS_A, &stored.key_hash).await.unwrap(),
        Some(stored.clone())
    );
    assert_eq!(
        store.get_exact(&NS_B, &stored.key_hash).await.unwrap(),
        None
    );
    assert_eq!(store.get_exact(&NS_A, &[9; 32]).await.unwrap(), None);
}

async fn upsert_replaces_by_key_hash(store: &impl VectorStore) {
    let first = entry(1, NS_A, vec![1.0, 0.0]);
    let mut second = entry(2, NS_A, vec![0.0, 1.0]);
    second.key_hash = first.key_hash;
    store.upsert(first.clone()).await.unwrap();
    store.upsert(second.clone()).await.unwrap();

    assert_eq!(
        store.get_exact(&NS_A, &first.key_hash).await.unwrap(),
        Some(second)
    );
    assert_eq!(store.search(&NS_A, &[1.0, 0.0], 5).await.unwrap().len(), 1);
    assert!(
        store
            .record_feedback(first.id, Feedback { correct: false })
            .await
            .is_err()
    );
}

async fn search_is_scoped_ordered_and_limited(store: &impl VectorStore) {
    store.upsert(entry(1, NS_A, vec![1.0, 0.0])).await.unwrap();
    store.upsert(entry(2, NS_A, vec![0.0, 1.0])).await.unwrap();
    store.upsert(entry(3, NS_A, vec![0.6, 0.8])).await.unwrap();
    store.upsert(entry(4, NS_B, vec![1.0, 0.0])).await.unwrap();

    let found = store.search(&NS_A, &[0.0, 1.0], 2).await.unwrap();
    let ids: Vec<_> = found.iter().map(|c| c.entry.id).collect();
    assert_eq!(ids, [EntryId(2), EntryId(3)]);
    assert!(found[0].score > found[1].score);

    let empty = Namespace([3; 32]);
    assert!(
        store
            .search(&empty, &[1.0, 0.0], 5)
            .await
            .unwrap()
            .is_empty()
    );
}

async fn search_rejects_dimension_mismatch(store: &impl VectorStore) {
    store.upsert(entry(1, NS_A, vec![1.0, 0.0])).await.unwrap();
    assert!(store.search(&NS_A, &[1.0, 0.0, 0.0], 5).await.is_err());
}

async fn delete_by_each_field<S: VectorStore>(make: &impl Fn(Clock) -> S) {
    let tagged = |store_entry: &mut CacheEntry| {
        store_entry.tags = vec!["tag".into()];
        store_entry.version = Some("v1".into());
        store_entry.created_at = at(500);
    };
    let filters = [
        DeleteFilter {
            namespace: Some(NS_B),
            ..Default::default()
        },
        DeleteFilter {
            tag: Some("tag".into()),
            ..Default::default()
        },
        DeleteFilter {
            version: Some("v1".into()),
            ..Default::default()
        },
        DeleteFilter {
            created_before: Some(at(600)),
            ..Default::default()
        },
    ];

    for filter in filters {
        let store = make(TestClock::new().1);
        let kept = entry(1, NS_A, vec![1.0, 0.0]);
        let mut removed = entry(2, NS_B, vec![1.0, 0.0]);
        tagged(&mut removed);
        store.upsert(kept.clone()).await.unwrap();
        store.upsert(removed.clone()).await.unwrap();

        assert_eq!(store.delete(filter.clone()).await.unwrap(), 1, "{filter:?}");
        assert_eq!(
            store.get_exact(&NS_B, &removed.key_hash).await.unwrap(),
            None
        );
        assert!(
            store
                .get_exact(&NS_A, &kept.key_hash)
                .await
                .unwrap()
                .is_some()
        );
    }

    let store = make(TestClock::new().1);
    let mut both = entry(1, NS_A, vec![1.0, 0.0]);
    tagged(&mut both);
    store.upsert(both).await.unwrap();
    let mismatch = DeleteFilter {
        namespace: Some(NS_B),
        tag: Some("tag".into()),
        ..Default::default()
    };
    assert_eq!(store.delete(mismatch).await.unwrap(), 0);
}

async fn empty_delete_filter_matches_nothing(store: &impl VectorStore) {
    store.upsert(entry(1, NS_A, vec![1.0, 0.0])).await.unwrap();
    assert_eq!(store.delete(DeleteFilter::default()).await.unwrap(), 0);
}

async fn feedback_requires_a_known_entry(store: &impl VectorStore) {
    let stored = entry(1, NS_A, vec![1.0, 0.0]);
    store.upsert(stored.clone()).await.unwrap();

    store
        .record_feedback(stored.id, Feedback { correct: false })
        .await
        .unwrap();
    let found = store.get_exact(&NS_A, &stored.key_hash).await.unwrap();
    assert_eq!(found.unwrap().stats.false_hit_reports, 1);

    assert!(
        store
            .record_feedback(EntryId(99), Feedback { correct: true })
            .await
            .is_err()
    );
}

async fn expired_entries_are_never_returned(store: &impl VectorStore, time: &TestClock) {
    let mut expiring = entry(1, NS_A, vec![1.0, 0.0]);
    expiring.ttl = Some(Duration::from_secs(10));
    store.upsert(expiring.clone()).await.unwrap();
    store.upsert(entry(2, NS_A, vec![1.0, 0.0])).await.unwrap();

    time.set(1_009);
    assert!(
        store
            .get_exact(&NS_A, &expiring.key_hash)
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(store.search(&NS_A, &[1.0, 0.0], 5).await.unwrap().len(), 2);

    time.set(1_010);
    assert_eq!(
        store.get_exact(&NS_A, &expiring.key_hash).await.unwrap(),
        None
    );
    let found = store.search(&NS_A, &[1.0, 0.0], 5).await.unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].entry.id, EntryId(2));
}
