//! Lookup latency at 100k entries, excluding real embedding and upstream time.
//!
//! The fake embedder only clones a precomputed vector. Guards are not in the
//! pipeline yet, so these numbers will rise when they are.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use async_trait::async_trait;
use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::Value;
use tokio::runtime::{Builder, Runtime};
use veracache::{
    Cache, CacheConfig, CacheRequest, ChatMessage, Embedder, Error, ResponseMeta, Result,
    SetOptions, memory_cache,
};

const ENTRIES: usize = 100_000;
const CLUSTERS: usize = 1_000;
const QUERIES: usize = 1_000;
const DIM: usize = 384;

struct Rng(u64);

impl Rng {
    fn signed(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn near(&mut self, center: &[f32], spread: f32) -> Vec<f32> {
        center.iter().map(|x| x + spread * self.signed()).collect()
    }
}

struct FixedEmbedder(HashMap<String, Vec<f32>>);

#[async_trait]
impl Embedder for FixedEmbedder {
    fn id(&self) -> &str {
        "fixed"
    }

    fn dim(&self) -> usize {
        DIM
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts
            .iter()
            .map(|text| {
                self.0
                    .get(text)
                    .cloned()
                    .ok_or_else(|| Error::Embed("unknown text".into()))
            })
            .collect()
    }
}

fn messages(text: String) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".into(),
        content: text,
    }]
}

fn request(messages: &[ChatMessage]) -> CacheRequest<'_> {
    CacheRequest {
        tenant: "bench",
        model: "model",
        messages,
        tools: &Value::Null,
        response_format: &Value::Null,
        version_tag: "",
    }
}

fn filled_cache(runtime: &Runtime) -> (Cache, Vec<Vec<ChatMessage>>) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let centers: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| (0..DIM).map(|_| rng.signed()).collect())
        .collect();

    let mut vectors = HashMap::new();
    for i in 0..ENTRIES {
        vectors.insert(
            format!("stored {i}"),
            rng.near(&centers[i % CLUSTERS], 0.02),
        );
    }
    for i in 0..QUERIES {
        vectors.insert(format!("query {i}"), rng.near(&centers[i % CLUSTERS], 0.02));
    }

    let config = CacheConfig {
        threshold: 0.5,
        ..CacheConfig::default()
    };
    let cache = memory_cache(Arc::new(FixedEmbedder(vectors)), config);
    runtime.block_on(async {
        for i in 0..ENTRIES {
            let stored = messages(format!("stored {i}"));
            let response = vec![0; 512];
            let meta = ResponseMeta::default();
            cache
                .set(&request(&stored), response, meta, SetOptions::default())
                .await;
        }
    });
    assert_eq!(cache.error_count(), 0);

    let queries = (0..QUERIES)
        .map(|i| messages(format!("query {i}")))
        .collect();
    (cache, queries)
}

fn lookup(c: &mut Criterion) {
    let runtime = Builder::new_current_thread().build().unwrap();
    let (cache, queries) = filled_cache(&runtime);

    let stored = messages("stored 12345".into());
    assert!(runtime.block_on(cache.get(&request(&stored))).is_some());
    c.bench_function("exact_hit_100k", |b| {
        b.iter(|| runtime.block_on(cache.get(black_box(&request(&stored)))));
    });

    let mut next = queries.iter().cycle();
    assert!(runtime.block_on(cache.get(&request(&queries[0]))).is_some());
    c.bench_function("semantic_hit_100k", |b| {
        b.iter(|| {
            let query = next.next().unwrap();
            runtime.block_on(cache.get(black_box(&request(query))))
        });
    });
}

criterion_group!(benches, lookup);
criterion_main!(benches);
