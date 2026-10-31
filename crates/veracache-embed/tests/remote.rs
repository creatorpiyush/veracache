//! `RemoteEmbedder` against a mock embeddings server.

use serde_json::{Value, json};
use veracache_core::Embedder;
use veracache_embed::{RemoteConfig, RemoteEmbedder};
use wiremock::matchers::{bearer_token, body_json, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const KEY: &str = "sk-secret-test-key";

fn embedder(server: &MockServer, dim: usize) -> RemoteEmbedder {
    RemoteEmbedder::new(RemoteConfig {
        base_url: format!("{}/v1/", server.uri()),
        api_key: Some(KEY.into()),
        model: "model".into(),
        dim,
    })
    .unwrap()
}

fn texts(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("text {i}")).collect()
}

async fn respond_with(server: &MockServer, body: Value) {
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

#[tokio::test]
async fn sends_model_input_and_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(bearer_token(KEY))
        .and(body_json(json!({"model": "model", "input": ["text 0"]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"index": 0, "embedding": [0.5, 0.25]}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let vectors = embedder(&server, 2).embed(&texts(1)).await.unwrap();
    assert_eq!(vectors, [vec![0.5, 0.25]]);
}

#[tokio::test]
async fn sends_no_key_when_none_is_set() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(|request: &Request| {
            let status = if request.headers.contains_key("authorization") {
                401
            } else {
                200
            };
            ResponseTemplate::new(status).set_body_json(json!({
                "data": [{"index": 0, "embedding": [1.0]}]
            }))
        })
        .mount(&server)
        .await;

    let keyless = RemoteEmbedder::new(RemoteConfig {
        base_url: server.uri(),
        api_key: None,
        model: "model".into(),
        dim: 1,
    })
    .unwrap();
    assert!(keyless.embed(&texts(1)).await.is_ok());
}

#[tokio::test]
async fn orders_vectors_by_index() {
    let server = MockServer::start().await;
    respond_with(
        &server,
        json!({"data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]},
        ]}),
    )
    .await;

    let vectors = embedder(&server, 1).embed(&texts(2)).await.unwrap();
    assert_eq!(vectors, [vec![1.0], vec![2.0]]);
}

#[tokio::test]
async fn splits_large_inputs_into_batches() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(|request: &Request| {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let input = body["input"].as_array().unwrap();
            let data: Vec<_> = input
                .iter()
                .enumerate()
                .map(|(index, text)| {
                    let number: f32 = text.as_str().unwrap()[5..].parse().unwrap();
                    json!({"index": index, "embedding": [number]})
                })
                .collect();
            ResponseTemplate::new(200).set_body_json(json!({ "data": data }))
        })
        .expect(2)
        .mount(&server)
        .await;

    let vectors = embedder(&server, 1).embed(&texts(300)).await.unwrap();
    assert_eq!(vectors.len(), 300);
    assert_eq!(vectors[0], [0.0]);
    assert_eq!(vectors[299], [299.0]);
}

#[tokio::test]
async fn empty_input_sends_no_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    assert!(embedder(&server, 1).embed(&[]).await.unwrap().is_empty());
}

#[tokio::test]
async fn server_error_fails_without_leaking_the_key_or_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500).set_body_string(KEY))
        .mount(&server)
        .await;

    let error = embedder(&server, 1).embed(&texts(1)).await.unwrap_err();
    assert!(!error.to_string().contains(KEY), "{error}");
}

#[tokio::test]
async fn rejects_bad_responses() {
    let cases = [
        (
            "dimension mismatch",
            json!({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}),
        ),
        ("wrong count", json!({"data": []})),
        ("no data", json!({"error": "nope"})),
        (
            "index out of range",
            json!({"data": [{"index": 5, "embedding": [1.0]}]}),
        ),
        ("missing index", json!({"data": [{"embedding": [1.0]}]})),
        (
            "non-numeric",
            json!({"data": [{"index": 0, "embedding": ["x"]}]}),
        ),
    ];
    for (name, body) in cases {
        let server = MockServer::start().await;
        respond_with(&server, body).await;
        assert!(
            embedder(&server, 1).embed(&texts(1)).await.is_err(),
            "{name}"
        );
    }

    let server = MockServer::start().await;
    respond_with(
        &server,
        json!({"data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 0, "embedding": [2.0]},
        ]}),
    )
    .await;
    assert!(embedder(&server, 1).embed(&texts(2)).await.is_err());
}

#[tokio::test]
async fn id_is_the_model_and_dim_is_configured() {
    let server = MockServer::start().await;
    let embedder = embedder(&server, 42);
    assert_eq!(embedder.id(), "model");
    assert_eq!(embedder.dim(), 42);
}

#[tokio::test]
async fn base_url_works_without_a_trailing_slash() {
    let server = MockServer::start().await;
    respond_with(&server, json!({"data": [{"index": 0, "embedding": [1.0]}]})).await;

    let embedder = RemoteEmbedder::new(RemoteConfig {
        base_url: format!("{}/v1", server.uri()),
        api_key: None,
        model: "model".into(),
        dim: 1,
    })
    .unwrap();
    assert!(embedder.embed(&texts(1)).await.is_ok());
}

#[tokio::test]
async fn unauthorized_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;

    assert!(embedder(&server, 1).embed(&texts(1)).await.is_err());
}

#[tokio::test]
async fn invalid_json_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    assert!(embedder(&server, 1).embed(&texts(1)).await.is_err());
}

#[tokio::test]
async fn a_failed_batch_fails_the_whole_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(1)
        .mount(&server)
        .await;

    assert!(embedder(&server, 1).embed(&texts(300)).await.is_err());
}

#[tokio::test]
async fn unreachable_server_is_an_error() {
    let unreachable = RemoteEmbedder::new(RemoteConfig {
        base_url: "http://127.0.0.1:9".into(),
        api_key: None,
        model: "model".into(),
        dim: 1,
    })
    .unwrap();
    assert!(unreachable.embed(&texts(1)).await.is_err());
}
