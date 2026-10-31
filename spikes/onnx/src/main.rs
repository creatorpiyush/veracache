//! Checks that fastembed builds and runs on the current target.

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

fn main() {
    let mut model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGESmallENV15))
        .expect("model loads");
    let vectors = model
        .embed(vec!["what is the capital of France?"], None)
        .expect("embedding succeeds");
    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), 384);
    println!("ok: dim {}", vectors[0].len());
}
