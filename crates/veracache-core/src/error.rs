use std::fmt;

/// Result type used across veracache.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure in a pluggable cache component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The embedder failed.
    Embed(String),
    /// The vector store failed.
    Store(String),
    /// The analyzer failed.
    Analyze(String),
    /// The reranker failed.
    Rerank(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embed(detail) => write!(f, "embedder error: {detail}"),
            Self::Store(detail) => write!(f, "store error: {detail}"),
            Self::Analyze(detail) => write!(f, "analyzer error: {detail}"),
            Self::Rerank(detail) => write!(f, "reranker error: {detail}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_failed_component() {
        let cases = [
            (Error::Embed("x".into()), "embedder error: x"),
            (Error::Store("x".into()), "store error: x"),
            (Error::Analyze("x".into()), "analyzer error: x"),
            (Error::Rerank("x".into()), "reranker error: x"),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }
}
