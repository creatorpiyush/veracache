use veracache_core::{Analyzed, Guard, RejectReason};

const DEFAULT_MAX_RATIO: f32 = 5.0;

/// Rejects a candidate when one prompt is much longer than the other, which
/// usually means it adds constraints the shorter one lacks.
#[derive(Debug, Clone, Copy)]
pub struct LengthRatioGuard {
    max_ratio: f32,
}

impl LengthRatioGuard {
    /// Rejects when the longer prompt has over `max_ratio` times the tokens.
    pub fn new(max_ratio: f32) -> Self {
        Self { max_ratio }
    }
}

impl Default for LengthRatioGuard {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_RATIO)
    }
}

impl Guard for LengthRatioGuard {
    fn name(&self) -> &'static str {
        "length_ratio"
    }

    fn check(&self, query: &Analyzed, candidate: &Analyzed) -> Result<(), RejectReason> {
        let tokens = |side: &Analyzed| side.text.split_whitespace().count();
        let (a, b) = (tokens(query), tokens(candidate));
        let (shorter, longer) = (a.min(b) as f32, a.max(b) as f32);
        if longer <= shorter * self.max_ratio {
            Ok(())
        } else {
            Err(RejectReason::new(self.name(), "lengths differ"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(guard: LengthRatioGuard, a: &str, b: &str) -> Result<(), RejectReason> {
        let analyzed = |text: &str| Analyzed {
            text: text.into(),
            ..Analyzed::default()
        };
        guard.check(&analyzed(a), &analyzed(b))
    }

    #[test]
    fn similar_lengths_pass() {
        assert!(check(LengthRatioGuard::default(), "a b c", "a b c d e").is_ok());
    }

    #[test]
    fn the_band_boundary_is_inclusive() {
        let guard = LengthRatioGuard::new(2.0);
        assert!(check(guard, "a b", "a b c d").is_ok());
        assert!(check(guard, "a b", "a b c d e").is_err());
    }

    #[test]
    fn works_in_either_direction() {
        let guard = LengthRatioGuard::new(2.0);
        assert!(check(guard, "a b c d e", "a b").is_err());
    }

    #[test]
    fn empty_text_only_matches_empty_text() {
        let guard = LengthRatioGuard::default();
        assert!(check(guard, "", "").is_ok());
        assert!(check(guard, "", "a").is_err());
    }

    #[test]
    fn rejection_names_the_guard() {
        let reason = check(LengthRatioGuard::new(1.0), "a", "a b").unwrap_err();
        assert_eq!(reason.guard, "length_ratio");
    }
}
