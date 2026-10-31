use crate::namespace::update_prefixed;

/// One message of a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Message role, such as `system`, `user`, or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// How prior conversation turns affect caching.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ContextMode {
    /// Cache only conversations with one user turn, plus system messages.
    #[default]
    SingleTurn,
    /// Hash up to this many prior messages into the namespace.
    LastNTurns(usize),
}

/// Fingerprints the conversation before the final user message.
///
/// Returns `None` when the conversation must not be cached. A conversation with
/// prior turns never shares a fingerprint with a standalone question, so a
/// follow-up cannot match one.
pub fn context_fingerprint(messages: &[ChatMessage], mode: ContextMode) -> Option<[u8; 32]> {
    let turns: Vec<_> = messages.iter().filter(|m| !is_system(m)).collect();
    let (last, prior) = turns.split_last()?;
    if last.role != "user" {
        return None;
    }
    let window = match mode {
        ContextMode::SingleTurn if prior.is_empty() => prior,
        ContextMode::SingleTurn => return None,
        ContextMode::LastNTurns(0) if !prior.is_empty() => return None,
        ContextMode::LastNTurns(n) => &prior[prior.len().saturating_sub(n)..],
    };

    let mut hasher = blake3::Hasher::new();
    for message in window {
        update_prefixed(&mut hasher, message.role.as_bytes());
        update_prefixed(&mut hasher, message.content.as_bytes());
    }
    Some(*hasher.finalize().as_bytes())
}

pub(crate) fn is_system(message: &ChatMessage) -> bool {
    matches!(message.role.as_str(), "system" | "developer")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    fn follow_up() -> Vec<ChatMessage> {
        vec![
            msg("user", "weather in Delhi?"),
            msg("assistant", "Sunny."),
            msg("user", "what about tomorrow?"),
        ]
    }

    #[test]
    fn single_turn_caches_one_user_message_plus_system() {
        let messages = [msg("system", "be brief"), msg("user", "hello")];
        assert!(context_fingerprint(&messages, ContextMode::SingleTurn).is_some());
    }

    #[test]
    fn single_turn_refuses_multi_turn_conversations() {
        assert_eq!(
            context_fingerprint(&follow_up(), ContextMode::SingleTurn),
            None
        );
    }

    #[test]
    fn refuses_conversations_not_ending_in_a_user_message() {
        let messages = [msg("user", "hi"), msg("assistant", "hello")];
        assert_eq!(
            context_fingerprint(&messages, ContextMode::LastNTurns(4)),
            None
        );
        assert_eq!(context_fingerprint(&[], ContextMode::SingleTurn), None);
    }

    #[test]
    fn follow_up_cannot_match_a_standalone_question() {
        let standalone = [msg("user", "what about tomorrow?")];
        for n in 0..4 {
            assert_ne!(
                context_fingerprint(&follow_up(), ContextMode::LastNTurns(n)),
                context_fingerprint(&standalone, ContextMode::LastNTurns(n)),
                "n = {n}"
            );
        }
    }

    #[test]
    fn different_histories_get_different_fingerprints() {
        let mut other = follow_up();
        other[0] = msg("user", "weather in Paris?");
        assert_ne!(
            context_fingerprint(&follow_up(), ContextMode::LastNTurns(2)),
            context_fingerprint(&other, ContextMode::LastNTurns(2)),
        );
    }

    #[test]
    fn same_history_gets_the_same_fingerprint() {
        let mode = ContextMode::LastNTurns(2);
        assert!(context_fingerprint(&follow_up(), mode).is_some());
        assert_eq!(
            context_fingerprint(&follow_up(), mode),
            context_fingerprint(&follow_up(), mode),
        );
    }

    #[test]
    fn the_final_message_is_not_part_of_the_fingerprint() {
        let mut other = follow_up();
        other[2] = msg("user", "and next week?");
        let mode = ContextMode::LastNTurns(2);
        assert_eq!(
            context_fingerprint(&follow_up(), mode),
            context_fingerprint(&other, mode)
        );
    }

    #[test]
    fn system_messages_do_not_count_as_turns() {
        let messages = [
            msg("developer", "be brief"),
            msg("user", "hello"),
            msg("system", "late instruction"),
        ];
        assert_eq!(
            context_fingerprint(&messages, ContextMode::SingleTurn),
            context_fingerprint(&[msg("user", "hi")], ContextMode::SingleTurn),
        );
    }

    #[test]
    fn only_system_messages_cannot_be_cached() {
        let messages = [msg("system", "be brief")];
        assert_eq!(
            context_fingerprint(&messages, ContextMode::SingleTurn),
            None
        );
    }

    #[test]
    fn window_ignores_turns_older_than_n() {
        let mut older = follow_up();
        older.insert(0, msg("user", "much earlier"));
        older.insert(1, msg("assistant", "ok"));
        let mode = ContextMode::LastNTurns(2);
        assert_eq!(
            context_fingerprint(&follow_up(), mode),
            context_fingerprint(&older, mode)
        );
    }

    #[test]
    fn message_boundaries_change_the_fingerprint() {
        let mode = ContextMode::LastNTurns(4);
        let one = [msg("user", "ab"), msg("assistant", "c"), msg("user", "q")];
        let two = [msg("user", "a"), msg("assistant", "bc"), msg("user", "q")];
        assert_ne!(
            context_fingerprint(&one, mode),
            context_fingerprint(&two, mode)
        );
    }
}
