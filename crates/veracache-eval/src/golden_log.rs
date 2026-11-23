//! Turns golden pairs into a replayable log with labels.

use serde_json::Value;
use veracache::ChatMessage;

use crate::{Label, Labels, LogRow, Pair};

/// Turns golden pairs into a log and labels, so the golden suite can be
/// replayed like real traffic.
///
/// Each pair becomes two rows, `<id>/a` then `<id>/b`, in a tenant of its
/// own, so the second prompt can only match the first. Earlier turns become
/// alternating user and assistant messages. Responses are empty: every pair
/// is labelled, so no response is compared.
pub fn golden_log(pairs: &[Pair]) -> (Vec<LogRow>, Labels) {
    let mut rows = Vec::with_capacity(pairs.len() * 2);
    let mut labels = Labels::default();
    for pair in pairs {
        let sides = [
            ("a", &pair.a, &pair.a_history),
            ("b", &pair.b, &pair.b_history),
        ];
        for (side, prompt, history) in sides {
            let mut messages: Vec<ChatMessage> = history
                .iter()
                .flatten()
                .enumerate()
                .map(|(turn, content)| ChatMessage {
                    role: if turn % 2 == 0 { "user" } else { "assistant" }.into(),
                    content: content.clone(),
                })
                .collect();
            messages.push(ChatMessage {
                role: "user".into(),
                content: prompt.clone(),
            });
            rows.push(LogRow {
                id: format!("{}/{side}", pair.id),
                tenant: pair.id.clone(),
                model: String::new(),
                messages,
                tools: Value::Null,
                response_format: Value::Null,
                response: Vec::new(),
                usage: None,
            });
        }
        let (a, b) = (format!("{}/a", pair.id), format!("{}/b", pair.id));
        labels.insert(&a, &b, pair.label == Label::Hit);
    }
    (rows, labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Split;

    fn pair(id: &str, label: Label, history: Option<Vec<String>>) -> Pair {
        Pair {
            id: id.into(),
            category: "number".into(),
            label,
            split: Split::Tuning,
            a: "stored prompt".into(),
            b: "query prompt".into(),
            a_history: history.clone(),
            b_history: history,
        }
    }

    #[test]
    fn each_pair_becomes_two_isolated_labelled_rows() {
        let history = Some(vec!["first question".into(), "first answer".into()]);
        let pairs = [
            pair("p1", Label::Hit, None),
            pair("p2", Label::Miss, history),
        ];
        let (rows, labels) = golden_log(&pairs);

        let ids: Vec<_> = rows.iter().map(|row| row.id.as_str()).collect();
        assert_eq!(ids, ["p1/a", "p1/b", "p2/a", "p2/b"]);
        assert_eq!(rows[1].tenant, "p1");
        assert_eq!(rows[2].tenant, "p2");
        assert_eq!(rows[1].messages.len(), 1);
        let roles: Vec<_> = rows[3].messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "user"]);
        assert_eq!(rows[3].messages[2].content, "query prompt");

        assert_eq!(labels.get("p1/b", "p1/a"), Some(true));
        assert_eq!(labels.get("p2/a", "p2/b"), Some(false));
    }
}
