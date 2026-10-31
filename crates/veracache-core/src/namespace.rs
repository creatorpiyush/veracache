use serde_json::Value;

use crate::Namespace;

/// Version of the namespace encoding. Bump it when the encoding changes.
pub const NAMESPACE_FORMAT_VERSION: u32 = 1;

/// The fields that isolate one namespace from another.
#[derive(Debug, Clone, Copy)]
pub struct NamespaceParts<'a> {
    /// Tenant or user scope.
    pub tenant: &'a str,
    /// Model name.
    pub model: &'a str,
    /// Embedder identifier.
    pub embedder_id: &'a str,
    /// System prompt text.
    pub system_prompt: &'a str,
    /// Tools or function schema. `Value::Null` when absent.
    pub tools: &'a Value,
    /// Response format. `Value::Null` when absent.
    pub response_format: &'a Value,
    /// From [`crate::context_fingerprint`].
    pub context_fingerprint: &'a [u8; 32],
    /// User-supplied cache version tag.
    pub version_tag: &'a str,
}

impl NamespaceParts<'_> {
    /// Builds the namespace. Equal parts give equal namespaces across releases.
    pub fn build(&self) -> Namespace {
        let system_hash = blake3::hash(self.system_prompt.as_bytes());
        let tools_hash = hash_json(self.tools);
        let format_hash = hash_json(self.response_format);

        let mut hasher = blake3::Hasher::new();
        hasher.update(&NAMESPACE_FORMAT_VERSION.to_le_bytes());
        update_prefixed(&mut hasher, self.tenant.as_bytes());
        update_prefixed(&mut hasher, self.model.as_bytes());
        update_prefixed(&mut hasher, self.embedder_id.as_bytes());
        update_prefixed(&mut hasher, system_hash.as_bytes());
        update_prefixed(&mut hasher, tools_hash.as_bytes());
        update_prefixed(&mut hasher, format_hash.as_bytes());
        update_prefixed(&mut hasher, self.context_fingerprint);
        update_prefixed(&mut hasher, self.version_tag.as_bytes());
        Namespace(*hasher.finalize().as_bytes())
    }
}

/// Writes a length prefix then the bytes, so adjacent fields cannot collide.
pub(crate) fn update_prefixed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_json(value: &Value) -> blake3::Hash {
    let mut canonical = String::new();
    write_canonical(value, &mut canonical);
    blake3::hash(canonical.as_bytes())
}

// Sorts object keys here instead of relying on serde_json's map order, which
// changes if any crate in the build enables its `preserve_order` feature.
fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::from(key.as_str()).to_string());
                out.push(':');
                write_canonical(&map[key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    fn parts<'a>(tools: &'a Value, format: &'a Value) -> NamespaceParts<'a> {
        NamespaceParts {
            tenant: "tenant",
            model: "model",
            embedder_id: "embedder",
            system_prompt: "system",
            tools,
            response_format: format,
            context_fingerprint: &[7; 32],
            version_tag: "v1",
        }
    }

    // Changing this hash means every deployed cache goes cold. Bump
    // NAMESPACE_FORMAT_VERSION on purpose; never just update the value.
    #[test]
    fn encoding_is_pinned() {
        let tools = json!({"b": 1, "a": [true, null]});
        let namespace = parts(&tools, &Value::Null).build();
        assert_eq!(
            blake3::Hash::from(namespace.0).to_hex().as_str(),
            "9f49a483e63c196772f24d6eff77f326a09f4e80bd6df26c10469732ca86e2b0"
        );
    }

    #[test]
    fn every_field_changes_the_namespace() {
        let (tools, format) = (Value::Null, Value::Null);
        let base = parts(&tools, &format);
        let other_tools = json!(["t"]);
        let other_format = json!("json");
        let variants = [
            NamespaceParts {
                tenant: "x",
                ..base
            },
            NamespaceParts { model: "x", ..base },
            NamespaceParts {
                embedder_id: "x",
                ..base
            },
            NamespaceParts {
                system_prompt: "x",
                ..base
            },
            NamespaceParts {
                tools: &other_tools,
                ..base
            },
            NamespaceParts {
                response_format: &other_format,
                ..base
            },
            NamespaceParts {
                context_fingerprint: &[8; 32],
                ..base
            },
            NamespaceParts {
                version_tag: "x",
                ..base
            },
        ];
        for variant in variants {
            assert_ne!(variant.build(), base.build());
        }
    }

    #[test]
    fn nested_key_order_does_not_change_the_namespace() {
        let a: Value = serde_json::from_str(r#"{"x":{"p":1,"q":2},"y":[{"m":1,"n":2}]}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"y":[{"n":2,"m":1}],"x":{"q":2,"p":1}}"#).unwrap();
        assert_eq!(
            parts(&a, &Value::Null).build(),
            parts(&b, &Value::Null).build()
        );
    }

    #[test]
    fn array_order_changes_the_namespace() {
        let (a, b) = (json!([1, 2]), json!([2, 1]));
        assert_ne!(
            parts(&a, &Value::Null).build(),
            parts(&b, &Value::Null).build()
        );
    }

    proptest! {
        #[test]
        fn adjacent_fields_cannot_collide(
            tenant in "[a-c]{0,6}",
            model in "[a-c]{0,6}",
            split in 0usize..12,
        ) {
            let joined = format!("{tenant}{model}");
            let split = split.min(joined.len());
            let (moved_tenant, moved_model) = joined.split_at(split);
            prop_assume!(moved_tenant != tenant);

            let (tools, format) = (Value::Null, Value::Null);
            let base = parts(&tools, &format);
            let a = NamespaceParts { tenant: &tenant, model: &model, ..base };
            let b = NamespaceParts { tenant: moved_tenant, model: moved_model, ..base };
            prop_assert_ne!(a.build(), b.build());
        }
    }
}
