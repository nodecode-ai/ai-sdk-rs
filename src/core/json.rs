use serde_json::Value;

/// Recursively remove all object fields whose value is `Value::Null`.
///
/// - Only removes object entries with null values.
/// - Does NOT remove array elements that are null (to avoid changing indices/semantics).
/// - Recurses into nested objects and arrays to prune null fields in objects.
pub fn prune_null_fields(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // First, recurse into children so we can also prune nested structures
            for v in map.values_mut() {
                prune_null_fields(v);
            }
            // Then remove null-valued entries
            map.retain(|_, v| !matches!(v, Value::Null));
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                prune_null_fields(v);
            }
        }
        _ => {}
    }
}

/// Return a cloned JSON value with all object fields that have `null` values removed.
pub fn without_null_fields(value: &Value) -> Value {
    let mut cloned = value.clone();
    prune_null_fields(&mut cloned);
    cloned
}
