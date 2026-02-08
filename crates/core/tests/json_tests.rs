use crate::ai_sdk_core::json::{prune_null_fields, without_null_fields};
use serde_json::json;

#[test]
fn removes_null_object_fields_top_level() {
    let mut v = json!({"a": null, "b": 1, "c": null});
    prune_null_fields(&mut v);
    assert_eq!(v, json!({"b": 1}));
}

#[test]
fn removes_nested_null_fields_but_keeps_array_nulls() {
    let input = json!({
        "a": null,
        "b": 1,
        "c": { "d": null, "e": 2, "f": {"g": null, "h": 3} },
        "arr": [ {"k": null, "m": 5}, null, [{"z": null}, {"z": 7}] ]
    });
    let out = without_null_fields(&input);
    assert_eq!(
        out,
        json!({
            "b": 1,
            "c": { "e": 2, "f": {"h": 3} },
            "arr": [ {"m": 5}, null, [{}, {"z": 7}] ]
        })
    );
}
