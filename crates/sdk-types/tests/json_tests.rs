use crate::ai_sdk_types::json::parse_json_loose;

#[test]
fn loose_parses_clean_object() {
    let s = r#"{"a":1,"b":"x"}"#;
    let v = parse_json_loose(s).unwrap();
    assert_eq!(v["a"], 1);
    assert_eq!(v["b"], "x");
}

#[test]
fn loose_ignores_trailing_marker() {
    let s = r#"{"a":1}<|tool_call_end|>"#;
    let v = parse_json_loose(s).unwrap();
    assert_eq!(v["a"], 1);
}
