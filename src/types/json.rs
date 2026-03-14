use serde_json::Value;

/// Parse a JSON value from a possibly noisy string.
///
/// Strategy:
/// 1) Try strict `serde_json::from_str` first.
/// 2) If it fails, scan for the first balanced JSON object/array substring
///    (handles nested structures and quotes) and parse that.
/// 3) Ignore any trailing or leading non‑JSON text (e.g., markers like
///    "<|tool_call_end|>").
pub fn parse_json_loose(s: &str) -> Option<Value> {
    // Fast path: strict parse
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return Some(v);
    }

    // Find the first opening bracket for JSON object or array
    let bytes = s.as_bytes();
    let mut start_idx: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'{' || b == b'[' {
            start_idx = Some(i);
            break;
        }
    }
    let start = match start_idx {
        Some(i) => i,
        None => return None,
    };

    // Scan forward to find the matching closing bracket using a stack.
    // Handle quoted strings and escapes to avoid counting braces inside strings.
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Br {
        Curly,
        Square,
    }

    let mut stack: Vec<Br> = Vec::new();
    // Push the first bracket
    stack.push(if bytes[start] == b'{' {
        Br::Curly
    } else {
        Br::Square
    });
    let mut in_str = false;
    let mut escape = false;
    let mut end = None;

    for (offset, &b) in bytes[start + 1..].iter().enumerate() {
        if in_str {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        } else {
            match b {
                b'"' => {
                    in_str = true;
                }
                b'{' => stack.push(Br::Curly),
                b'[' => stack.push(Br::Square),
                b'}' => {
                    if let Some(top) = stack.pop() {
                        if top != Br::Curly {
                            return None;
                        }
                        if stack.is_empty() {
                            end = Some(start + 1 + offset + 1); // inclusive end index
                            break;
                        }
                    } else {
                        return None;
                    }
                }
                b']' => {
                    if let Some(top) = stack.pop() {
                        if top != Br::Square {
                            return None;
                        }
                        if stack.is_empty() {
                            end = Some(start + 1 + offset + 1);
                            break;
                        }
                    } else {
                        return None;
                    }
                }
                _ => {}
            }
        }
    }

    let end = end?;
    let slice = &s[start..end];
    serde_json::from_str::<Value>(slice).ok()
}

/// Return a sanitized JSON fragment string from `raw` if possible.
/// If a valid first balanced JSON object/array is found and parses,
/// return its string; otherwise, return the original `raw` unchanged.
pub fn sanitize_json_fragment(raw: &str) -> String {
    match parse_json_loose(raw) {
        Some(v) => v.to_string(),
        None => raw.to_string(),
    }
}
