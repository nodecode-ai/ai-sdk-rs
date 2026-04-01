use std::path::PathBuf;

#[path = "../benches/support/mod.rs"]
mod bench_support;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn openai_fixture_path(fixture_name: &str) -> PathBuf {
    repo_root()
        .join("crates")
        .join("providers")
        .join("openai")
        .join("tests")
        .join("fixtures")
        .join(format!("{fixture_name}.chunks.txt"))
}

fn non_empty_fixture_lines(fixture_name: &str) -> Vec<String> {
    std::fs::read_to_string(openai_fixture_path(fixture_name))
        .unwrap_or_else(|err| panic!("missing openai fixture {fixture_name}: {err}"))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn providers_with_checked_in_chunk_fixtures() -> Vec<String> {
    let providers_dir = repo_root().join("crates").join("providers");
    let mut owners = std::fs::read_dir(providers_dir)
        .expect("providers directory")
        .filter_map(|entry| {
            let entry = entry.expect("provider entry");
            let fixture_dir = entry.path().join("tests").join("fixtures");
            if !fixture_dir.is_dir() {
                return None;
            }

            let has_chunk_fixture = std::fs::read_dir(&fixture_dir)
                .expect("fixture directory")
                .filter_map(Result::ok)
                .any(|fixture| {
                    fixture.file_type().map(|ty| ty.is_file()).unwrap_or(false)
                        && fixture
                            .file_name()
                            .to_string_lossy()
                            .ends_with(".chunks.txt")
                });

            has_chunk_fixture.then(|| entry.file_name().to_string_lossy().into_owned())
        })
        .collect::<Vec<_>>();
    owners.sort();
    owners
}

#[test]
fn replay_support_wraps_fixture_lines_as_sse_events_and_appends_done() {
    let fixture_name = "openai-web-search-tool.1";
    let lines = non_empty_fixture_lines(fixture_name);
    let chunks = bench_support::stream_fixture_chunks(fixture_name);

    assert_eq!(
        chunks.len(),
        lines.len() + 1,
        "expected one SSE chunk per non-empty line plus [DONE]",
    );

    for (index, (chunk, line)) in chunks.iter().zip(lines.iter()).enumerate() {
        let expected = format!("data: {line}\n\n");
        assert_eq!(
            chunk.as_ref(),
            expected.as_bytes(),
            "fixture replay should preserve line {index} as a single SSE data frame",
        );
    }

    assert_eq!(
        chunks.last().expect("done sentinel").as_ref(),
        b"data: [DONE]\n\n",
        "fixture replay must end with the done sentinel frame",
    );
}

#[test]
fn cargo_manifest_pins_the_current_offline_bench_inventory() {
    let manifest = include_str!("../Cargo.toml");

    assert_eq!(
        manifest.matches("[[bench]]").count(),
        3,
        "BH0 locks the current three-binary benchmark scaffold before refactors",
    );

    for bench_name in ["openai_responses", "streaming_pipeline", "core_hot_paths"] {
        assert!(
            manifest.contains(&format!("name = \"{bench_name}\"")),
            "expected Cargo.toml to keep the current benchmark target {bench_name}",
        );
    }
}

#[test]
fn checked_in_stream_fixture_inventory_is_currently_openai_only() {
    assert_eq!(
        providers_with_checked_in_chunk_fixtures(),
        vec!["openai".to_string()],
        "BH0 should make the current single-provider fixture gap explicit before expanding coverage",
    );
}

#[test]
fn benchmarking_docs_call_out_the_current_scope_and_ci_policy() {
    let docs = include_str!("../docs/benchmarking.md");

    for expected in [
        "Real captured benchmark fixtures currently exist only for the OpenAI Responses path",
        "No Anthropic, Google, Google Vertex, Amazon Bedrock, Azure, Gateway, or OpenAI-compatible benchmark fixtures are checked in yet.",
        "CI only proves that the current offline scaffold compiles with `cargo bench --workspace --no-run`",
    ] {
        assert!(
            docs.contains(expected),
            "expected docs/benchmarking.md to contain: {expected}",
        );
    }
}
