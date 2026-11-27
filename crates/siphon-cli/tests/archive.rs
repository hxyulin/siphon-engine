use std::{
    fs,
    io::Write as _,
    process::{Command, Output, Stdio},
};

use serde_json::{Value, json};

fn siphon() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_siphon"));
    command.arg("archive");
    command
}

fn pipe(operation: &str, input: &[u8]) -> Output {
    let mut child = siphon()
        .arg(operation)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start siphon archive");
    child
        .stdin
        .take()
        .expect("child stdin is piped")
        .write_all(input)
        .expect("write archive input");
    child.wait_with_output().expect("wait for siphon archive")
}

fn success(output: Output) -> Vec<u8> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn jsonl(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes)
        .expect("output is UTF-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line is JSON"))
        .collect()
}

#[test]
fn round_trips_branches_compaction_and_retries() -> anyhow::Result<()> {
    let user = json!({"role": "user", "content": "Run the checks"});
    let tool = json!({"type": "function_call_output", "call_id": "call_1", "output": "ok\n✓"});
    let reply = json!({"role": "assistant", "content": ["All checks passed"]});
    let rows = [
        json!({"id":"first","model":"inkling","request":{"instructions":"Be concise",
            "input":[user, tool],"tools":[{"type":"function","name":"check"}],"stream":true},
            "response":{"output":[reply],"usage":{"output_tokens":5}},"status":200,"cached":0}),
        json!({"id":"other","request":{"messages":[user, reply]},"response":{"content":[tool]},
            "prompt":20,"cached":12,"extra":{"id":0,"header":{},"turn":null}}),
        json!({"id":"branch","request":{"input":[user, tool, reply, {"role":"user","content":"Why?"}]},
            "response":{"choices":[]},"status":499,"client":"disconnected","cached":0}),
        json!({"id":"compact","request":{"input":"Checks previously passed.","tools": []},
            "output":null,"status":503}),
        json!({"id":"retry","request":{"input":"Checks previously passed.","tools": []},
            "output":[reply],"status":200}),
    ];
    let mut input = Vec::new();
    for row in &rows {
        writeln!(input, "{row}")?;
    }
    let archive = success(pipe("pack", &input));
    let packed = zstd::decode_all(&archive[..]).expect("archive is zstd");
    let records = jsonl(&packed);
    assert_eq!(records.first(), Some(&json!({"header":{"version":1}})));
    assert_eq!(records.last(), Some(&json!({"end":{}})));
    for item in [&user, &tool, &reply] {
        assert_eq!(
            records
                .iter()
                .filter(|record| record.get("item") == Some(item))
                .count(),
            1,
            "item {item} was not stored exactly once"
        );
    }
    assert!(records.iter().any(|row| row["turn"]["prompt"] == 20));
    assert_eq!(rows.as_slice(), jsonl(&success(pipe("expand", &archive))));
    Ok(())
}

#[test]
fn preserves_large_numbers_and_accepts_concatenated_archives() {
    let first = br#"{"request":{"messages":[{"n":9007199254740993.125}]},"extra":1e400}"#;
    let second = br#"{"input":[{"n":9007199254740993.125}],"output":[]}"#;
    let mut archive = success(pipe("pack", first));
    archive.extend(success(pipe("pack", b"\n\r\n")));
    archive.extend(success(pipe("pack", second)));
    let restored = success(pipe("expand", &archive));
    assert_eq!(
        std::str::from_utf8(&restored).unwrap(),
        "{\"extra\":1e400,\"request\":{\"messages\":[{\"n\":9007199254740993.125}]}}\n\
         {\"input\":[{\"n\":9007199254740993.125}],\"output\":[]}\n"
    );
}

#[test]
fn rejects_invalid_archives() {
    for (input, message) in [
        (
            "{\"header\":{\"version\":2}}\n{\"end\":{}}\n",
            "unsupported archive version 2",
        ),
        (
            "{\"header\":{\"version\":1}}\n{\"turn\":{\"input\":[0]}}\n{\"end\":{}}\n",
            "unknown item ID 0",
        ),
        ("{\"header\":{\"version\":1}}\n", "missing archive end"),
        (
            "{\"header\":{\"version\":1}}\n{\"end\":{}}\n{\"item\":0}\n",
            "data after archive end",
        ),
    ] {
        let archive = zstd::encode_all(input.as_bytes(), 0).expect("compress test archive");
        let output = pipe("expand", &archive);
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(message), "{error}");
    }
    let mut archive = success(pipe("pack", b"{\"content\":\"text\"}"));
    archive.pop();
    assert!(!pipe("expand", &archive).status.success());
}

#[test]
fn publishes_named_outputs_only_on_success_and_never_replaces_inputs() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("turns.jsonl");
    let archive = dir.path().join("turns.jsonl.zst");
    fs::write(&input, b"{}\n[]\n")?;
    let pack = |output: &std::path::Path| {
        siphon()
            .arg("pack")
            .arg(&input)
            .arg("-o")
            .arg(output)
            .output()
    };
    let failed = pack(&archive)?;
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("JSONL line 2"));
    assert!(!archive.exists());

    fs::write(&input, b"{\"messages\":[\"hello\"]}\n")?;
    assert!(!pack(&input)?.status.success());
    success(pack(&archive)?);
    let restored = dir.path().join("restored.jsonl");
    success(
        siphon()
            .arg("expand")
            .arg(&archive)
            .arg("-o")
            .arg(&restored)
            .output()?,
    );
    assert_eq!(fs::read(&input)?, fs::read(&restored)?);

    let mut truncated = fs::read(&archive)?;
    truncated.pop();
    fs::write(&archive, truncated)?;
    let failed_path = dir.path().join("incomplete.jsonl");
    assert!(
        !siphon()
            .arg("expand")
            .arg(&archive)
            .arg("-o")
            .arg(&failed_path)
            .output()?
            .status
            .success()
    );
    assert!(!failed_path.exists());
    Ok(())
}
