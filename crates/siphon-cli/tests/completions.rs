use std::process::Command;

#[test]
fn generates_completions_for_the_full_command_tree() {
    let output = Command::new(env!("CARGO_BIN_EXE_siphon"))
        .args(["completions", "bash"])
        .output()
        .expect("run siphon completions bash");
    assert!(
        output.status.success(),
        "generating completions failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let script = String::from_utf8(output.stdout).expect("Bash completions are UTF-8");
    for flag in ["--all", "--interactive", "--streams"] {
        assert!(script.contains(flag), "completion script is missing {flag}");
    }
}
