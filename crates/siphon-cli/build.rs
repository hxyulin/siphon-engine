//! Records the commit the server is built from, for its startup log line and `/version`:
//! `SIPHON_COMMIT` when set (`scripts/remote.sh` and the Docker build set it, as their
//! copies hold no `.git`), else `git rev-parse` of the checkout, else `unknown`.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn main() {
    println!("cargo:rerun-if-env-changed=SIPHON_COMMIT");
    let commit = std::env::var("SIPHON_COMMIT")
        .ok()
        .filter(|c| !c.is_empty())
        .or_else(|| {
            // Rebuild when HEAD moves: the file HEAD names a branch whose ref changes.
            for path in ["HEAD", "packed-refs"] {
                if let Some(p) = git(&["rev-parse", "--git-path", path]) {
                    println!("cargo:rerun-if-changed={p}");
                }
            }
            if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"])
                && let Some(p) = git(&["rev-parse", "--git-path", &branch])
            {
                println!("cargo:rerun-if-changed={p}");
            }
            git(&["rev-parse", "--short=12", "HEAD"])
        })
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=SIPHON_COMMIT={commit}");
}
