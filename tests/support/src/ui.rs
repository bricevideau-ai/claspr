//! Shared [`ui_test`] harness plumbing for the compile-fail/-pass
//! suites (`tests/tier1/tests/image_compile_fail.rs`,
//! `tests/tier2/tests/safety_compile_fail.rs`).
//!
//! Each harness is a `harness = false` test binary whose fixtures
//! deliberately (fail configs) or deliberately-not (pass configs)
//! violate a compile-time invariant; captured stderr is diffed
//! against golden `.stderr` files committed alongside. What differs
//! per harness — the fixture dirs, the `--extern` crate list, the
//! re-bless command — comes in through [`run_compile_tests`]; the
//! rlib discovery and rustc wiring below is the shared part.
//!
//! ## Why ui_test instead of trybuild
//!
//! trybuild forks `cargo check` per fixture. In our setup that
//! subprocess rebuilt the entire rust-gpu compiler chain
//! (spirv-tools, spirv-builder, the kernel crates' build scripts) —
//! a 5-minute floor per invocation — and trybuild ≥1.0.50 has a
//! known intermittent false-success bug in its bulk `--keep-going`
//! mode (upstream issues #299, #286, #242) that can mark every
//! fixture as passing when it shouldn't.
//!
//! We use ui_test's direct-rustc mode (no DependencyBuilder, no shim
//! crate). The parent `cargo test` has already built the requested
//! crates as part of the workspace compile; we discover those rlibs
//! in the same `deps/` dir the harness binary was built into and
//! pass them as `--extern` to rustc. Per-fixture compile time drops
//! from minutes to milliseconds, and the bulk-mode bug is gone
//! because there is no bulk mode.

use std::path::{Path, PathBuf};

use ui_test::Config;
use ui_test::color_eyre::eyre::eyre;
use ui_test::run_tests_generic;
use ui_test::spanned::Spanned;
use ui_test::status_emitter::StatusEmitter;

pub use ui_test::color_eyre::Result;

/// Whether a fixture dir's files must fail (exit status 1) or pass
/// (exit status 0) compilation.
#[derive(Copy, Clone)]
pub enum Mode {
    Fail,
    Pass,
}

/// Run every fixture dir in `fixtures` (paths relative to the
/// harness crate's root) against rustc, with `--extern`s for each
/// crate name in `externs` resolved from the parent cargo
/// invocation's `deps/` dir. `bless_command` is echoed in failure
/// output so the reader knows how to re-bless the goldens.
pub fn run_compile_tests(
    externs: &[&str],
    bless_command: &str,
    fixtures: &[(&str, Mode)],
) -> Result<()> {
    let mut args = ui_test::Args::test()?;
    args.bless |= std::env::var_os("RUSTC_BLESS").is_some_and(|v| v != "0");

    let externs = ExternRlibs::discover(externs)?;

    let configs = fixtures
        .iter()
        .map(|&(path, mode)| make_config(path, mode, bless_command, &args, &externs))
        .collect();

    run_tests_generic(
        configs,
        ui_test::default_file_filter,
        |_, _| {},
        Box::<dyn StatusEmitter>::from(args.format),
    )
}

fn make_config(
    path: &str,
    mode: Mode,
    bless_command: &str,
    args: &ui_test::Args,
    externs: &ExternRlibs,
) -> Config {
    let mut config = Config::rustc(path);
    config.with_args(args);

    // Pass `--extern <name>=<rlib>` per crate the fixtures `use`, plus
    // `-L dependency=<deps dir>` so rustc can resolve the transitive deps
    // (opencl3, etc.) those rlibs reference. If the deps dir is under a
    // target-triple subdir, the parent build used `--target` explicitly — we
    // must too, so rustc's lookup matches the triple stamped into the rlibs'
    // metadata. We also add the host `deps/` as a secondary search path
    // so host artifacts (proc-macros, build script outputs) resolve.
    let program = &mut config.program;
    for (name, rlib) in &externs.externs {
        program.args.push("--extern".into());
        let mut spec = std::ffi::OsString::from(name);
        spec.push("=");
        spec.push(rlib);
        program.args.push(spec);
    }
    program.args.push("-L".into());
    let mut dep_search = std::ffi::OsString::from("dependency=");
    dep_search.push(&externs.deps_dir);
    program.args.push(dep_search);
    if let Some(triple) = externs.target_triple.as_deref() {
        program.args.push(format!("--target={triple}").into());
        if let Some(host_deps) = &externs.host_deps_dir {
            program.args.push("-L".into());
            let mut host_search = std::ffi::OsString::from("dependency=");
            host_search.push(host_deps);
            program.args.push(host_search);
        }
    }

    // We use `.stderr` golden files, not inline `//~ ERROR` markers.
    config.comment_defaults.base().require_annotations = Spanned::dummy(false).into();
    let expected_status = match mode {
        Mode::Fail => 1,
        Mode::Pass => 0,
    };
    config.comment_defaults.base().exit_status = Spanned::dummy(expected_status).into();

    config.bless_command = Some(bless_command.into());

    config
}

struct ExternRlibs {
    /// `(crate_name, path_to_rlib)` for each requested crate.
    externs: Vec<(String, PathBuf)>,
    /// `target/<triple>/<profile>/deps/` or `target/<profile>/deps/` —
    /// passed as `-L dependency=…` so rustc resolves transitive deps
    /// (opencl3, etc.) the rlibs reference.
    deps_dir: PathBuf,
    /// `Some(triple)` when the parent build used `--target X`. We must pass
    /// the same `--target` so rustc's lookup matches the triple stamped into
    /// the rlibs' metadata. `None` when the parent built without `--target`
    /// (everything is host-mode and rustc defaults work).
    target_triple: Option<String>,
    /// Secondary search path for host artifacts (proc-macros, build script
    /// outputs) — `target/<profile>/deps/` — present only when
    /// `target_triple` is set. Mirrors how cargo adds both paths when
    /// `--target` matches the host.
    host_deps_dir: Option<PathBuf>,
}

impl ExternRlibs {
    /// Locate the `.rlib` for each requested crate alongside this test binary
    /// in the parent cargo invocation's `deps/` dir.
    ///
    /// Pulling from `current_exe().parent()` guarantees we read rlibs from
    /// the same cargo invocation that built and ran us (matching profile,
    /// `--target`, and feature set), regardless of which sibling target
    /// subdir may have stale leftovers from a different earlier `cargo test`
    /// run.
    fn discover(crates: &[&str]) -> Result<Self> {
        let test_exe =
            std::env::current_exe().map_err(|e| eyre!("could not read current_exe(): {e}"))?;
        let deps_dir = test_exe
            .parent()
            .ok_or_else(|| eyre!("test exe {test_exe:?} has no parent"))?
            .to_path_buf();

        // Detect target-triple mode: `<root>/<triple>/<profile>/deps/` vs
        // `<root>/<profile>/deps/`. Walking up: deps → profile → <maybe
        // triple> → <root>. If the third ancestor isn't "target", we're in
        // --target mode.
        let mut target_triple = None;
        let mut host_deps_dir = None;
        if let Some(parent3) = deps_dir.ancestors().nth(2)
            && let Some(parent_name) = parent3.file_name().and_then(|n| n.to_str())
            && parent_name != "target"
            && let Some(workspace_target) = parent3.parent()
        {
            // Mirror the profile dir (debug/release) the parent used.
            let profile = deps_dir
                .ancestors()
                .nth(1)
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("debug")
                .to_string();
            target_triple = Some(parent_name.to_string());
            host_deps_dir = Some(workspace_target.join(profile).join("deps"));
        }

        let externs = crates
            .iter()
            .map(|&name| {
                let rlib = find_newest_rlib(&deps_dir, name)?;
                Ok::<_, ui_test::color_eyre::Report>((name.to_string(), rlib))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            externs,
            deps_dir,
            target_triple,
            host_deps_dir,
        })
    }
}

fn find_newest_rlib(deps_dir: &Path, crate_name: &str) -> Result<PathBuf> {
    // Same dir can hold multiple rlibs for one crate when cargo's dep-graph
    // metadata hash differs across builds (test deps vs bin deps, feature
    // flips, per-package `cargo test -p …` invocations, etc). Picking the
    // globally newest is WRONG when a later, unrelated invocation rebuilt a
    // different variant of a dep: the fixtures then mix rlibs from two
    // builds and rustc reports "multiple different versions of crate X".
    // The rlibs consistent with each other are the ones from the SAME
    // cargo invocation that built THIS harness binary — so prefer the
    // newest rlib not newer than the harness binary itself, and only fall
    // back to the globally newest when everything postdates it (e.g. a
    // `cargo build` refreshed rlibs without relinking the test).
    let exe_mtime = std::env::current_exe()
        .ok()
        .and_then(|p| p.metadata().ok())
        .and_then(|m| m.modified().ok());
    let prefix = format!("lib{crate_name}-");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let mut best_coeval: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(deps_dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if !name.starts_with(&prefix) || !name.ends_with(".rlib") {
            continue;
        }
        let mtime = entry.metadata()?.modified()?;
        if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
            best = Some((mtime, path.clone()));
        }
        // "Coeval": produced no later than the harness binary (plus a
        // small slack for filesystem timestamp granularity).
        //
        // KNOWN LIMIT: mtime heuristics cannot fully identify the set
        // one cargo invocation built — mixed `cargo test -p …` runs
        // leave differently-feature-unified variants side by side, and
        // an unchanged unit keeps its old mtime even in the current
        // run. If fixtures fail with rustc's "multiple different
        // versions of crate `claspr`", that's this: run
        // `cargo clean -p claspr` and re-run the suite through one
        // workspace-level invocation.
        if exe_mtime.is_some_and(|e| mtime <= e + std::time::Duration::from_secs(2))
            && best_coeval.as_ref().is_none_or(|(t, _)| mtime > *t)
        {
            best_coeval = Some((mtime, path));
        }
    }
    best_coeval.or(best).map(|(_, p)| p).ok_or_else(|| {
        eyre!(
            "no `{prefix}*.rlib` in {deps_dir:?} — was `{crate_name}` built by the parent \
             `cargo test` invocation before this test ran?"
        )
    })
}
