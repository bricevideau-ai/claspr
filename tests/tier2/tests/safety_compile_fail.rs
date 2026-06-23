//! Compile-fail surface for claspr's unified `DeviceOp` API, driven by
//! [`ui_test`].
//!
//! After the Tier-1/Tier-2 reunification there is one `DeviceOp` trait and
//! the old closure layer is gone — but a couple of type-system safety
//! invariants must survive the fold. Each fixture in
//! `tests/tier2/compile_fail/` deliberately violates one and is expected to
//! fail to compile; the captured stderr is diffed against the golden
//! `.stderr` files committed alongside.
//!
//! Coverage:
//!
//! - `fill_on_frozen` — `DeviceSlice::fill` (now a `DeviceOp`-returning verb)
//!   requires `M: Fillable`; `Frozen` isn't `Fillable`, so the fill must be
//!   rejected. Restatement of the deleted `buffer_ops_fill_on_frozen`.
//! - `arc_to_writable_arg` — `Arc<DeviceSlice<T, M>>` impls only
//!   `KernelSliceReadArg`, so a writable kernel slot must reject it.
//!   Restatement of the deleted `arc_to_writable_arg`.
//! - `buffer_ops_write_on_host_read_only` — `DeviceSlice::write` requires
//!   `M: HostWritable`; `HostReadOnly` isn't, so the write must be rejected.
//! - `bundle_aliased_owned_writes` — two `bundle!` arms can't both move-write
//!   the same owned buffer (the second arm is a use-after-move).
//! - `sequential_use_after_move` — an `.and_then` follow-up can't reach a
//!   moved outer `buf` instead of the upstream handle it is handed.
//! - `and_then_host_escapes_buffer` — the mapped view can't escape the
//!   `and_then_host` closure (HRTB `for<'a> FnOnce(View<'a>) -> Result<()>`).
//! - `host_view_outlives_release` — use-after-move on
//!   `DeviceSliceHostView::release_to_device(self)`.
//!
//! A second `compile_pass` config (`Mode::Pass`) runs `compile_pass/sanity.rs`:
//! a known-good unified-API `DeviceOp` chain that MUST compile. It is the
//! harness-integrity guard — if the `--extern`/`-L` wiring were misconfigured,
//! every compile-fail fixture would fail for the wrong reason and the suite
//! would pass spuriously; the pass fixture catches that.
//!
//! ## Why ui_test instead of trybuild
//!
//! Same rationale as `tests/tier1/tests/image_compile_fail.rs`: trybuild
//! forks `cargo check` per fixture (rebuilding the whole rust-gpu chain —
//! minutes per invocation) and has a known bulk-mode false-success bug. We
//! use ui_test's direct-rustc mode: the parent `cargo test` has already built
//! `claspr` + `claspr-test-kernels`; we discover those rlibs in `deps/` and
//! pass them as `--extern` to rustc. Per-fixture compile drops to
//! milliseconds.
//!
//! ## Running and re-blessing
//!
//! ```text
//! cargo test -p claspr-tier2-tests --test safety_compile_fail
//! cargo test -p claspr-tier2-tests --test safety_compile_fail -- --bless
//! ```
//!
//! No OpenCL device needed — these are pure compile-time checks.

use std::path::{Path, PathBuf};

use ui_test::Config;
use ui_test::color_eyre::Result;
use ui_test::color_eyre::eyre::eyre;
use ui_test::run_tests_generic;
use ui_test::spanned::Spanned;
use ui_test::status_emitter::StatusEmitter;

fn main() -> Result<()> {
    let mut args = ui_test::Args::test()?;
    args.bless |= std::env::var_os("RUSTC_BLESS").is_some_and(|v| v != "0");

    let externs = ExternRlibs::discover(&["claspr", "claspr_test_kernels"])?;

    let fail_config = make_config("compile_fail", Mode::Fail, &args, &externs);
    let pass_config = make_config("compile_pass", Mode::Pass, &args, &externs);

    run_tests_generic(
        vec![fail_config, pass_config],
        ui_test::default_file_filter,
        |_, _| {},
        Box::<dyn StatusEmitter>::from(args.format),
    )
}

#[derive(Copy, Clone)]
enum Mode {
    Fail,
    Pass,
}

fn make_config(path: &str, mode: Mode, args: &ui_test::Args, externs: &ExternRlibs) -> Config {
    let mut config = Config::rustc(path);
    config.with_args(args);

    // Pass `--extern <name>=<rlib>` per crate the fixtures `use`, plus
    // `-L dependency=<deps dir>` so rustc can resolve the transitive deps
    // (opencl3, etc.) those rlibs reference. If the deps dir is under a
    // target-triple subdir, the parent build used `--target` explicitly — we
    // must too, so rustc's lookup matches the triple stamped into the rlibs'
    // metadata. We also add `target/release/deps` as a secondary search path
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

    config.bless_command =
        Some("cargo test -p claspr-tier2-tests --test safety_compile_fail -- --bless".into());

    config
}

struct ExternRlibs {
    /// `(crate_name, path_to_rlib)` for each requested crate.
    externs: Vec<(String, PathBuf)>,
    /// `target/<triple>/release/deps/` or `target/release/deps/` — passed as
    /// `-L dependency=…` so rustc resolves transitive deps the rlibs
    /// reference.
    deps_dir: PathBuf,
    /// `Some(triple)` when the parent build used `--target X`. We must pass
    /// the same `--target` so rustc's lookup matches the triple stamped into
    /// the rlibs' metadata.
    target_triple: Option<String>,
    /// Secondary search path for host artifacts (proc-macros, build script
    /// outputs) — `target/release/deps/` — present only when `target_triple`
    /// is set.
    host_deps_dir: Option<PathBuf>,
}

impl ExternRlibs {
    /// Locate the `.rlib` for each requested crate alongside this test binary
    /// in the parent cargo invocation's `deps/` dir.
    ///
    /// Pulling from `current_exe().parent()` guarantees we read rlibs from the
    /// same cargo invocation that built and ran us (matching profile,
    /// `--target`, and feature set).
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
    // metadata hash differs across builds. We want the freshest one — mtime
    // ordering matches what cargo just wrote for this run.
    let prefix = format!("lib{crate_name}-");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
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
            best = Some((mtime, path));
        }
    }
    best.map(|(_, p)| p).ok_or_else(|| {
        eyre!(
            "no `{prefix}*.rlib` in {deps_dir:?} — was `{crate_name}` built by the parent \
             `cargo test` before this test ran?"
        )
    })
}
