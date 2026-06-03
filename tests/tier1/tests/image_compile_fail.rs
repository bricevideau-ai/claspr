//! Compile-fail surface for claspr's image trait dispatch, driven
//! by [`ui_test`].
//!
//! Each fixture in `tests/tier1/compile_fail/image/` deliberately
//! passes the wrong host-side `Image2D<A, F>` (or related image
//! type) to a kernel and is expected to fail with a trait-bound
//! error. The kernel side lives in `claspr-test-image-kernels`;
//! this file just invokes rustc per fixture and diffs the captured
//! stderr against the golden `.stderr` files committed alongside.
//!
//! Coverage:
//!
//! - `family_mismatch_*` — kernel says `type=u32` (`Uint` family);
//!   host passes a `Float`/`Sint` format. Should fail at the
//!   `KernelImage2DWriteArg<Uint>` bound.
//! - `access_*` — kernel says `&Image` (read-only access qualifier);
//!   host passes a `WriteOnly` image (which only impls the write
//!   trait variant). Or vice-versa.
//! - `dim_mismatch_*` — host's `Image<N>D<…>` doesn't match the
//!   dim the kernel declared. Wrong trait family entirely.
//! - `view_*` — `Image1DBufferView` access/lifetime checks.
//!
//! ## Why ui_test instead of trybuild
//!
//! trybuild forks `cargo check` per test invocation. In our setup
//! that subprocess rebuilt the entire rust-gpu compiler chain
//! (spirv-tools, spirv-builder, claspr-test-image-kernels' build
//! script which compiles 8 SPIR-V kernels) — a 5-minute floor per
//! test invocation — and trybuild ≥1.0.50 has a known intermittent
//! false-success bug in its bulk `--keep-going` mode (upstream
//! issues #299, #286, #242) that can mark every fixture as
//! passing when it shouldn't.
//!
//! We use ui_test's direct-rustc mode (no DependencyBuilder, no
//! shim crate). The parent `cargo test --release` has already built
//! `claspr` and `claspr-test-image-kernels` as part of the workspace
//! compile; we discover those rlibs in
//! `target/$TARGET/release/deps/` and pass them as `--extern` to
//! rustc. Per-fixture compile time drops from minutes to
//! milliseconds, and the bulk-mode bug is gone because there is no
//! bulk mode.
//!
//! ## Running and re-blessing
//!
//! ```text
//! cargo test -p claspr-tier1-tests --test image_compile_fail
//! cargo test -p claspr-tier1-tests --test image_compile_fail -- --bless
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

    let externs = ExternRlibs::discover(&["claspr", "claspr_test_image_kernels"])?;

    let fail_config = make_config("compile_fail/image", Mode::Fail, &args, &externs);
    let pass_config = make_config("compile_pass/image", Mode::Pass, &args, &externs);

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

    // Pass `--extern <name>=<rlib>` per crate the fixtures `use`,
    // plus `-L dependency=<deps dir>` so rustc can resolve the
    // transitive deps (opencl3, etc.) that those rlibs reference.
    // If the deps dir is under a target-triple subdir
    // (`target/<triple>/release/deps`), the parent build used
    // `--target` explicitly — we must too, so rustc's lookup
    // matches the triple stamped into the rlibs' metadata. We also
    // add `target/release/deps` as a secondary search path the way
    // cargo does, so host artifacts (proc-macros, build script
    // outputs) resolve.
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
        Some("cargo test -p claspr-tier1-tests --test image_compile_fail -- --bless".into());

    config
}

struct ExternRlibs {
    /// `(crate_name, path_to_rlib)` for each requested crate.
    externs: Vec<(String, PathBuf)>,
    /// `target/<triple>/release/deps/` or `target/release/deps/` —
    /// passed as `-L dependency=…` so rustc resolves transitive
    /// deps (opencl3, etc.) the rlibs reference.
    deps_dir: PathBuf,
    /// `Some(triple)` when the parent build used `--target X`.
    /// We must pass the same `--target` to ui_test's rustc so its
    /// lookup matches the triple stamped into the rlibs' metadata.
    /// `None` when the parent built without `--target` (everything
    /// is host-mode and rustc defaults work).
    target_triple: Option<String>,
    /// Secondary search path for host artifacts (proc-macros,
    /// build script outputs) — `target/release/deps/` — present
    /// only when `target_triple` is set. Mirrors how cargo adds
    /// both paths when `--target` matches the host.
    host_deps_dir: Option<PathBuf>,
}

impl ExternRlibs {
    /// Locate the `.rlib` for each requested crate alongside this
    /// test binary in the parent cargo invocation's `deps/` dir.
    ///
    /// Pulling from `current_exe().parent()` guarantees we read
    /// rlibs from the same cargo invocation that built and ran us
    /// (matching profile, --target, and feature set), regardless
    /// of which sibling target subdir may have stale leftovers
    /// from a different earlier `cargo test` run.
    fn discover(crates: &[&str]) -> Result<Self> {
        let test_exe =
            std::env::current_exe().map_err(|e| eyre!("could not read current_exe(): {e}"))?;
        let deps_dir = test_exe
            .parent()
            .ok_or_else(|| eyre!("test exe {test_exe:?} has no parent"))?
            .to_path_buf();

        // Detect target-triple mode: deps_dir under cargo's `--target`
        // takes the shape `<target_root>/<triple>/release/deps/`,
        // whereas no-target builds use `<target_root>/release/deps/`.
        // Walking up from deps_dir: deps → release → <maybe triple>
        // → <target_root>. If the third ancestor's name parses as
        // a triple (i.e. it's not "target"), we're in --target mode.
        let mut target_triple = None;
        let mut host_deps_dir = None;
        if let Some(parent3) = deps_dir.ancestors().nth(2)
            && let Some(parent_name) = parent3.file_name().and_then(|n| n.to_str())
            && parent_name != "target"
            && let Some(workspace_target) = parent3.parent()
        {
            target_triple = Some(parent_name.to_string());
            host_deps_dir = Some(workspace_target.join("release").join("deps"));
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
    // Same dir can hold multiple rlibs for one crate when cargo's
    // dep-graph metadata hash differs across builds (test deps vs
    // bin deps, feature flips, etc). We want the freshest one —
    // mtime ordering matches what cargo just wrote for this run.
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
             `cargo test --release` before this test ran?"
        )
    })
}
