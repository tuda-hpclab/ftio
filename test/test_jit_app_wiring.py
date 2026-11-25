"""Tests for the JIT app wiring rules that silently break a run when violated.

These are pure-logic guards over jitsettings' helpers and regexes. They must stay
portable: no JitSettings() (its branches key off the hostname and hardcode machine
paths), no benchmarks, no GekkoFS.

Three rules are pinned here, each of which cost a debugging session:

1. A checkpoint regex that matches nothing means `get_items_to_submit` selects
   nothing and the async flush silently stages zero files.
2. WRF has no CLI: the restart path only moves via the namelist's rst_outname.
   QMCPACK likewise takes its output prefix from <project id>.
3. Neither may be wired with a `cdf`/`cpf` pre_app_call: those are cluster shell
   aliases and do not exist elsewhere, so the pre-call dies before the app runs.
   (A pre_app_call as such is fine -- DLIO uses one.)

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import inspect
import re
from types import SimpleNamespace

from ftio.api.gekkoFs.jit.jitsettings import JitSettings

MNT = "/mnt/gkfs"

# The patterns jitsettings assigns per app, and a path each one must select.
APP_PATTERNS = {
    "lammps": (r".*/ckpt\.restart\.\d+(\.(\d+|base))?$", f"{MNT}/ckpt.restart.80"),
    "castro": (r".*/sedov_3d_sph_chk\d+(?=/)", f"{MNT}/sedov_3d_sph_chk00010/Header"),
    "warpx": (r".*/chk\d+(?=/)", f"{MNT}/chk000010/WarpXHeader"),
    "qmcpack": (r".*/glass_heg\.s\d+\.config\.h5$", f"{MNT}/glass_heg.s000.config.h5"),
    "wrf": (r".*/wrfrst_d\d+_.*$", f"{MNT}/wrfrst_d01_0001-01-01_01:00:00"),
    "btio": (r".*/btio\.simple\.out\.\d+$", f"{MNT}/btio.simple.out.0007"),
}


def test_every_app_regex_matches_its_own_checkpoint():
    for app, (pattern, path) in APP_PATTERNS.items():
        assert re.compile(pattern).match(path), f"{app}: regex misses its checkpoint"


def test_wrf_regex_matches_restarts_not_history_or_logs():
    # The old pattern matched wrfout_* (history) and rsl.* (per-rank logs), so it
    # would have flushed the logs and never a checkpoint. Running WRF next to the
    # mount also puts namelist.input / rsl.* in reach -- none may be staged out.
    wrf = re.compile(APP_PATTERNS["wrf"][0])
    assert wrf.match(f"{MNT}/wrfrst_d01_0001-01-01_02:00:00")
    for never in ("wrfout_d01_0001-01-01_00:00:00", "rsl.error.0000", "namelist.input"):
        assert not wrf.match(f"{MNT}/{never}"), f"{never} must never be staged out"


def test_lammps_regex_matches_both_writer_counts():
    # in.ckpt's -v mp knob switches between single-writer restarts
    # (ckpt.restart.<step>) and multi-file "%" ones (ckpt.restart.<step>.<idx>
    # plus a .base metadata file). Flipping it once without updating this regex
    # meant nothing matched and every FTIO trigger staged 0 items for a whole
    # run (BSC 43752428). One pattern covers both so that cannot recur.
    lammps = re.compile(APP_PATTERNS["lammps"][0])
    assert lammps.match(f"{MNT}/ckpt.restart.80")
    assert lammps.match(f"{MNT}/ckpt.restart.80.base")
    assert lammps.match(f"{MNT}/ckpt.restart.80.5")
    # but it must still not sweep up unrelated files next to the mount
    assert not lammps.match(f"{MNT}/ckpt.restart.80.tmp")
    assert not lammps.match(f"{MNT}/log.lammps")


def test_amrex_regexes_exclude_the_temp_rename_artifacts():
    # AMReX writes <name>.temp then renames. The (?=/) lookahead keeps the
    # .old.<pid> / .temp artifacts out of the flush set.
    castro = re.compile(APP_PATTERNS["castro"][0])
    assert castro.match(f"{MNT}/sedov_3d_sph_chk00010/Header")
    assert not castro.match(f"{MNT}/sedov_3d_sph_chk00010.temp")


# ── the two output-redirect helpers ──────────────────────────────────────────


def _settings(tmp_path) -> SimpleNamespace:
    """A stand-in with just the attributes the helpers touch."""
    return SimpleNamespace(run_dir=str(tmp_path))


def test_point_wrf_restarts_at_rewrites_rst_outname(tmp_path):
    namelist = tmp_path / "namelist.input"
    namelist.write_text(
        " &time_control\n"
        " restart_interval                    = 60,\n"
        " rst_outname                         = 'wrfrst_d<domain>_<date>',\n"
        " /\n"
    )
    JitSettings.point_wrf_restarts_at(_settings(tmp_path), MNT)
    out = namelist.read_text()
    assert f"'{MNT}/wrfrst_d<domain>_<date>'" in out
    # the rest of the namelist survives
    assert "restart_interval                    = 60," in out
    assert out.count("rst_outname") == 1


def test_point_wrf_restarts_at_is_idempotent(tmp_path):
    namelist = tmp_path / "namelist.input"
    namelist.write_text(" rst_outname = 'wrfrst_d<domain>_<date>',\n")
    JitSettings.point_wrf_restarts_at(_settings(tmp_path), MNT)
    first = namelist.read_text()
    JitSettings.point_wrf_restarts_at(_settings(tmp_path), MNT)
    assert namelist.read_text() == first, "a second run must not double-prefix"


def test_point_wrf_restarts_at_survives_a_missing_namelist(tmp_path):
    # On a machine without the deck this must warn, not raise: JitSettings is
    # built during argument parsing.
    JitSettings.point_wrf_restarts_at(_settings(tmp_path / "nope"), MNT)


def test_point_qmcpack_output_at_rewrites_the_project_id(tmp_path):
    xml = tmp_path / "glass.xml"
    xml.write_text('<simulation>\n<project id="glass_heg" series="0">\n</simulation>\n')
    JitSettings.point_qmcpack_output_at(_settings(tmp_path), MNT)
    assert f'<project id="{MNT}/glass_heg"' in xml.read_text()


def test_point_qmcpack_output_at_is_idempotent(tmp_path):
    xml = tmp_path / "glass.xml"
    xml.write_text('<project id="glass_heg" series="0">\n')
    JitSettings.point_qmcpack_output_at(_settings(tmp_path), MNT)
    first = xml.read_text()
    JitSettings.point_qmcpack_output_at(_settings(tmp_path), "/other/mnt")
    # rewriting to a new mount replaces the old absolute path, never nests it
    assert xml.read_text().count("glass_heg") == first.count("glass_heg")
    assert '<project id="/other/mnt/glass_heg"' in xml.read_text()


def test_point_qmcpack_output_at_survives_a_missing_input(tmp_path):
    JitSettings.point_qmcpack_output_at(_settings(tmp_path / "nope"), MNT)


def test_set_qmc_steps_rewrites_every_section_and_leaves_walkers_alone(tmp_path):
    section = (
        '<qmc method="vmc">\n'
        '  <parameter name="walkers"   >  15625  </parameter>\n'
        '  <parameter name="steps"     >    1    </parameter>\n'
        '  <parameter name="subSteps"  >    1    </parameter>\n'
        '  <parameter name="warmupSteps">   3    </parameter>\n'
        "</qmc>\n"
    )
    xml = tmp_path / "glass.xml"
    xml.write_text(section * 3)
    JitSettings.set_qmc_steps(_settings(tmp_path), 40)
    out = xml.read_text()
    assert out.count(">    40    </parameter>") == 3
    # subSteps / warmupSteps / walkers must not be touched
    assert out.count('"subSteps"  >    1    <') == 3
    assert out.count('"warmupSteps">   3    <') == 3
    assert out.count(">  15625  <") == 3


def test_set_qmc_steps_survives_a_missing_input(tmp_path):
    JitSettings.set_qmc_steps(_settings(tmp_path / "nope"), 40)


def _qmc_deck(n: int) -> str:
    first = '   <qmc method="vmc">\n  <parameter name="warmupSteps">3</parameter>\n   </qmc>\n'
    rest = '   <qmc method="vmc">\n  <parameter name="warmupSteps">0</parameter>\n   </qmc>\n'
    return "<simulation>\n<project/>\n\n" + first + rest * (n - 1) + "</simulation>\n"


def test_set_qmc_sections_grows_and_shrinks_the_deck(tmp_path):
    xml = tmp_path / "glass.xml"
    for start, want in ((30, 120), (30, 4)):
        xml.write_text(_qmc_deck(start))
        JitSettings.set_qmc_sections(_settings(tmp_path), want)
        out = xml.read_text()
        assert out.count("<qmc ") == want
        # only the first section keeps the warm-up
        assert out.count(">3</parameter>") == 1
        assert out.startswith("<simulation>\n<project/>\n\n   <qmc")
        assert out.endswith("   </qmc>\n</simulation>\n")


def test_set_qmc_sections_survives_a_missing_input(tmp_path):
    JitSettings.set_qmc_sections(_settings(tmp_path / "nope"), 120)


def test_prepare_nek_input_stages_mesh_and_sets_run_length(tmp_path):
    deck = tmp_path / "deck"
    deck.mkdir()
    (deck / "chan.re2").write_bytes(b"mesh")
    (deck / "chan.ma2").write_bytes(b"map")
    (deck / "chan.par").write_text(
        "[GENERAL]\nstopAt = numSteps\nnumSteps = 20\n\nwriteControl = timeStep\n"
        "writeInterval = 5\nuserParam02 = 20\n"
    )
    run = tmp_path / "run"
    s = _settings(run)
    s.nek_case = "chan"
    JitSettings.prepare_nek_input(s, str(deck), 120, 12)
    assert (run / "input" / "chan.re2").read_bytes() == b"mesh"
    assert (run / "input" / "chan.ma2").read_bytes() == b"map"
    par = (run / "input" / "chan.par").read_text()
    assert "numSteps = 120\n" in par and "writeInterval = 12\n" in par
    assert "userParam02 = 20" in par
    # the deck itself stays untouched
    assert "numSteps = 20" in (deck / "chan.par").read_text()


def test_prepare_nek_input_survives_a_missing_deck(tmp_path):
    s = _settings(tmp_path / "run")
    s.nek_case = "chan"
    JitSettings.prepare_nek_input(s, str(tmp_path / "nope"), 120, 12)


def test_prepare_btio_input_sets_class_grid_and_dump_interval(tmp_path):
    JitSettings.prepare_btio_input(_settings(tmp_path), "D", 0, 10)
    lines = (tmp_path / "inputbt.data").read_text().splitlines()
    assert lines[:4] == ["250", "0.00002d0", "408 408 408", "10 0"]


def test_prepare_btio_input_overrides_the_step_count(tmp_path):
    JitSettings.prepare_btio_input(_settings(tmp_path), "C", 40, 5)
    lines = (tmp_path / "inputbt.data").read_text().splitlines()
    assert lines[0] == "40" and lines[2] == "162 162 162"


def test_btio_regex_skips_the_input_deck():
    btio = re.compile(APP_PATTERNS["btio"][0])
    assert not btio.match(f"{MNT}/inputbt.data")


# ── --app-flags must actually override (it used to be ignored) ───────────────


def _flag_settings(user_flags: str, run_dir: str = "/run") -> SimpleNamespace:
    return SimpleNamespace(app_flags=user_flags, run_dir=run_dir)


DEFAULT = "-v x 142 -v y 142 -v z 142 -v every 10"


def test_no_app_flags_falls_back_to_the_tuned_default():
    s = _flag_settings("")
    assert JitSettings.resolve_app_flags(s, DEFAULT, MNT) == DEFAULT


def test_user_app_flags_win_over_the_default():
    # The driver apps used to overwrite app_flags unconditionally, so --app-flags
    # was silently ignored for exactly the apps whose phases you want to retune.
    s = _flag_settings("-v x 179 -v y 179 -v z 179 -v every 25 -v ckptdir {ckptdir}")
    out = JitSettings.resolve_app_flags(s, DEFAULT, MNT)
    assert "-v x 179" in out
    assert "142" not in out, "the default must not leak through"


def test_ckptdir_placeholder_is_substituted():
    # The mount path is only known at runtime, so the user cannot hardcode it.
    s = _flag_settings("-v ckptdir {ckptdir} -v x 179")
    out = JitSettings.resolve_app_flags(s, DEFAULT, MNT)
    assert out == f"-v ckptdir {MNT} -v x 179"
    assert "{ckptdir}" not in out


def test_run_dir_placeholder_is_substituted():
    s = _flag_settings("-in {run_dir}/in.ckpt -v ckptdir {ckptdir}", run_dir="/deck")
    out = JitSettings.resolve_app_flags(s, DEFAULT, MNT)
    assert out == f"-in /deck/in.ckpt -v ckptdir {MNT}"


# ── the cluster-only shell aliases must not leak into a pre_app_call ─────────


def test_no_app_is_wired_with_the_cdf_cpf_cluster_aliases():
    """`cdf` / `cpf` are cluster shell aliases; they do not exist elsewhere.

    The old WRF branch built its pre_app_call from them (and from $HOME paths),
    so the pre-call died with "cdf: command not found" before the app ever ran.
    It also pointed WRF at the mount as its cwd, which cannot work: WRF reads
    each namelist group with a REWIND and that fails through GekkoFS.

    A pre_app_call is fine in general -- DLIO uses one -- so this guards the
    aliases, not the mechanism.
    """
    src = inspect.getsource(JitSettings.set_variables)
    for app in ('elif "wrf" in self.app:', 'elif "qmc" in self.app:'):
        assert app in src, f"{app} branch vanished from set_variables"
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("self.pre_app_call") and (
            "cdf " in stripped or "cpf " in stripped
        ):
            raise AssertionError(
                "a cdf/cpf pre_app_call is back -- those aliases do not exist "
                "off the cluster: " + stripped
            )


# ── cpus-per-task must not scale with procs_app for auto-threading apps ─────


def test_qmcpack_pins_a_low_fixed_cpus_per_task_regardless_of_procs_app():
    """QMCPACK reads --cpus-per-task and spawns that many OpenMP threads per
    rank on top of its walker-level MPI parallelism. The shared srun launch
    line ties --cpus-per-task to procs_app, so raising procs_app for more MPI
    ranks (e.g. 8 -> 56) also multiplies OMP threads/rank by the same amount:
    56 ranks * 56 threads = 3,136 threads/node, ~28x oversubscribed on a
    112-core node -- confirmed the cause of a 6-8x slowdown vs. the
    procs_app=8 baseline (2026-09-01).

    Pinning cpus_per_task_app to 1 fixes the oversubscription but hangs at
    launch (job 45275298/87/88, 2026-09-01): GekkoFS's client intercept needs
    its own progress thread alongside the app thread, so 1 core/rank starves
    it too. 2 threads/rank (56 ranks * 2 = 112, exactly the node's core count,
    no oversubscription) is the fix.

    set_variables() defaults cpus_per_task_app to procs_app (every other app
    keeps its current launch shape) and must override it inside the qmcpack
    branch specifically, so procs_app only ever controls MPI rank count for
    this app.
    """
    src = inspect.getsource(JitSettings.set_variables)
    lines = src.splitlines()
    default_idx = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == "self.cpus_per_task_app = self.procs_app"
    )
    qmc_idx = next(i for i, line in enumerate(lines) if 'elif "qmc" in self.app:' in line)
    assert default_idx < qmc_idx, "default cpus_per_task_app must precede the qmc branch"

    qmc_block = []
    for line in lines[qmc_idx + 1 :]:
        if line.strip().startswith("elif ") or line.strip().startswith("else"):
            break
        qmc_block.append(line)
    assert any(
        line.strip() == "self.cpus_per_task_app = 2" for line in qmc_block
    ), "qmcpack branch must pin cpus_per_task_app to 2 (1 hangs at launch)"


# ── pre_app_call as a list (dlio: mkdir before the racy parallel mpirun) ─────


def _node_settings(pre_app_call, procs_app=4, app_nodes=64) -> SimpleNamespace:
    return SimpleNamespace(
        pre_app_call=pre_app_call, procs_app=procs_app, app_nodes=app_nodes
    )


def test_update_app_nodes_substitutes_every_call_in_a_list():
    # dlio's pre_app_call is a [mkdir, mpirun] list: pre-creating data/
    # checkpoints/hydra_log serially avoids every one of the $APP_PROCS_X_NODES
    # ranks racing to create the same three dirs under the FUSE mount at once
    # (that race is what hung dlio glass/gekko at 33 and 65 nodes -- confirmed
    # against gekko_fuse.log spinning on root readdir for the full timeout).
    # `"$APP_PROCS_X_NODES" in a_list` checks list membership, not substring
    # containment, so a naive fix would silently skip substitution here.
    s = _node_settings(["mkdir -p /mnt/data", "mpirun -np $APP_PROCS_X_NODES foo"])
    JitSettings.update_app_nodes(s)
    assert s.pre_app_call == ["mkdir -p /mnt/data", "mpirun -np 256 foo"]


def test_set_dir_gekko_relocates_mntdir_inside_a_pre_app_call_list(monkeypatch):
    from ftio.api.gekkoFs.jit.setup_helper import set_dir_gekko

    monkeypatch.setattr("socket.gethostname", lambda: "gs01r1b01")
    s = SimpleNamespace(
        node_local=True,
        cluster=True,
        job_id="123",
        gkfs_rootdir="/dev/shm/tarraf_gkfs_rootdir",
        gkfs_mntdir="/dev/shm/tarraf_gkfs_mountdir",
        run_dir="/run",
        app_flags="",
        pre_app_call=["mkdir -p /dev/shm/tarraf_gkfs_mountdir/data", "mpirun foo"],
        post_app_call="",
        app_call="",
        update_files_with_gkfs_mntdir=[],
    )
    set_dir_gekko(s)
    assert s.pre_app_call[0] == "mkdir -p /scratch/tmp/123/tarraf_gkfs_mountdir/data"
    assert s.pre_app_call[1] == "mpirun foo"


def test_pre_call_srun_wraps_non_mpi_list_items_onto_an_app_node():
    # setup_core.py's pre_app_call list dispatch used to only flag-wrap items
    # containing mpirun/mpiexec; a bare `mkdir` ran unwrapped via
    # execute_block_and_monitor, i.e. as a local subprocess on whatever host
    # is executing the JIT driver process (the ftio_node), not on an app
    # node's FUSE mount -- so dlio's pre-created dirs were invisible to the
    # mpirun step that followed. flaged_call must now be reached for both
    # branches of the list loop, not just the mpirun one.
    src = inspect.getsource(
        __import__("ftio.api.gekkoFs.jit.setup_core", fromlist=["pre_call"])
    )
    list_branch = src[src.index("elif isinstance(settings.pre_app_call, list):") :]
    list_branch = list_branch[: list_branch.index("if returncode == 0:")]
    assert list_branch.count("flaged_call(") >= 2, (
        "non-mpi pre_app_call list items must also go through flaged_call "
        "so they land on an app node instead of running bare"
    )


# ── LAMMPS checkpoint-phase default ────────────────────────────────────────


def test_lammps_defaults_to_60_checkpoint_phases():
    """At 30 phases the weak-scaled LAMMPS run finished in ~30-45 s and glass
    lost pfs by 0.72-0.88x across 8-33+1 (too little run for the fixed ~8 s
    staging overhead to amortise). The full 2/4/8/16/32/64/96+1 resweep at 60
    (jobs 45532561-69, 2026-09-09) closed that: glass wins pfs from 16+1 up and
    beats gekko at every N. The default is what every concurrently-queued job
    reads, so it must be 60, still overridable via LAMMPS_PHASES for probes.
    """
    src = inspect.getsource(JitSettings.set_variables)
    lammps_idx = src.index('elif "lammps" in self.app:')
    lammps_block = src[lammps_idx:]
    lammps_block = lammps_block[: lammps_block.index("#  ├─ DLIO")]
    assert "os.getenv('LAMMPS_PHASES', '60')" in lammps_block, (
        "LAMMPS default checkpoint phases must be 60 (30 was a structural "
        "small-N pfs loss)"
    )


def test_metrics_flush_interval_is_one_setting_defaulting_to_5_s():
    """FTIO sees new data only once per client metrics push, so the interval is
    an opt-in knob (METRICS_FLUSH_INTERVAL) for short app phases like S3D-IO.
    Every launch path must read the setting, and the default must stay 5 s so
    runs that don't set it are unchanged.
    """
    from ftio.api.gekkoFs.jit import jitsettings, setup_core

    assert 'os.getenv("METRICS_FLUSH_INTERVAL", "5")' in inspect.getsource(jitsettings)
    core = inspect.getsource(setup_core)
    assert "LIBGKFS_METRICS_FLUSH_INTERVAL=5" not in core
    assert (
        core.count("LIBGKFS_METRICS_FLUSH_INTERVAL={settings.metrics_flush_interval}")
        == 5
    )
    # an explicit interval also caps FTIO's ZMQ batch at that interval; the
    # default run must not get the flag
    js = inspect.getsource(jitsettings)
    assert 'if os.getenv("METRICS_FLUSH_INTERVAL"):' in js
    assert "--max-batch-window {self.metrics_flush_interval}" in js


def test_gkfs_daemon_args_reach_both_cluster_launch_paths():
    """GKFS_DAEMON_ARGS (e.g. --auto-sm) is appended to the srun and mpirun daemon
    calls; unset, the daemon command is unchanged."""
    from ftio.api.gekkoFs.jit import jitsettings, setup_core

    assert 'os.getenv("GKFS_DAEMON_ARGS", "")' in inspect.getsource(jitsettings)
    core = inspect.getsource(setup_core)
    assert (
        core.count(
            "-l ib0 -P {settings.gkfs_daemon_protocol} {settings.gkfs_daemon_args}"
        )
        == 2
    )


def test_hacc_flush_regex_matches_per_rank_checkpoints_only():
    """HACC-IO writes <mnt>/mpi[.<loop>]-PartNNNNNNNN-of-NNNNNNNN.data per rank;
    the flush regex must take both namings and nothing else in the mount."""
    s = SimpleNamespace(app_call="./HACC_ASYNC_IO")
    JitSettings.select_regexes(s)
    for path in (
        "/dev/shm/mnt/mpi-Part00000000-of-00000002.data",
        "/dev/shm/mnt/mpi.12-Part00004607-of-00004608.data",
    ):
        assert re.match(s.regex_flush_match, path)
        assert re.match(s.regex_stage_out_match, path)
    assert not re.match(s.regex_flush_match, "/dev/shm/mnt/mpi.3-Part1.data.tmp")
    assert not re.match(s.regex_flush_match, "/dev/shm/mnt/checkpoint_all.txt")


def test_nek_flush_regex_skips_the_mpi_io_temp_files():
    """Nek5000 writes <mnt>/chan0.f<N> through MPI-IO, and Open MPI leaves
    per-rank chan0.f<N>.data.<r> / .metadata.<r> temp files next to it while it
    writes; only the dump itself may be flushed (BSC 48871819 queued ~3600)."""
    s = SimpleNamespace(app_call="./nek5000", nek_case="chan")
    JitSettings.select_regexes(s)
    assert re.match(s.regex_flush_match, "/dev/shm/mnt/chan0.f00008")
    assert re.match(s.regex_stage_out_match, "/dev/shm/mnt/chan0.f00008")
    for tmp in (
        "/dev/shm/mnt/chan0.f00008.data.456",
        "/dev/shm/mnt/chan0.f00008.metadata.4",
    ):
        assert not re.match(s.regex_flush_match, tmp)
        assert not re.match(s.regex_stage_out_match, tmp)


def test_e3sm_flush_regex_matches_rotated_history_files_only():
    """E3SM-IO with E3SM_FILES writes one history pair per checkpoint,
    <mnt>/e3sm.<k>_h0.nc and _h1.nc (blob: plus a per-node .<NNNN> subfile
    suffix); the unrotated name must not match."""
    s = SimpleNamespace(app_call="./e3sm_io")
    JitSettings.select_regexes(s)
    for path in (
        "/dev/shm/mnt/e3sm.0_h0.nc",
        "/dev/shm/mnt/e3sm.17_h1.nc",
        "/dev/shm/mnt/e3sm.4_h0.nc.0015",
    ):
        assert re.match(s.regex_flush_match, path)
        assert re.match(s.regex_stage_out_match, path)
    assert not re.match(s.regex_flush_match, "/dev/shm/mnt/e3sm_h0.nc")
    assert not re.match(s.regex_flush_match, "/dev/shm/mnt/e3sm.3_h0.nc.tmp")


def test_milc_input_saves_one_config_per_set_and_regex_takes_them(tmp_path):
    """su3_ora runs one parameter set per checkpoint: the first starts fresh,
    the rest continue, and each set saves <outdir>/cfg.<k>; the flush regex
    takes the configs and their .info files only."""
    s = SimpleNamespace(run_dir=str(tmp_path))
    JitSettings.prepare_milc_input(s, "/dev/shm/mnt", 16, 3, 5)
    text = (tmp_path / "milc.in").read_text()
    assert text.count("save_parallel /dev/shm/mnt/cfg.") == 3
    assert text.count("fresh") == 1 and text.count("continue") == 2
    assert "trajecs 5" in text and "nt 16" in text
    s = SimpleNamespace(app_call="/scratch/run/milc/su3_ora")
    JitSettings.select_regexes(s)
    for path in ("/dev/shm/mnt/cfg.0", "/dev/shm/mnt/cfg.12.info"):
        assert re.match(s.regex_flush_match, path)
        assert re.match(s.regex_stage_out_match, path)
    assert not re.match(s.regex_flush_match, "/dev/shm/mnt/milc.in")


def test_vpic_input_and_regex_take_one_file_per_dump(tmp_path):
    """h5bench_write gets a VPIC-IO config (contiguous 1D, emulated compute);
    with H5BENCH_FILE_PER_TS=1 dump i is <mnt>/vpic.h5.<i>, and only those are
    flushed, not the empty vpic.h5 or the config."""
    s = SimpleNamespace(run_dir=str(tmp_path))
    JitSettings.prepare_vpic_input(s, "2M", 12, 20)
    lines = (tmp_path / "vpic.cfg").read_text().splitlines()
    assert "TIMESTEPS=12" in lines and "DIM_1=2M" in lines
    assert "EMULATED_COMPUTE_TIME_PER_TIMESTEP=20 s" in lines
    assert "DELAYED_CLOSE_TIMESTEPS=0" in lines
    s = SimpleNamespace(
        app_call="env H5BENCH_FILE_PER_TS=1 /scratch/run/vpic/h5bench_write"
    )
    JitSettings.select_regexes(s)
    assert re.match(s.regex_flush_match, "/dev/shm/mnt/vpic.h5.7")
    for path in ("/dev/shm/mnt/vpic.h5", "/scratch/run/vpic/vpic.cfg"):
        assert not re.match(s.regex_flush_match, path)


# ── WRF weak-grid e_sn cap ──────────────────────────────────────────────────


def test_weak_scale_wrf_grid_never_exceeds_the_jet_sounding_table():
    """WRF's stock em_b_wave idealized case hardcodes a 64x80 jet-sounding
    lookup table (module_initialize_ideal.F: nz_jet=64, ny_jet=80) and indexes
    it by the model's raw south-north grid index with no bounds check. Any
    e_sn > 80 reads past the table; e_sn=648 (the 32+1 weak-grid point)
    SIGSEGVs in calc_jet_sounding (root-caused via gdb, job 45731615,
    2026-09-11 -- confirmed unrelated to GekkoFS, ideal.exe never touches the
    mount). e_sn must stay <= 80 at every rank count the sweep uses.
    """
    for ranks in (2, 4, 8, 16, 32, 64, 96, 124):
        _e_we, e_sn, _dx, _dy, _dt, _nproc_x, _nproc_y = JitSettings.weak_scale_wrf_grid(
            ranks
        )
        assert e_sn <= 80, (
            f"e_sn={e_sn} at ranks={ranks} exceeds WRF's 80-column jet-sounding "
            "table and will SIGSEGV ideal.exe"
        )


def test_weak_scale_wrf_grid_decomposition_keeps_patches_above_minimum():
    """Once e_sn is capped at 80 (see above), WRF's own auto-decomposition
    (nproc_x=nproc_y=-1) starts splitting the south-north axis across more
    than 8 processes once total ranks exceeds ~64, giving each rank fewer
    than 10 south-north grid points -- below WRF's minimum MPI-patch size.
    wrf.exe was killed within ~2s of starting at 96+1/124+1 under exactly this
    condition (jobs 45780469/45780470, 2026-09-12). nproc_x/nproc_y must
    always multiply back to the full rank count and keep both axes' per-rank
    patch size at or above the minimum.
    """
    MIN_PATCH = 10
    for ranks in (2, 4, 8, 16, 32, 64, 96, 124):
        e_we, e_sn, _dx, _dy, _dt, nproc_x, nproc_y = JitSettings.weak_scale_wrf_grid(
            ranks
        )
        assert (
            nproc_x * nproc_y == ranks
        ), f"nproc_x={nproc_x} * nproc_y={nproc_y} != ranks={ranks}"
        assert e_sn / nproc_y >= MIN_PATCH, (
            f"ranks={ranks}: south-north patch {e_sn}/{nproc_y}="
            f"{e_sn / nproc_y:.1f} below WRF's minimum {MIN_PATCH}"
        )
        assert e_we / nproc_x >= MIN_PATCH, (
            f"ranks={ranks}: east-west patch {e_we}/{nproc_x}="
            f"{e_we / nproc_x:.1f} below WRF's minimum {MIN_PATCH}"
        )


def test_glass_stage_out_times_from_app_end_and_waits_for_ftio(tmp_path, monkeypatch):
    # The GLASS stage-out must cover FTIO's in-flight copies: from app end until
    # both jit_move and FTIO are done. jit_move must leave the in-flight item alone.
    import threading
    import time

    from ftio.api.gekkoFs import posix_control
    from ftio.api.gekkoFs.jit import setup_core

    settings = JitSettings()
    settings.exclude_all = False
    settings.exclude_cargo = True
    settings.exclude_ftio = False
    settings.flush_log = str(tmp_path / "flush.log")
    settings.app_end_time = time.time() - 5  # app ended 5 s ago
    registry = posix_control.inflight_registry(settings.flush_log)
    posix_control._update_inflight(registry, [f"{MNT}/ckpt.restart.80"], add=True)

    calls, recorded = [], {}
    monkeypatch.setattr(setup_core, "adjust_regex", lambda *a: None)
    monkeypatch.setattr(setup_core, "shut_down", lambda *a: calls.append("shutdown"))
    monkeypatch.setattr(
        setup_core, "jit_move", lambda s, exclude=None: calls.append(("move", exclude))
    )
    monkeypatch.setattr(
        setup_core,
        "elapsed_time",
        lambda s, r, name, elapsed: recorded.setdefault(name, elapsed),
    )
    # FTIO finishes its copy 0.5 s after stage_out starts
    threading.Timer(0.5, posix_control.reset_inflight, args=(registry,)).start()

    setup_core.stage_out(settings, runtime=None)

    assert calls[0] == ("move", [f"{MNT}/ckpt.restart.80"])
    assert calls[1:] == ["shutdown", ("move", None)]
    assert recorded["Stage out"] >= 5.5
    with open(settings.flush_log) as f:
        log = f.read()
    assert "SKIP-INFLIGHT" in log and "FTIO-SHUTDOWN" in log


def test_gekko_stage_out_timing_is_unchanged(tmp_path, monkeypatch):
    import time

    from ftio.api.gekkoFs.jit import setup_core

    settings = JitSettings()
    settings.exclude_all = False
    settings.exclude_cargo = True
    settings.exclude_ftio = True
    settings.app_end_time = time.time() - 100  # must be ignored without FTIO
    recorded = {}
    monkeypatch.setattr(setup_core, "adjust_regex", lambda *a: None)
    monkeypatch.setattr(setup_core, "jit_move", lambda s, exclude=None: None)
    monkeypatch.setattr(
        setup_core,
        "elapsed_time",
        lambda s, r, name, elapsed: recorded.setdefault(name, elapsed),
    )
    setup_core.stage_out(settings, runtime=None)
    assert recorded["Stage out"] < 5


def _hostfile_settings(path, nodes: int) -> SimpleNamespace:
    return SimpleNamespace(
        dry_run=False, cluster=True, app_nodes=nodes, gkfs_hostfile=path
    )


def test_hostfile_check_waits_until_every_daemon_has_registered(tmp_path):
    # Each daemon appends its own line, so the file appears with one line and
    # grows. A client started on the partial file hashes paths onto the wrong
    # daemons (WarpX chk000000 ENOENT, DLIO EBUSY), so the check must wait.
    import threading
    import time

    from ftio.api.gekkoFs.jit import setup_core

    hosts = tmp_path / "gkfs_hosts.txt"
    hosts.write_text("node0 ofi+verbs://a\n")

    def late_daemons():
        for i in (1, 2, 3):
            time.sleep(0.3)
            with open(hosts, "a") as f:
                f.write(f"node{i} ofi+verbs://{i}\n")

    t = threading.Thread(target=late_daemons)
    start = time.time()
    t.start()
    assert setup_core.hostfile_is_sane(_hostfile_settings(str(hosts), 4), timeout=10)
    t.join()
    assert time.time() - start >= 0.8


def test_hostfile_check_fails_when_daemons_never_register(tmp_path):
    from ftio.api.gekkoFs.jit import setup_core

    hosts = tmp_path / "gkfs_hosts.txt"
    hosts.write_text("node0 ofi+verbs://a\nnode1 ofi+verbs://b\n")
    assert not setup_core.hostfile_is_sane(_hostfile_settings(str(hosts), 4), timeout=1)


def test_hostfile_check_still_rejects_a_stale_extra_line(tmp_path):
    from ftio.api.gekkoFs.jit import setup_core

    hosts = tmp_path / "gkfs_hosts.txt"
    hosts.write_text("dead ofi+verbs://x\nnode0 ofi+verbs://a\nnode1 ofi+verbs://b\n")
    assert not setup_core.hostfile_is_sane(_hostfile_settings(str(hosts), 2), timeout=1)


def test_gekko_hostfile_lands_in_the_job_dir_not_home():
    # the daemons create <hostfile>.ready next to it; in $HOME every job killed
    # at its time limit left one behind
    s = SimpleNamespace(
        exclude_daemon=False,
        exclude_proxy=True,
        job_id=42,
        dir="/home/u/jit/42",
        home="/home/u",
        gkfs_hostfile="/home/u/gkfs_hosts.txt",
    )
    JitSettings.update_geko_files(s)
    assert s.gkfs_hostfile == "/home/u/jit/42/gkfs_hosts_42.txt"


def test_skip_last_qmc_checkpoint_only_touches_the_last_section(tmp_path):
    sec = '   <qmc method="vmc" move="pbyp" checkpoint="1">\n   </qmc>\n'
    (tmp_path / "glass.xml").write_text("<simulation>\n" + sec * 3 + "</simulation>\n")
    JitSettings.skip_last_qmc_checkpoint(_settings(tmp_path))
    out = (tmp_path / "glass.xml").read_text()
    assert out.count('checkpoint="1"') == 2
    assert out.rfind('checkpoint="-1"') > out.rfind('checkpoint="1"')


def test_milc_skip_last_forgets_only_the_last_save(tmp_path):
    JitSettings.prepare_milc_input(_settings(tmp_path), "/mnt", 64, 3, 10, skip_last=True)
    text = (tmp_path / "milc.in").read_text()
    assert text.count("save_parallel") == 2
    assert "cfg.2" not in text
    assert text.rstrip().endswith("forget")
