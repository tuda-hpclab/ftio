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
    "lammps": (r".*/ckpt\.restart\.\d+\.(\d+|base)$", f"{MNT}/ckpt.restart.80.5"),
    "castro": (r".*/sedov_3d_sph_chk\d+(?=/)", f"{MNT}/sedov_3d_sph_chk00010/Header"),
    "warpx": (r".*/chk\d+(?=/)", f"{MNT}/chk000010/WarpXHeader"),
    "qmcpack": (r".*/glass_heg\.s\d+\.config\.h5$", f"{MNT}/glass_heg.s000.config.h5"),
    "wrf": (r".*/wrfrst_d\d+_.*$", f"{MNT}/wrfrst_d01_0001-01-01_01:00:00"),
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


def test_lammps_regex_matches_the_base_metadata_file_too():
    # nfile 64 splits each checkpoint into per-group data files (.0, .1, ...)
    # plus one ckpt.restart.<step>.base metadata file -- confirmed against a
    # real local LAMMPS run, not assumed from the docs. Missing it here means
    # it never gets staged out and piles up in the mount.
    lammps = re.compile(APP_PATTERNS["lammps"][0])
    assert lammps.match(f"{MNT}/ckpt.restart.80.base")
    assert lammps.match(f"{MNT}/ckpt.restart.80.0")


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
