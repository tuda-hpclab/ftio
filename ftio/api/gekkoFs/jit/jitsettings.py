"""
JIT Settings

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import os
import re
import shutil
import socket

import numpy as np
from rich.console import Console

console = Console()


class JitSettings:
    def __init__(self) -> None:
        """sets the internal variables, don't modify this part (except flags if needed).
        only Adjust the paths in the function set_variables
        """

        # app
        ##############
        # can be controlled through command line arguments see jit -h)
        self.app = ""

        # flushing strategies:
        self.strategy = "flush"  # options are "flush", "job_end", and "buffer_size"
        self.job_time = 0  # time in seconds required for strategy "job_end"
        self.buffer_size = 0  # Size in bytes required for strategy "buffer_size"
        self.flush_call = "cp"  # decide if compress (tar) or copy (cp)

        # flags
        ##############
        self.set_tasks_affinity = False  # required for ls and cp
        self.cargo_mode = "posix"  # "parallel" or "posix"
        self.debug_lvl = 0  # >0 FTIO, >1 GKFS & FTIO, >2 GKFS & FTIO & CARGO
        self.verbose = True
        self.verbose_error = True
        self.node_local = True  # execute in node local space or memory
        self.env_var = {}
        self.log_speed = 0.1  # how fast to read the log

        # Variable initialization (don't change)
        ################
        self.dry_run = False
        self.log_suffix = "DPCF"
        self.mode_name = ""  # glass | gekko | pfs -- the run mode as a folder name
        self.run_dir = ""
        self.dir = ""
        self.cluster = False
        self.ignore_mtime = False
        self.parallel_move_threads = 32  # Can be slower if many files are moved
        self.lock_generator = False
        self.lock_consumer = False
        self.handle_new_prediction = "cancel"
        self.job_id = 0
        self.static_allocation = False
        self.ftio_node = ""
        self.single_node = ""
        self.app_nodes = 0
        self.all_nodes = 0
        self.app_nodes_command = ""
        self.ftio_node_command = ""
        self.single_node_command = ""
        self.alloc_call_flags = ""
        self.job_name = ""
        self.ftio_bin_location = ""
        self.gkfs_hostfile = ""
        self.gkfs_proxyfile = ""
        self.mpi_hostfile = ""
        self.parsed_gkfs_daemon = ""
        self.parsed_gkfs_intercept = ""
        self.home = ""
        self.fuse = False

        self.log_dir = ""
        self.result_dir = ""  # logs/<app> -- holds result.json + the plot
        self.gkfs_daemon_log = ""
        self.gkfs_daemon_err = ""
        self.gkfs_proxy_log = ""
        self.gkfs_proxy_err = ""
        self.gkfs_client_log = ""
        self.cargo_log = ""
        self.cargo_err = ""
        self.ftio_log = ""
        self.ftio_err = ""
        self.app_log = ""
        self.app_err = ""
        self.app_start_file = ""
        self.flush_log = ""

        # exclude flags
        ################
        self.exclude_ftio = False
        self.exclude_cargo = False
        self.exclude_daemon = False
        self.exclude_proxy = True
        self.exclude_all = False

        # pid of processes
        ################
        self.ftio_pid = 0
        self.gkfs_daemon_pid = 0
        self.gkfs_fuse_pid = 0
        self.gkfs_proxy_pid = 0
        self.cargo_pid = 0
        self.app_pid = 0

        # parsed variables
        ###################
        self.address_ftio = "127.0.0.1"
        self.port_ftio = "5555"  # port between ftio and gkfs
        self.address_cargo = "127.0.0.1"
        self.port_cargo = "62000"  # port between cargo and ftio

        self.nodes = 1
        self.max_time = None
        # -au merges autocorrelation into the DFT prediction: on jittery signals
        # (qmc and warpx drift by design) DFT alone can find no dominant frequency
        # at all, while the merged result recovers it and lifts confidence past the
        # 0.5 flush threshold sooner (scratchpad/periodic_check.py).
        self.ftio_args = "-m write -v --freq 10 --ingest-workers 4 -au "
        self.gkfs_daemon_protocol = (
            "ofi+verbs"  # "ofi+verbs" #"ofi+sockets"  or "ofi+verbs"
        )
        self.skip_confirm = False
        self.use_mpirun = False
        self.preload_via_export = (
            False  # if True, use legacy --export/mpiexec -x; default wraps in bash -c
        )
        self.gkfs_use_syscall = False
        self.trap_exit = True
        self.soft_kill = True
        self.hard_kill = True
        self.procs = min((len(os.sched_getaffinity(0)), os.cpu_count() or 1, 128))
        self.omp_threads = 64
        self.task_set_0 = ""
        self.task_set_1 = ""
        self.procs_daemon = 0
        self.procs_proxy = 0
        self.procs_cargo = 0
        self.procs_app = 0
        self.procs_ftio = 0
        self.fuse_idle_threads = 0  # finalized in parse_options after procs_app is set
        self.cmd_call = ""

        self.set_cluster_mode()
        self.set_default_procs()
        self.set_variables()

    def __str__(self) -> str:
        """returns the settings in a readable format"""
        return str(self.to_dict())

    def set_cluster_mode(self) -> None:
        """automatically identifies if it's a cluster or local machine"""
        hostname = socket.gethostname()
        if any(x in hostname for x in ("cpu", "mogon", "login", "gs")):
            self.cluster = True
            if any(x in hostname for x in ("login", "mogon")):
                console.print(
                    "[bold red]Execute this script on CPU nodes\n mpiexec still has some bugs[/]"
                )

        console.print(f"[bold  green]CLUSTER MODE: {self.cluster}[/]")

        if "gp" in hostname:
            self.gkfs_use_syscall = True
            self.fuse = True
            # self.procs = os.cpu_count() / 2
            console.print("[bold green]FUSE MODE: ON[/]")
            self.port_ftio = "5558"

    def update(self) -> None:
        """updates the flags and pass variables after the passed options are read.
        This is necessary to adapt to the cluster mode and the installation path
        """
        self.set_flags()
        self.set_variables()
        self.set_absolute_path()
        self.update_settings()

    def update_settings(self):
        """updates settings after command line arguments gave been parsed"""
        # Dry run settings
        if self.dry_run:
            new_name = "Dry_" + self.job_name
            self.alloc_call_flags = self.alloc_call_flags.replace(self.job_name, new_name)
            self.job_name = new_name

        # Gekko settings
        if self.gkfs_use_syscall:
            # already correct
            pass
        else:
            self.gkfs_intercept = self.gkfs_intercept.replace(
                "_intercept.so", "_libc_intercept.so"
            )

    def update_app_nodes(self):
        def _substitute(call: str) -> str:
            if "$APP_PROCS_X_NODES" in call:
                return call.replace(
                    "$APP_PROCS_X_NODES", str(self.procs_app * self.app_nodes)
                )
            if "$APP_NODES" in call:
                return call.replace("$APP_NODES", str(self.app_nodes))
            return call

        if isinstance(self.pre_app_call, list):
            self.pre_app_call = [_substitute(call) for call in self.pre_app_call]
        elif self.pre_app_call:
            self.pre_app_call = _substitute(self.pre_app_call)

    def set_absolute_path(self) -> None:
        self.run_dir = os.path.expanduser(self.run_dir)
        self.dir = os.path.expanduser(os.getcwd())
        self.ftio_bin_location = os.path.expanduser(self.ftio_bin_location)

    def set_default_procs(self) -> None:
        # default values for the procs in proc_list is not passed
        if self.cluster:
            self.procs_proxy = int(np.floor(self.procs / 2))
            self.procs_daemon = int(np.floor(self.procs / 2))
            self.procs_cargo = 2
            self.procs_ftio = self.procs
            self.procs_app = int(np.floor(self.procs / 2))
        else:
            self.procs = 2  # os.cpu_count() / 2 if os.cpu_count() else 4
            self.procs_daemon = 1
            self.procs_proxy = 1
            self.procs_cargo = 2
            self.procs_ftio = 1
            self.procs_app = self.procs

    def update_geko_files(self):
        if not self.exclude_daemon:
            self.gkfs_hostfile = self.gkfs_hostfile.replace(".txt", f"_{self.job_id}.txt")
        if not self.exclude_proxy:
            self.gkfs_proxyfile = self.gkfs_proxyfile.replace(
                ".pid", f"_{self.job_id}.pid"
            )

    def set_mpi_host_file(self, job_id=None):
        if job_id:
            self.mpi_hostfile = f"{self.dir}/mpi_hostfile_{self.job_id}"
        else:
            self.mpi_hostfile = f"{self.dir}/mpi_hostfile"

    def set_flags(self) -> None:
        """sets the flags in case exclude all is specified
        in the options passed
        """

        if (
            self.exclude_ftio
            and self.exclude_cargo
            and self.exclude_daemon
            and self.exclude_proxy
        ):
            self.exclude_all = True

        if self.exclude_all:
            self.exclude_ftio = True
            self.exclude_cargo = True
            self.exclude_daemon = True
            self.exclude_proxy = True

        if not self.cluster and self.nodes > 1:
            self.procs = self.nodes
            self.nodes = 1
            console.print(
                f"[bold green]JIT [bold  cyan]correcting nodes to {self.nodes} and processes to {self.procs} [/]"
            )
        self.log_suffix = "DPCF"
        if self.exclude_daemon:
            self.procs_daemon = 0
            self.log_suffix = self.log_suffix.replace("D", "")
        if self.exclude_proxy:
            self.procs_proxy = 0
            self.log_suffix = self.log_suffix.replace("P", "")
        if self.exclude_cargo:
            self.procs_cargo = 0
            self.log_suffix = self.log_suffix.replace("C", "")
        if self.exclude_ftio:
            self.procs_ftio = 0
            self.log_suffix = self.log_suffix.replace("F", "")

        self.mode_name = JitSettings.mode_label(self.exclude_daemon, self.exclude_ftio)

        if self.set_tasks_affinity:
            self.task_set_0 = f"taskset -c 0-{np.floor(self.procs / 2) - 1:.0f}"
            if self.procs - np.floor(self.procs / 2) >= self.procs_app:
                self.task_set_1 = (
                    f"taskset -c {np.ceil(self.procs / 2):.0f}-{self.procs - 1:.0f}"
                )

    @staticmethod
    def mode_label(exclude_daemon: bool, exclude_ftio: bool) -> str:
        """Name the run mode from the exclude flags, for the log-dir layout.

        The paper compares three: GLASS (GekkoFS + FTIO), GekkoFS-only (no FTIO),
        and the plain parallel FS (`-x`). The exclude flags pin exactly these:

        - no daemon (``-x`` sets exclude_all -> exclude_daemon)  -> ``pfs``
        - daemon on, FTIO off (``-e cargo,ftio``)                -> ``gekko``
        - daemon + FTIO on (``-e cargo``)                        -> ``glass``
        """
        if exclude_daemon:
            return "pfs"
        if exclude_ftio:
            return "gekko"
        return "glass"

    def set_log_dirs(self):
        self.gkfs_daemon_log = os.path.join(self.log_dir, "gekko_daemon.log")
        self.gkfs_daemon_err = os.path.join(self.log_dir, "gekko_daemon.err")
        self.gkfs_proxy_log = os.path.join(self.log_dir, "gekko_proxy.log")
        self.gkfs_proxy_err = os.path.join(self.log_dir, "gekko_proxy.err")
        self.gkfs_client_log = os.path.join(self.log_dir, "gekko_client.log")
        self.gkfs_fuse_log = os.path.join(self.log_dir, "gekko_fuse.log")
        self.gkfs_fuse_err = os.path.join(self.log_dir, "gekko_fuse.err")
        self.cargo_log = os.path.join(self.log_dir, "cargo.log")
        self.cargo_err = os.path.join(self.log_dir, "cargo.err")
        self.ftio_log = os.path.join(self.log_dir, "ftio.log")
        self.ftio_err = os.path.join(self.log_dir, "ftio.err")
        self.app_log = os.path.join(self.log_dir, "app.log")
        self.app_err = os.path.join(self.log_dir, "app.err")
        self.app_start_file = os.path.join(self.log_dir, "app_start.flag")
        self.flush_log = os.path.join(self.log_dir, "flush.log")

    def to_dict(self):  # -> dict[str, Any]:
        # Define a list of attribute names to be promoted to top-level keys
        top_level_keys = [
            "log_suffix",  # This will be renamed to 'mode'
            "app",  # This will be renamed to 'app name'
            "nodes",
            "procs",
            "procs_app",
            "procs_ftio",
            "procs_daemon",
            "procs_proxy",
            "procs_cargo",
            "omp_threads",
            "task_set_0",
            "task_set_1",
        ]

        # Initialize the result dictionary with top-level keys in the desired order
        result = {}

        # Iterate over the top-level keys and add them to the result dictionary if they exist
        for key in top_level_keys:
            value = getattr(self, key, None)
            if value is not None:
                if key == "log_suffix":
                    result["mode"] = value
                elif key == "app":
                    result["app name"] = value
                else:
                    result[key.replace("_", " ")] = value

        # Add the 'settings' dictionary at the end
        result["settings"] = {}

        # Iterate over all instance attributes
        for key, value in self.__dict__.items():
            # Skip keys that have already been added as top-level keys
            if key in top_level_keys:
                continue
            # Replace underscores with spaces in the key for settings
            new_key = key.replace("_", " ")
            result["settings"][new_key] = value

        # Handle specific conditions for certain fields
        if self.exclude_daemon:
            result["settings"]["gkfs mntdir"] = ""
            result["settings"]["gkfs rootdir"] = ""

        return result

    #!##########################
    #! only modify here
    #!##########################
    def set_variables(self) -> None:
        """sets the path variables"""
        # ****** install location ******
        if self.cluster:
            self.install_location = "/beegfs/home/Shared/admire/JIT"

        # ****** job allocation call ******
        # self.alloc_call_flags = "--overcommit --oversubscribe --partition parallel -A nhr-gekko --job-name JIT --no-shell --exclude=cpu0082"
        self.job_name = "JIT"
        # self.alloc_call_flags = f"--overcommit --oversubscribe --partition largemem -A nhr-gekko --job-name {self.job_name} --no-shell --exclude=cpu0081,cpu0082,cpu0083,cpu0084,cpu0401"
        self.alloc_call_flags = f"--overcommit --oversubscribe --partition parallel -A nhr-gekko --job-name {self.job_name} --no-shell --exclude=cpu0081,cpu0082,cpu0083,cpu0084,cpu0085,cpu0086,cpu0087,cpu0088,cpu0401"

        # ? Tools
        # ?##########################
        # self.home = "/lustre/project/nhr-gekko/tarraf"  # mogon
        # self.tmp_dir = self.home # dir tro store stage in and out files
        self.home = str(os.path.expanduser("~"))  # bsc
        self.tmp_dir = os.getenv("STAGE_DIR", "")

        # ****** ftio variables ******
        self.ftio_bin_location = f"{self.home}/FTIO/.venv/bin"

        # ****** gkfs variables ******
        # self.gkfs_dir = f"{self.home}/deps/gekkofs_zmq_install"  # mogon
        # self.gkfs_dir = "/apps/GPP/GEKKOFS/gkfs-master"  # bsc
        self.gkfs_dir = os.getenv("GKFS_DIR", f"{self.home}/deps/install")  # bsc

        if self.parsed_gkfs_daemon:
            self.gkfs_daemon = self.parsed_gkfs_daemon
        else:
            self.gkfs_daemon = f"{self.gkfs_dir}/bin/gkfs_daemon"

        if self.parsed_gkfs_intercept:
            self.gkfs_intercept = self.parsed_gkfs_intercept
        else:
            self.gkfs_intercept = f"{self.gkfs_dir}/lib64/libgkfs_intercept.so"

        self.gkfs_fuse = f"{self.gkfs_dir}/bin/fuse_client"
        self.gkfs_mntdir = "/dev/shm/tarraf_gkfs_mountdir"
        self.gkfs_rootdir = "/dev/shm/tarraf_gkfs_rootdir"
        self.gkfs_hostfile = f"{self.home}/gkfs_hosts.txt"
        self.gkfs_proxy = f"{self.home}/gekkofs/build/src/proxy/gkfs_proxy"
        self.gkfs_proxyfile = "/dev/shm/tarraf_gkfs_proxy.pid"
        self.update_files_with_gkfs_mntdir = []

        # ****** cargo variables ******
        self.cargo_bin = f"{self.gkfs_dir}/bin"  # f"{self.home}/cargo/build/cli"

        # ? APP settings
        # ?##########################
        # ****** app call ******
        #  ├─ IOR
        if "ior" in self.app:
            self.app_call = "./ior "
            self.run_dir = f"{self.home}/ior/src"
            self.app_flags = "-a POSIX -i 4 -o ./iortest -t 128k -b 512m -F"
        #  ├─ HACCIO
        elif "hacc" in self.app:
            self.app_call = "./HACC_ASYNC_IO"
            self.run_dir = f"{self.home}/HACC-IO"
            self.app_flags = "1000000 test_run/mpi"
        # ├─ NEK5000 --> change gkfs_daemon_protocol to socket
        elif "nek" in self.app:
            self.app_call = "./nek5000"
            self.run_dir = "/home/tarrafah/nhr-gekko/shared/run_gkfs_marc"
            self.app_flags = ""
        #  ├─ Wacom++ --> change wacom.json if needed
        elif "wacom" in self.app:
            self.app_call = "./wacommplusplus"
            # self.run_dir = f"{self.home}/wacommplusplus/build"
            # self.run_dir = f"{self.home}/wacommplusplus/roms"
            self.run_dir = f"{self.home}/wacommplusplus/build_new"
            if not self.app_flags:  # default value if app_flags is not set
                self.app_flags = ""
        #  ├─ LAMMPS
        elif "lammps" in self.app:
            # The GLASS driver deck: 168^3 -> 18.97 M atoms -> 1.67 GB per
            # restart (88 B/atom), 12 phases. Sized up from 142^3/8-phases so the
            # app still runs long enough at 32 nodes (~40-60 s) for FTIO to see
            # several checkpoint periods and flush during compute -- at the old
            # size the 32-node run collapsed to 11 s and FTIO found no period.
            # tail keeps the app alive past the last checkpoint. Scale x/y/z and
            # phases/every together to retune (see Cluster Install BSC, section 4).
            # The old cluster setting pointed at /lustre/project/nhr-gekko (Mogon)
            # and ran in.spce.hex, which is a different experiment and cannot
            # resolve on BSC.
            self.app_call = f"{self.home}/mylammps/build/lmp"
            # installed in $HOME, but run from scratch -- see prepare_run_dir
            self.run_dir = self.prepare_run_dir(f"{self.home}/mylammps/glass")
            ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            self.app_flags = self.resolve_app_flags(
                f"-in {self.run_dir}/in.ckpt -v ckptdir {ckptdir} "
                f"-v x 168 -v y 168 -v z 168 "
                # 54 phases: nsteps = phases*every + tail = 818 = ~100 s at 8 compute
                # nodes (0.12 s/step measured in 43420026), 54 restarts x 1.67 GB.
                f"-v every 15 -v nb 15 -v phases 54 -v tail 8",
                ckptdir,
            )
        #  ├─ DLIO
        elif "dlio" in self.app:
            self.app_call = "dlio_benchmark"
            # self.run_dir = "."
            self.run_dir = self.tmp_dir
            workload = os.getenv("WORKLOAD")
            # if not set, take a fixed one
            if workload is None:
                # workload = "cosmoflow_a100"
                # workload = "bert"
                # workload = "bert_small"
                # workload = "bert_v100_pytorch" #paper
                # workload = "bert_v100_pytorch_2" # good
                # workload = "resnet50_v100"# work with fues on bsc
                # workload = "resnet50_v100_new"  # bsc best for real test
                workload = "resnet50_v100_new_small"  # bsc
                # workload = "bert_v100_pytorch_allranksyaml "
                # workload = "unet3d_my_a100"
                # workload = "resnet50_my_a100"
                # workload = "llama_my_7b_zero3"
                # workload = "resnet50_my_a100_pytorch"
            # ensure surrounding spaces
            workload = f" workload={workload} "
        #  ├─ S3D-IO
        elif "s3d" in self.app:
            self.app_call = "/lustre/project/nhr-gekko/shared/S3D-IO/s3d_io.x"
            self.run_dir = "."
            if not self.app_flags:  # default value if app_flags is not set
                self.app_flags = "200 200 200 2 2 2 0 F ."
        #  ├─ WRF
        elif "wrf" in self.app:
            # em_b_wave_glass is our copy of the idealized baroclinic-wave case
            # (run_hours=8, restart_interval=60, history off -> 8 restarts of
            # ~18 MB, ~65 s apart). WRF must NOT run inside the mount: it reads
            # each namelist group with a REWIND and that fails through GekkoFS.
            # So the inputs stay on the parallel FS and only rst_outname points
            # into the mount.
            self.app_call = f"{self.home}/WRF/main/wrf.exe"
            # WRF drops rsl.out.<rank> + rsl.error.<rank> in its cwd: never $HOME.
            self.run_dir = self.prepare_run_dir(
                f"{self.home}/WRF/test/em_b_wave_glass",
                ["namelist.input", "wrfinput_d01"],
            )
            self.app_flags = ""
            self.point_wrf_restarts_at(
                self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            )
        #  ├─ Castro (AMReX)
        elif "castro" in self.app:
            # Sedov blast. fixed_dt + init_shrink=1 + max_level=0 keep the
            # checkpoints wall-clock periodic; without them dt grows ~35x and the
            # gaps run 5 s -> 46 s.
            self.app_call = (
                f"{self.home}/Castro/Exec/hydro_tests/Sedov/Castro3d.gnu.MPI.ex"
            )
            self.run_dir = self.prepare_run_dir(
                f"{self.home}/Castro/Exec/hydro_tests/Sedov", ["inputs.3d.sph"]
            )
            ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            # 60 checkpoint phases. FTIO needs ~40 s of signal to lock on (43420026:
            # P(periodic) hit 88.9% at prediction #8, but the 50 s / 30-phase app was
            # already ending -> zero runtime flushes). Locking at ~40 s into a ~100 s
            # run leaves half the checkpoints to flush during compute. n_cell held at
            # 160 so the many-file checkpoint stage-out does not grow (fragile path).
            self.app_flags = self.resolve_app_flags(
                f"inputs.3d.sph max_step=1440 amr.check_int=24 amr.plot_int=-1 "
                f"amr.max_level=0 castro.fixed_dt=4e-6 castro.init_shrink=1.0 "
                f"amr.check_file={ckptdir}/sedov_3d_sph_chk amr.n_cell = 160 160 160",
                ckptdir,
            )
        #  ├─ WarpX (AMReX)
        elif "warpx" in self.app:
            # WarpX was compute-bound to the point of hiding the FS entirely: 830 s
            # of app against 6 s of stage-out, so glass/gekko/pfs came out within 5%
            # of each other. Halve the compute (max_step 85 -> 40) and checkpoint 5x
            # more often (every 2 steps -> 20 dirs), which raises the I/O fraction
            # by ~10x. n_cell stays at 128 so a checkpoint dir remains ~484 MB.
            self.app_call = f"{self.home}/WarpX/build/bin/warpx.3d"
            self.run_dir = self.prepare_run_dir(f"{self.home}/WarpX/glass", ["inputs"])
            ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            # n_cell 128 -> 192 (~3.4x cells): 20 checkpoints of ~484 MB totalled
            # 9.7 GB against an 830 s app -- gekko staged it in 1.5 s, so the FS was
            # invisible. ~1.6 GB per checkpoint (~32 GB total) makes the I/O matter.
            # Only knob changed; steps/intervals stay from the last calibration.
            self.app_flags = self.resolve_app_flags(
                f"inputs max_step=40 chk.intervals=2 diag1.intervals=1000 "
                f"chk.file_prefix={ckptdir}/chk amr.n_cell = 192 192 192",
                ckptdir,
            )
        #  └─ QMCPACK
        elif "qmc" in self.app:
            # The irregular-workload case: the block time drifts, FTIO finds no
            # dominant frequency and correctly suppresses the flush trigger.
            # <project id> takes an absolute path, so the inputs stay on the
            # parallel FS and only the output goes into the mount.
            self.app_call = f"{self.home}/qmcpack/build/bin/qmcpack"
            self.run_dir = self.prepare_run_dir(f"{self.home}/qmcpack/glass_stagein")
            self.app_flags = "glass.xml"
            self.point_qmcpack_output_at(
                self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            )
        else:
            self.app_call = ""
            self.run_dir = ""
            self.app_flags = ""
        # ? Pre and post app settings
        # ?##########################
        # Application specific calls executed before the actual run. Executed as
        # > ${PRE_APP_CALL}
        # > cd self.run_dir && mpiexec ${some flags} ..${APP_CALL}
        # > ${POST_APP_CALL}
        # ├─ dlio
        if "dlio" in self.app:
            if self.exclude_daemon:
                self.app_flags = (
                    f"{workload} "
                    f"++workload.workflow.generate_data=False "
                    f"++workload.workflow.train=True "
                    f"++workload.workflow.checkpoint=True "
                    f"++workload.dataset.data_folder={self.run_dir}/data "
                    f"++workload.checkpoint.checkpoint_folder={self.run_dir}/checkpoints "
                    f"++workload.output.output_folder={self.run_dir}/hydra_log "
                )
                # self.pre_app_call = f"mpirun -np 8 dlio_benchmark {self.app_flags} ++workload.workflow.generate_data=True ++workload.workflow.train=False"
                # self.pre_app_call = f"mpirun -np $APP_NODES dlio_benchmark {self.app_flags} ++workload.workflow.generate_data=True ++workload.workflow.train=False"
                self.pre_app_call = (
                    f"mpirun -np $APP_PROCS_X_NODES dlio_benchmark "
                    f"{workload} "
                    f"++workload.workflow.generate_data=True "
                    f"++workload.workflow.train=False "
                    f"++workload.workflow.checkpoint=False "
                    f"++workload.dataset.data_folder={self.run_dir}/data "
                    f"++workload.checkpoint.checkpoint_folder={self.run_dir}/checkpoints "
                    f"++workload.output.output_folder={self.run_dir}/hydra_log "
                )
                self.post_app_call = ""
            else:
                # self.run_dir = self.gkfs_mntdir #? don't enable this flag, as the executing node doesn't have this folder
                self.app_flags = (
                    # f"++workload.workflow.generate_data=True ++workload.workflow.train=True ++workload.workflow.checkpoint=True "
                    f"{workload} "
                    f"++workload.workflow.generate_data=False "
                    f"++workload.workflow.train=True "
                    f"++workload.workflow.checkpoint=True "
                    f"++workload.dataset.data_folder={self.gkfs_mntdir}/data "
                    f"++workload.checkpoint.checkpoint_folder={self.gkfs_mntdir}/checkpoints "
                    f"++workload.output.output_folder={self.gkfs_mntdir}/hydra_log "
                )
                dlio_dir = (
                    f"{self.gkfs_mntdir}"  # no stag-in requiored, directly write to gkfs
                )
                # dlio_dir = (
                # f"{self.tmp_dir}/stage-in"  # write to stag-in, than stage-in data
                # )
                # generate_data used to launch straight into the $APP_PROCS_X_NODES
                # mpirun below, so every rank raced to create data/checkpoints/
                # hydra_log (and, deeper, the data/train/ subdir tf.io.TFRecordWriter
                # creates itself) under the FUSE mount at once, which hangs/crashes
                # at scale (65 nodes: 30-min hang; 448 ranks: NotFoundError racing
                # data/train/). Pre-creating the three dirs as a single serial step
                # first removes the race. That step used to silently land on the
                # JIT driver's own node instead of an app node's FUSE mount --
                # setup_core.py's pre_app_call list dispatch only srun-wrapped
                # items containing mpirun/mpiexec, so a bare mkdir ran locally on
                # whatever host was executing the JIT driver process. Fixed by
                # having pre_call() flag-wrap non-MPI list items too (single node,
                # single proc), so the mkdir now runs via srun on an actual app
                # node.
                self.pre_app_call = [
                    f"mkdir -p {dlio_dir}/data {dlio_dir}/checkpoints {dlio_dir}/hydra_log",
                    (
                        f"mpirun -np $APP_PROCS_X_NODES dlio_benchmark "
                        f"{workload} "
                        f"++workload.workflow.generate_data=True "
                        f"++workload.workflow.train=False "
                        f"++workload.workflow.checkpoint=False "
                        f"++workload.dataset.data_folder={dlio_dir}/data "
                        f"++workload.checkpoint.checkpoint_folder={dlio_dir}/checkpoints "
                        f"++workload.output.output_folder={dlio_dir}/hydra_log "
                    ),
                ]
                self.post_app_call = ""
        # ├─ Nek5000
        elif "nek" in self.app:
            if self.exclude_daemon:
                self.pre_app_call = f"echo -e 'turbPipe\\n{self.run_dir}/input' > {self.run_dir}/SESSION.NAME"
                self.post_app_call = f"rm {self.run_dir}/input/*.f* || echo true"
            else:
                self.pre_app_call = f"echo -e 'turbPipe\\n{self.gkfs_mntdir}' > {self.run_dir}/SESSION.NAME"
                self.post_app_call = ""
        # ├─ Wacom++
        elif "wacom" in self.app:
            if self.exclude_daemon:
                # in case a previous simulation fails
                self.pre_app_call = (
                    f"export OMP_NUM_THREADS={self.omp_threads}; ln -sf {self.run_dir}/wacomm.pfs.json {self.run_dir}/wacomm.json; "
                    f"cd {self.run_dir} && rm -rf input restart results processed output; cp -r stage-in/*  {self.run_dir} "
                )
                self.post_app_call = ""
            else:
                # modify wacomm.gkfs.json to include gkfs_mntdir
                self.update_files_with_gkfs_mntdir = [f"{self.run_dir}/wacomm.gkfs.json"]
                self.pre_app_call = f"export OMP_NUM_THREADS={self.omp_threads}; ln -sf {self.run_dir}/wacomm.gkfs.json {self.run_dir}/wacomm.json; "
                self.post_app_call = (
                    f"ln -sf {self.run_dir}/wacomm.pfs.json {self.run_dir}/wacomm.json"
                )
        # ├─ S3D-IO
        elif "s3d" in self.app:
            self.pre_app_call = ""
            self.post_app_call = ""
            if not self.exclude_daemon:
                self.app_flags = self.app_flags.replace(".", f"{self.gkfs_mntdir}")
        # ├─ IOR
        elif "ior" in self.app:
            self.pre_app_call = ""
            self.post_app_call = ""
            if not self.exclude_daemon:
                self.app_flags = self.app_flags.replace(
                    "./iortest", f"{self.gkfs_mntdir}/iortest"
                )
        #  ├─ HACCIO
        elif "hacc" in self.app:
            self.pre_app_call = ""
            self.post_app_call = ""
            if not self.exclude_daemon:
                self.app_flags = self.app_flags.replace("test_run", f"{self.gkfs_mntdir}")
        # ├─ wrf
        elif "wrf" in self.app:
            # Deliberately empty. The old body ran WRF *inside* the mount, which
            # cannot work: WRF reads each namelist group with a REWIND and that
            # fails through GekkoFS ("ERROR while reading namelist diags" -> FATAL
            # -> MPI_ABORT), while the identical file parses fine on a real FS.
            # It also built the call from the `cdf`/`cpf` cluster aliases and
            # $HOME paths, neither of which exists locally.
            # Restarts go into the mount via point_wrf_restarts_at() instead.
            self.pre_app_call = ""
            self.post_app_call = ""
        else:
            self.pre_app_call = ""
            self.post_app_call = ""

        # ? Stage in/out
        # ?##########################
        # ├─ Nek5000
        if "nek" in self.app:
            self.stage_in_path = f"{self.run_dir}/input"
            self.stage_out_path = f"{self.tmp_dir}/stage-out"
        # ├─ Wacom++
        elif "wacom" in self.app:
            self.stage_in_path = f"{self.run_dir}/stage-in"
            self.stage_out_path = f"{self.tmp_dir}/stage-out"
        # ├─ DLIO
        elif "dlio" in self.app:
            self.stage_in_path = f"{self.tmp_dir}/stage-in"
            self.stage_out_path = f"{self.tmp_dir}/stage-out"
        # ├─ LAMMPS
        elif "lammps" in self.app:
            # Nothing to stage in: the glass deck (in.ckpt) is read from run_dir
            # on the parallel FS and writes its restarts straight to ckptdir.
            self.stage_in_path = f"{self.tmp_dir}/stage-in"
            self.stage_out_path = f"{self.tmp_dir}/stage-out"
        # ├─ WRF
        elif "wrf" in self.app:
            # Nothing to stage in: WRF reads namelist.input / wrfinput_d01 from
            # run_dir on the parallel FS (it cannot read them through GekkoFS).
            self.stage_in_path = f"{self.tmp_dir}/stage-in"
            self.stage_out_path = f"{self.tmp_dir}/stage-out"
        # └─ Other
        else:
            self.stage_in_path = f"{self.tmp_dir}/stage-in"
            self.stage_out_path = f"{self.tmp_dir}/stage-out"

        # ? Regex relevant files (move matches out and in)
        # ?##########################
        self.regex_file = f"{self.tmp_dir}/nek_regex4cargo.txt"
        self.env_var = {"CARGO_REGEX": self.regex_file}

        # With GENERATOR (app): At open/create we create an extra .lockgekko file with size = number of opens to that file (it is distributed). We decrease and delete the file on close
        # with CONSUMER (Cargo): At Open we wait until (40 seconds~) for the lock file to dissapear. No modifications needed on the client, it is transparent.
        if self.lock_generator and not self.exclude_daemon:
            self.env_var["LIBGKFS_PROTECT_FILES_GENERATOR"] = "1"  # app, i.e., Gekko
        # else:
        #     self.env_var["LIBGKFS_PROTECT_FILES_GENERATOR"]="0"

        if self.lock_consumer and not self.exclude_cargo:
            self.env_var["LIBGKFS_PROTECT_FILES_CONSUMER"] = "1"  # Cargo
        # else:
        #     self.env_var["LIBGKFS_PROTECT_FILES_CONSUMER"]="0"

        # ? local machine settings
        # ?###############################
        if not self.cluster:
            self.gkfs_daemon_protocol = "ofi+sockets"  # "ofi+tcp"
            self.install_location = "/d/github/JIT"
            self.ftio_bin_location = "/d/github/FTIO/.venv/bin"
            if self.parsed_gkfs_daemon:
                self.gkfs_daemon = self.parsed_gkfs_daemon
            else:
                self.gkfs_daemon = f"{self.install_location}/iodeps/bin/gkfs_daemon"

            if self.parsed_gkfs_intercept:
                self.gkfs_intercept = self.parsed_gkfs_intercept
            else:
                self.gkfs_intercept = (
                    f"{self.install_location}/iodeps/lib/libgkfs_intercept.so"
                )

            self.gkfs_mntdir = "/tmp/jit/tarraf_gkfs_mountdir"
            self.gkfs_rootdir = "/tmp/jit/tarraf_gkfs_rootdir"
            self.gkfs_hostfile = f"{os.getcwd()}/gkfs_hosts.txt"
            self.gkfs_proxy = (
                f"{self.install_location}/gekkofs/build/src/proxy/gkfs_proxy"
            )
            self.gkfs_proxyfile = f"{self.install_location}/tarraf_gkfs_proxy.pid"
            self.cargo_bin = f"{self.install_location}/iodeps/bin"

            self.regex_file = "/tmp/jit/nek_regex4cargo.txt"
            self.env_var = {"CARGO_REGEX": self.regex_file}

            # Stage-out copies land next to stage_out_path. Keep it off the tmpfs
            # backing the rootdir above, or a GB-scale checkpoint set has to fit
            # in there twice (source + copy) and the flush dies with EDQUOT.
            local_stage = self.tmp_dir or "/tmp"
            self.stage_in_path = f"{local_stage}/input"
            self.stage_out_path = f"{local_stage}/output"

            # Create the folder if it doesn't exist
            os.makedirs(self.stage_in_path, exist_ok=True)
            os.makedirs(self.stage_out_path, exist_ok=True)
            with open(os.path.join(self.stage_in_path, "test.txt"), "w"):
                pass

            if "dlio" in self.app:
                # generate data with
                # self.stage_in_path = "/d/github/dlio_benchmark/data"
                workload = " workload=resnet50_small_a100_pytorch "
                if self.exclude_daemon:
                    self.app_flags = (
                        f"{workload} "
                        f"++workload.workflow.generate_data=False "
                        # f"++workload.workflow.generate_data=True "
                        f"++workload.workflow.train=True ++workload.workflow.checkpoint=True "
                        f"++workload.dataset.data_folder={self.run_dir}/data/ ++workload.checkpoint.checkpoint_folder={self.run_dir}/checkpoints/ "
                        f"++workload.output.output_folder={self.run_dir}/hydra_log/ "
                    )
                    self.pre_app_call = (
                        f"mpirun -np  $APP_PROCS_X_NODES dlio_benchmark "
                        f"{workload} "
                        f"++workload.workflow.generate_data=True ++workload.workflow.train=False ++workload.workflow.checkpoint=True "  # ++workload.workflow.evaluation=True "
                        f"++workload.dataset.data_folder={self.run_dir}/data/ ++workload.checkpoint.checkpoint_folder={self.run_dir}/checkpoints/ "
                        f"++workload.output.output_folder={self.run_dir}/hydra_log/ "
                    )
                    self.post_app_call = ""
                else:
                    self.app_flags = (
                        f"{workload} "
                        f"++workload.workflow.generate_data=False ++workload.workflow.train=True ++workload.workflow.checkpoint=True "  # ++workload.workflow.evaluation=True "
                        f"++workload.dataset.data_folder={self.gkfs_mntdir}/data/ ++workload.checkpoint.checkpoint_folder={self.gkfs_mntdir}/checkpoints/ "
                        f"++workload.output.output_folder={self.gkfs_mntdir}/hydra_log/ "
                    )
                    self.pre_app_call = f"mpirun -np 4 dlio_benchmark {self.app_flags} ++workload.workflow.generate_data=True ++workload.workflow.train=False ++workload.dataset.data_folder={self.stage_in_path}/data"
                    self.post_app_call = ""
                # ├─ Nek5000
            elif "nek" in self.app:
                self.run_dir = "/d/benchmark/Nek5000/turbPipe/run"
                self.stage_in_path = "/d/benchmark/Nek5000/turbPipe/run/input"
                if self.exclude_daemon:
                    self.pre_app_call = "echo -e 'turbPipe\\n/d/benchmark/Nek5000/turbPipe/run/input' > /d/benchmark/Nek5000/turbPipe/run/SESSION.NAME"
                    self.post_app_call = (
                        "rm /d/benchmark/Nek5000/turbPipe/run/input/*.f* || true"
                    )
                else:
                    self.pre_app_call = f"echo -e 'turbPipe\\n{self.gkfs_mntdir}' > /d/benchmark/Nek5000/turbPipe/run/SESSION.NAME"
                    self.post_app_call = f"rm {self.stage_out_path}/*.f* || true"
            elif "s3d" in self.app:
                self.app_call = "/d/benchmark/S3D-IO/s3d_io.x"
                # execute "mkdir -p /d/benchmark/S3D-IO/input && touch /d/benchmark/S3D-IO/input/test"
                self.stage_in_path = "/tmp/input"
                if not self.exclude_daemon:
                    self.app_flags = re.sub(r"/[^\s]+", self.gkfs_mntdir, self.app_flags)
            elif "hacc" in self.app:
                self.run_dir = "/d/github/HACC-IO"
                if not self.exclude_daemon:
                    self.app_flags = re.sub(r"/[^\s]+", self.gkfs_mntdir, self.app_flags)
            elif "ior" in self.app:
                self.run_dir = "/d/github/ior/src"
                self.app_flags = re.sub(r"/[^\s]+", self.gkfs_mntdir, self.app_flags)
            elif "lammps" in self.app:
                # Writes restart files straight to ${ckptdir} (no temp rename), so
                # point that at the gkfs mountdir.
                # 142³ -> 11.45 M atoms -> 1.008 GB per checkpoint (88 B/atom).
                # every=10 steps ~ 21 s of compute between the ~2.4 s writes, so
                # the 8 phases are well separated; tail=5 keeps the app alive
                # past the last checkpoint long enough for FTIO to predict it.
                # Size / period knobs and cluster paths in LAMMPS.md.
                self.app_call = "/d/benchmark/lammps/build/lmp"
                self.run_dir = "/d/benchmark/lammps/glass"
                ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
                self.app_flags = self.resolve_app_flags(
                    f"-in /d/benchmark/lammps/glass/in.ckpt -v ckptdir {ckptdir} "
                    f"-v x 142 -v y 142 -v z 142 "
                    f"-v every 10 -v nb 10 -v phases 8 -v tail 5",
                    ckptdir,
                )
            elif "wrf" in self.app:
                # Idealized baroclinic wave (em_b_wave): self-contained, writes
                # periodic wrfrst_d01_* restart checkpoints (period = namelist
                # restart_interval). Build / size knobs / cluster setup in WRF.md.
                #
                # wrf.exe has no CLI: it reads namelist.input from the cwd. Running
                # *in* the mount does not work -- WRF reads each namelist group with
                # a REWIND, and that fails through GekkoFS, so it aborts with
                # "ERROR while reading namelist diags". So keep the inputs on the
                # real filesystem and redirect only the restarts into the mount with
                # the namelist's own rst_outname key.
                #
                # em_b_wave_glass is our own copy of the case (stock em_b_wave is
                # left alone): run_hours=8, restart_interval=60, history off, which
                # yields 8 restarts of ~18 MB about 55 s apart.
                self.app_call = "/d/benchmark/WRF/main/wrf.exe"
                self.run_dir = "/d/benchmark/WRF/test/em_b_wave_glass"
                self.app_flags = ""
                ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
                self.point_wrf_restarts_at(ckptdir)
                # The wrf branch above (not cluster-gated) leaves a pre_app_call
                # built from the `cdf`/`cpf` cluster aliases and $HOME paths, which
                # do not exist here. Clear it; the namelist rewrite above does the
                # job on its own.
                self.pre_app_call = ""
                self.post_app_call = ""
            elif "qmc" in self.app:
                # QMCPACK VMC (self-contained heg case): checkpoints config.h5
                # every block (checkpoint="1"); write-only, rolling file. Size =
                # walkers/system, period = checkpoint blocks. Notes in Qmpack.md.
                #
                # Same story as WRF: the output prefix comes from <project id> in
                # the input, not from a flag, so the files land in the cwd. Unlike
                # WRF though, <project id> accepts an absolute path, so point it at
                # the mount and leave the xml inputs on the real filesystem.
                # glass_stagein is our own input set (the glass/ dir also carries
                # previous outputs).
                self.app_call = "/d/benchmark/qmcpack/build/bin/qmcpack"
                self.run_dir = "/d/benchmark/qmcpack/glass_stagein"
                self.app_flags = "glass.xml"
                ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
                self.point_qmcpack_output_at(ckptdir)
                self.pre_app_call = ""
                self.post_app_call = ""
            elif "castro" in self.app:
                # Castro/AMReX Sedov: real hydro, writes periodic AMReX checkpoint
                # DIRECTORIES (sedov_3d_sph_chk*) every amr.check_int steps;
                # write-only. Size = amr.n_cell/max_level, period = check_int.
                # Note: AMReX checkpoints are multi-file dirs. Build/knobs in Castro.md.
                self.app_call = (
                    "/d/benchmark/Castro/Exec/hydro_tests/Sedov/Castro3d.gnu.MPI.ex"
                )
                self.run_dir = "/d/benchmark/Castro/glass"
                # inputs.3d.sph is the stock upstream deck; everything below is a
                # command-line override (AMReX reads "name = v1 v2 v3" from argv).
                #
                # fixed_dt + init_shrink=1 + max_level=0 are what make the I/O
                # *wall-clock* periodic. Stock Castro shrinks the initial dt 100x
                # and grows it back while AMR refines, so a constant check_int
                # yields phases 5 s apart at the start and 46 s apart at the end,
                # and FTIO correctly reports no dominant frequency.
                #
                # 160^3 -> 4.10 M cells -> 617 MB per checkpoint, 4.55 s per step.
                # check_int=2 -> ~9 s of compute per phase; max_step=18 -> 10
                # checkpoints (step 0,2..18). check_file must point into the mount,
                # else Castro writes to run_dir and gkfs never sees the I/O.
                ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
                self.app_flags = (
                    f"inputs.3d.sph max_step=18 amr.check_int=2 amr.plot_int=-1 "
                    f"amr.max_level=0 castro.fixed_dt=4e-6 castro.init_shrink=1.0 "
                    f"amr.check_file={ckptdir}/sedov_3d_sph_chk "
                    f"amr.n_cell = 160 160 160"
                )
            elif "warpx" in self.app:
                # WarpX (AMReX plasma PIC): real PIC compute, writes periodic
                # AMReX checkpoint DIRECTORIES (diags/chk*) every chk.intervals
                # steps; write-only. Size = amr.n_cell, period = chk.intervals.
                # Build/knobs in WarpX.md (reuses the local AMReX).
                self.app_call = "/d/benchmark/WarpX/build/bin/warpx.3d"
                self.run_dir = "/d/benchmark/WarpX/glass"
                # Stock `inputs`; everything below is a command-line override.
                # 9 checkpoints (steps 0,10..80) + a 5-step tail. diag1 is pushed
                # out of range so only chk* lands in the trace, and the prefix must
                # point into the mount or gkfs never sees the writes.
                # Stock 64x32x32 gives 17 MB checkpoints and sub-second steps -- the
                # phases collapse into each other. 128^3 -> 484 MB per checkpoint and
                # ~1 s per step, so the 9 phases sit ~10 s apart.
                ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
                self.app_flags = (
                    f"inputs max_step=85 chk.intervals=10 diag1.intervals=1000 "
                    f"chk.file_prefix={ckptdir}/chk "
                    f"amr.n_cell = 128 128 128"
                )

        # app_call is final only here -- pick the regexes now.
        self.select_regexes()

    def resolve_app_flags(self, default: str, ckptdir: str) -> str:
        """Return the user's --app-flags if given, otherwise the tuned default.

        The driver apps used to overwrite app_flags unconditionally, so --app-flags
        was silently ignored for exactly the apps whose phases you want to retune
        (LAMMPS, Castro, WarpX, ...) and the only way to change a phase was to edit
        this file. Overriding matters on a cluster: more ranks shrink the compute
        gap, so the size and period knobs have to be scaled per machine.

        The checkpoint directory is only known at runtime, so a user-supplied
        string can refer to it as `{ckptdir}` (and to the run directory as
        `{run_dir}`); both are substituted here.

        Args:
            default (str): The flags to use when the user gave none.
            ckptdir (str): Directory the app should write its checkpoints to.

        Returns:
            str: The flags to launch the application with.
        """
        if not self.app_flags:
            return default
        flags = self.app_flags.replace("{ckptdir}", ckptdir).replace(
            "{run_dir}", self.run_dir
        )
        if ckptdir not in flags:
            # The checkpoints would land outside the mount, GekkoFS would see no
            # I/O at all, and FTIO would make zero predictions -- with nothing in
            # the log to say why. Say it here instead.
            console.print(
                f"[bold yellow]--app-flags does not mention the checkpoint dir "
                f"({ckptdir}). Use the {{ckptdir}} placeholder, or the app will "
                f"write outside the mount and FTIO will see nothing.[/]"
            )
        return flags

    def prepare_run_dir(self, deck_dir: str, files: list[str] | None = None) -> str:
        """Copy an app's deck to scratch and return that as the run directory.

        The apps are *installed* under $HOME, but they must not *run* there: the
        cwd is where they drop their per-rank output. WRF alone writes
        rsl.out.<rank> + rsl.error.<rank>, so a 16 x 112 job would land ~3600 files
        in GPFS home -- which has a quota and is not meant for parallel I/O.

        So the deck is copied to $STAGE_DIR/run/<app> and the app runs from there.
        The install tree stays clean, and a rerun starts from a fresh copy.

        Falls back to `deck_dir` when there is no tmp_dir (e.g. STAGE_DIR unset
        locally), so nothing changes off the cluster.

        Args:
            deck_dir (str): Where the deck lives in the install tree.
            files (list[str] | None): Which files to copy; None copies the lot.

        Returns:
            str: The directory the application should run in.
        """
        if not self.tmp_dir:
            return deck_dir

        run_dir = os.path.join(self.tmp_dir, "run", self.app)
        try:
            os.makedirs(run_dir, exist_ok=True)
            names = files if files is not None else os.listdir(deck_dir)
            for name in names:
                src = os.path.join(deck_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(run_dir, name))
                elif files is not None:
                    # Named explicitly, so its absence is a mistake, not a choice:
                    # the app would start in an empty directory and fail obscurely.
                    console.print(
                        f"[bold red]Deck file missing: {src}. "
                        f"{self.app} will run without it.[/]"
                    )
        except OSError as e:
            console.print(
                f"[yellow]Could not stage the deck into {run_dir} ({e}); "
                f"running from {deck_dir} instead[/]"
            )
            return deck_dir
        return run_dir

    def point_wrf_restarts_at(self, ckptdir: str) -> None:
        """Rewrite rst_outname in the WRF namelist so restarts land in `ckptdir`.

        wrf.exe takes no command line arguments: the only knob for where the
        restart files go is the namelist's rst_outname. The namelist lives in
        em_b_wave_glass, our own copy of the case, so rewriting it leaves the
        stock em_b_wave deck untouched.

        Done in Python rather than as a pre_app_call so the rewrite works the same
        locally and on the cluster, without depending on shell aliases.

        Args:
            ckptdir (str): Directory the restart files should be written to.
        """
        namelist = os.path.join(self.run_dir, "namelist.input")
        target = (
            f" rst_outname                         = '{ckptdir}/wrfrst_d<domain>_<date>',"
        )
        try:
            with open(namelist) as f:
                lines = f.read().splitlines()
            rewritten = [
                target if line.lstrip().startswith("rst_outname") else line
                for line in lines
            ]
            if rewritten != lines:
                with open(namelist, "w") as f:
                    f.write("\n".join(rewritten) + "\n")
            # set_dir_gekko finalizes gkfs_mntdir to node-local scratch AFTER this
            # runs, so rst_outname still holds the default mount dir. Register the
            # namelist so the mntdir-rewrite pass corrects it; otherwise WRF writes
            # its restarts to a non-existent dir.
            if (
                "_gkfs_mountdir" in ckptdir
                and namelist not in self.update_files_with_gkfs_mntdir
            ):
                self.update_files_with_gkfs_mntdir.append(namelist)
        except OSError as e:
            console.print(f"[yellow]Could not set rst_outname in {namelist}: {e}[/]")

    def point_qmcpack_output_at(self, ckptdir: str) -> None:
        """Rewrite <project id> in the QMCPACK input so output lands in `ckptdir`.

        qmcpack has no flag for the output prefix -- it comes from the project id,
        which is why the files otherwise appear in the cwd. The id does accept an
        absolute path, so the inputs can stay on the real filesystem while the
        rolling config.h5 checkpoint is written into the mount.

        Args:
            ckptdir (str): Directory the QMCPACK output should be written to.
        """
        xml = os.path.join(self.run_dir, "glass.xml")
        try:
            with open(xml) as f:
                text = f.read()
            rewritten = re.sub(
                r'<project id="[^"]*glass_heg"',
                f'<project id="{ckptdir}/glass_heg"',
                text,
            )
            if rewritten != text:
                with open(xml, "w") as f:
                    f.write(rewritten)
            # set_dir_gekko finalizes gkfs_mntdir to node-local scratch AFTER this
            # runs, so the path just written still holds the default mount dir.
            # Register the file so the mntdir-rewrite pass corrects it; otherwise
            # QMCPACK writes to a non-existent dir and aborts ("cannot open file").
            if (
                "_gkfs_mountdir" in ckptdir
                and xml not in self.update_files_with_gkfs_mntdir
            ):
                self.update_files_with_gkfs_mntdir.append(xml)
        except OSError as e:
            console.print(f"[yellow]Could not set the project id in {xml}: {e}[/]")

    def select_regexes(self) -> None:
        """Pick the flush / stage-out / stage-in patterns for the current app.

        Keyed off ``app_call``, so this must run *after* the cluster and local
        blocks have finalized it. Running it earlier picks the pattern of
        whatever app_call happened to hold at the time, which silently left
        castro/warpx/qmc with an empty flush regex -- a flush that stages
        nothing at all.
        """
        # ├─ Nek5000
        if "nek" in self.app_call:
            self.regex_flush_match = ".*/[a-zA-Z0-9]*turbPipe0\\.f\\d+"
            self.regex_stage_out_match = ".*/[a-zA-Z0-9]*turbPipe0\\.f\\d+"  # ".*"
            self.regex_stage_in_match = ".*"
        # ├─ Wacom++
        elif "wacom" in self.app_call:
            self.regex_flush_match = ".*/(history|restart|output)/.*\\.(nc|json)$"
            # self.regex_flush_match = ".*/output/.*\\.nc$"
            # self.regex_stage_out_match = ".*"
            self.regex_stage_out_match = ".*/(history|restart|output)/.*\\.(nc|json)$"
            self.regex_stage_in_match = ".*"
        # ├─ DLIO
        elif "dlio" in self.app_call:
            # The lookahead makes the per-epoch dir a single flush unit (as castro
            # does); the second alternative keeps the flat checkpoints/*.pt layout.
            self.regex_flush_match = (
                ".*/checkpoints/global_epoch\\d+_step\\d+(?=/)|.*/(checkpoints)/.*"
            )
            self.regex_stage_out_match = (
                ".*/checkpoints/global_epoch\\d+_step\\d+(?=/)|.*/(checkpoints)/.*"
            )
            self.regex_stage_in_match = ".*"
        # ├─ LAMMPS
        elif "lmp" in self.app_call:
            # An empty flush pattern matches nothing, so FTIO would trigger on
            # time but stage 0 items and the checkpoints would pile up in the
            # rootdir until the post-app copy hits EDQUOT. The mountdir only
            # ever holds ckpt.restart.<step>, so match them explicitly.
            #
            # Tried nfile 64 (multi-file "%" restart, regex
            # ckpt\.restart\.\d+\.(\d+|base)$) to let GekkoFS's daemons fan
            # out. Confirmed on BSC (43561676, 448 ranks): it segfaults inside
            # libc _IO_ferror under GekkoFS's LD_PRELOAD intercept specifically
            # -- identical config on plain GPFS runs fine. Reverted to
            # single-writer until that's resolved on the GekkoFS side.
            self.regex_flush_match = ".*/ckpt\\.restart\\.\\d+$"
            self.regex_stage_out_match = ".*"
            self.regex_stage_in_match = ".*"
        # ├─ S3D-IO
        elif "s3d" in self.app_call:
            self.regex_flush_match = ".*/pressure_wave_test\\..*\\.field\\.nc$"
            self.regex_stage_out_match = ".*"
            self.regex_stage_in_match = ".*"
        elif "wrf" in self.app_call:
            # The checkpoints are the restart files wrfrst_d<domain>_<date>. The
            # old pattern matched wrfout_* (history) and rsl.* (per-rank logs)
            # instead, so it would have flushed the logs and never a checkpoint.
            # Running in the mount also puts namelist.input / wrfinput_d01 / rsl.*
            # there, and none of those may be staged out mid-run: wrf keeps the
            # rsl logs open, and removing an input under a running app is fatal.
            self.regex_flush_match = ".*/wrfrst_d\\d+_.*$"
            self.regex_stage_out_match = ".*/wrfrst_d\\d+_.*$"
            self.regex_stage_in_match = ".*"
        # ├─ Castro (AMReX)
        elif "Castro" in self.app_call or "castro" in self.app_call:
            # AMReX writes each checkpoint as a directory tree with a dot-free
            # name (<mnt>/sedov_3d_sph_chk<step>/Header, .../Level_0/Cell_D_*).
            # Match the top-level checkpoint directory (no end anchor) so the
            # whole tree is flushed as a single unit. The (?=/) keeps AMReX's
            # "<chk>.old.<pid>" rename artifacts from matching as a bare file
            # unit whose path no longer exists.
            self.regex_flush_match = ".*/sedov_3d_sph_chk\\d+(?=/)"
            self.regex_stage_out_match = ".*/sedov_3d_sph_chk\\d+(?=/)"
            self.regex_stage_in_match = ".*"
        # ├─ WarpX (AMReX)
        elif "warpx" in self.app_call:
            # WarpX (AMReX) writes checkpoint directories <mnt>/chk<step>/...
            # Match the top-level checkpoint directory so the tree flushes as one.
            self.regex_flush_match = ".*/chk\\d+(?=/)"
            self.regex_stage_out_match = ".*/chk\\d+(?=/)"
            self.regex_stage_in_match = ".*"
        # ├─ QMCPACK
        elif "qmcpack" in self.app_call:
            # QMCPACK rolls a single config file <mnt>/glass_heg.s000.config.h5.
            self.regex_flush_match = ".*/glass_heg\\.s\\d+\\.config\\.h5$"
            self.regex_stage_out_match = ".*/glass_heg\\.s\\d+\\.config\\.h5$"
            self.regex_stage_in_match = ".*"
        # └─ Other
        else:
            self.regex_flush_match = ""
            self.regex_stage_out_match = ".*"
            self.regex_stage_in_match = ".*"

        self.regex_match = self.regex_flush_match
