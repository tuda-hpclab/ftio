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

import hashlib
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
        # GLASS_STRATEGY=job_end keeps metrics + FTIO but never flushes mid-run, which
        # isolates the cost of the metrics messages from the cost of the flushes
        self.strategy = os.getenv(
            "GLASS_STRATEGY", "flush"
        )  # flush, job_end, buffer_size
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
        # execute in node local space (disk) or memory (/dev/shm, tmpfs).
        # False (memory) is the default: job 43710099 (2026-07-23) measured
        # --use_mem roughly halving app time for both glass and gekko over
        # the node-local-disk path, on IOR at 9 and 19 nodes. --use_mem is
        # now a no-op (already False); kept for explicitness on the command
        # line rather than removed.
        #
        # 2026-09-23: --use_mem's own help text still says "GekkoFS uses
        # nodelocal space (default) or memory (if this flag is passed)" --
        # that stopped being true when the default below was flipped to
        # False (memory) after 43710099, and setup_helper.py's handler
        # (`if parsed_args.use_mem: settings.node_local = False`) is a
        # no-op either way the flag is used. There is currently no CLI path
        # back to node-local-disk mode (`/scratch/tmp` on BSC, real
        # node-local SSD, not RAM) at all. JIT_NODE_LOCAL=1 restores it
        # without touching --use_mem's now-misleading behavior or any other
        # app's default -- e.g. for DLIO's large-checkpoint /dev/shm-fills-up
        # problem (job 46323800, 2026-09-22), where the SSD path has real
        # disk capacity instead of RAM-backed tmpfs.
        self.node_local = os.getenv("JIT_NODE_LOCAL", "") not in ("", "0")
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
        self.defer_delete = False  # --defer_delete: FTIO copies, keeps the source
        # STAGE_OUT_KEEP=1: the final stage-out copies only, no delete (the burst
        # buffer goes away with the job). STAGE_OUT_ALL_NODES=1: every app node
        # copies a share of the files, STAGE_OUT_TASKS_PER_NODE tasks per node.
        self.stage_out_keep = os.getenv("STAGE_OUT_KEEP", "0") == "1"
        self.stage_out_all_nodes = os.getenv("STAGE_OUT_ALL_NODES", "0") == "1"
        self.stage_out_tasks_per_node = int(os.getenv("STAGE_OUT_TASKS_PER_NODE", "4"))
        # LAUNCH_WARMUP=1: an untimed srun with the app's node list and task layout
        # before the app, so a slow first job step in the allocation is not timed.
        self.launch_warmup = os.getenv("LAUNCH_WARMUP", "0") == "1"
        # GLASS_FLUSH_BATCH=K: FTIO copies up to K files of one directory per cp
        # call instead of one process per file (see posix_control._batches).
        self.flush_batch = int(os.getenv("GLASS_FLUSH_BATCH", "0"))
        # GLASS_FREQ_SCALE=s: the trigger uses s x FTIO's predicted frequency, a
        # deliberately wrong prediction (0.5: period twice too long) for the
        # misprediction study (SC26 R4). 1 leaves the prediction untouched.
        self.freq_scale = float(os.getenv("GLASS_FREQ_SCALE", "1"))
        # GLASS_EAGER_SIZE=1: also copy a checkpoint once it has the constant size of
        # the earlier ones and stopped changing (stage_data.eager_size_loop).
        self.eager_size = os.getenv("GLASS_EAGER_SIZE", "0") == "1"
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
        self.ftio_state_dir = ""
        # time.time() when the app returned; the GLASS stage-out timer starts here
        self.app_end_time = 0.0

        # exclude flags
        ################
        self.exclude_ftio = False
        self.exclude_cargo = False
        self.exclude_daemon = False
        self.exclude_proxy = True
        self.exclude_all = False
        self.exclude_stage_in = False

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
        # Additive, opt-in escape hatch to append extra raw FTIO flags (e.g.
        # "--transformation astft" or "-w adwin") without touching the base
        # string above or the sbatch script's leg-dispatch logic -- unset
        # (the default) leaves every existing run byte-for-byte unchanged.
        extra_ftio_args = os.getenv("FTIO_EXTRA_ARGS", "")
        if extra_ftio_args:
            self.ftio_args += f" {extra_ftio_args}"
        # Seconds between the GekkoFS client's metric pushes to FTIO. FTIO only
        # sees new data at this cadence, so an app phase of a few seconds (S3D-IO
        # at small N) ends before FTIO's first prediction. Default 5 = every run
        # so far.
        self.metrics_flush_interval = int(os.getenv("METRICS_FLUSH_INTERVAL", "5"))
        # At a short interval the ranks' pushes leave no 1 s gap, so FTIO's
        # receive loop never closes a batch until the app stops writing. Cap
        # the batch at one interval -- only when the interval was set
        # explicitly, so default runs keep the unbounded batch.
        if os.getenv("METRICS_FLUSH_INTERVAL"):
            self.ftio_args += f" --max-batch-window {self.metrics_flush_interval}"
        self.gkfs_daemon_protocol = (
            "ofi+verbs"  # "ofi+verbs" #"ofi+sockets"  or "ofi+verbs"
        )
        # Extra gkfs_daemon flags, e.g. "--auto-sm" (intra-node IPC via shared memory)
        self.gkfs_daemon_args = os.getenv("GKFS_DAEMON_ARGS", "")
        self.skip_confirm = False
        self.use_mpirun = False
        self.preload_via_export = (
            False  # if True, use legacy --export/mpiexec -x; default wraps in bash -c
        )
        self.gkfs_use_syscall = True
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
        self.cpus_per_task_app = 0  # --cpus-per-task; defaults to procs_app below
        self.stack_unlimited = False  # wrap the app launch in `ulimit -s unlimited`
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

        # Gekko settings. Syscall interception is the default and should stay
        # that way: the libc interceptor fabricates a FILE* with only
        # _mode/_flags/_fileno set and never intercepts ferror(), so any app
        # calling it locks a garbage pointer and segfaults (LAMMPS does, after
        # every restart -- see glass/gekkofs_ferror_bug.md). If syscall
        # interception does not work for an app, try FUSE, not libc.
        if not self.gkfs_use_syscall:
            self.gkfs_intercept = self.gkfs_intercept.replace(
                "_intercept.so", "_libc_intercept.so"
            )
        # GKFS_SI_PRELOAD=<patched libsyscall_intercept.so.0>: the stock
        # intercept_wrapper restores ymm0-7 with vmovups and no vzeroupper, so the
        # upper YMM state stays dirty and every legacy-SSE instruction of the app
        # takes an SSE/AVX mix assist (MILC compute 44 -> 74 s, BSC 48744154).
        # The client's RPATH pins the stock library, so LD_LIBRARY_PATH does not
        # help; preloading the patched one first makes the soname resolve to it.
        si_preload = os.getenv("GKFS_SI_PRELOAD")
        if si_preload and self.gkfs_use_syscall:
            self.gkfs_intercept = f"{si_preload}:{self.gkfs_intercept}"

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
            # next to mpi_hostfile in the job dir, not in $HOME: the daemons
            # create <hostfile>.ready beside it, and a job killed at its limit
            # never gets to clean that up
            name = os.path.basename(self.gkfs_hostfile).replace(
                ".txt", f"_{self.job_id}.txt"
            )
            self.gkfs_hostfile = os.path.join(self.dir or self.home, name)
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
        # FTIO's run-time files (app start flag, flush log with its registries and
        # locks, bandwidth.json) stay on the node-local /dev/shm of the jit node, where
        # FTIO runs too. In the log dir on GPFS, flock and file creation waited up to
        # 82 s while the app ran (LAMMPS 2+1, BSC 48864899), which stalled the flush
        # workers and the predictions. save_ftio_state copies them to the log dir.
        tag = hashlib.md5(self.log_dir.encode()).hexdigest()[:12]
        self.ftio_state_dir = f"/dev/shm/{os.getenv('USER', 'jit')}_ftio_state_{tag}"
        self.app_start_file = os.path.join(self.ftio_state_dir, "app_start.flag")
        self.flush_log = os.path.join(self.ftio_state_dir, "flush.log")

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
        # BSC default: the maintained module build (module load GekkoFS/master-0.9.6).
        # Override with GKFS_DIR for a hand-built tree.
        self.gkfs_dir = os.getenv("GKFS_DIR", "/apps/GPP/GEKKOFS/gkfs-master")

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
        # Cargo is a separate project and is not shipped in the GekkoFS module,
        # so it is not derived from gkfs_dir. Override with CARGO_DIR.
        self.cargo_bin = os.getenv("CARGO_DIR", f"{self.home}/deps/install/bin")

        # ? APP settings
        # ?##########################
        # ****** app call ******
        self.cpus_per_task_app = self.procs_app
        #  ├─ IOR
        if "ior" in self.app:
            self.app_call = "./ior "
            # installed in $HOME, but run from scratch -- see prepare_run_dir
            self.run_dir = self.prepare_run_dir(f"{self.home}/ior/src", ["ior"])
            self.app_flags = "-a POSIX -i 4 -o ./iortest -t 128k -b 512m -F"
        #  ├─ HACCIO
        elif "hacc" in self.app:
            self.app_call = "./HACC_ASYNC_IO"
            # installed in $HOME, but run from scratch -- see prepare_run_dir
            self.run_dir = self.prepare_run_dir(f"{self.home}/HACC-IO", ["HACC_ASYNC_IO"])
            # 38 B per particle -> 1 M particles = 38 MB per rank and checkpoint.
            self.app_flags = f"{os.getenv('HACC_PARTICLES', '1000000')} test_run/mpi"
        #  ├─ E3SM-IO
        elif "e3sm" in self.app:
            # E3SM's history output through PnetCDF, replayed from a production
            # decomposition map (F case = atmosphere, 512 procs). Per file: -r
            # records, each after -t s of emulated compute. E3SM_FILES (read by
            # our patched e3sm_io) rotates the output into e3sm.<k>_h{0,1}.nc,
            # one file pair per checkpoint, so GLASS can flush finished files.
            # E3SM_STRATEGY=blob packs each rank's data into one contiguous block
            # in a per-node subfile <file>.<NNNN>; canonical issues ~54 M tiny
            # noncontiguous requests per F-case file. E3SM_SCALE multiplies the
            # data volume (-e: last dimension times n).
            self.app_call = "./e3sm_io"
            self.run_dir = self.prepare_run_dir(f"{self.home}/E3SM-IO/src", ["e3sm_io"])
            e3sm_map = os.getenv(
                "E3SM_MAP", f"{self.home}/E3SM-IO/datasets/f_case_48602x72_512p.nc"
            )
            self.app_flags = (
                f"-k -y 1 -r {os.getenv('E3SM_RECS', '1')} "
                f"-x {os.getenv('E3SM_STRATEGY', 'canonical')} "
                f"-e {os.getenv('E3SM_SCALE', '1')} "
                f"-t {os.getenv('E3SM_COMPUTE_S', '30')} -o test_run/e3sm.nc {e3sm_map}"
            )
        # ├─ NEK5000 --> change gkfs_daemon_protocol to socket
        elif "nek" in self.app:
            # turbChannel from NekExamples, scaled to a 32^3 box with genbox
            # (32768 elements, lx1=8, lpmin=112 -> at least 2 nodes at 56/node).
            # Every dump is one chan0.f<N> written by all ranks through MPI-IO,
            # the N->1 pattern of the GekkoFS paper's Nek5000 run (3.8x). The
            # mesh and par are read from the session dir, which on the GekkoFS
            # legs is the mount, so both are staged in from run_dir/input.
            self.nek_case = "chan"
            # NEK_DIR=chan64: the same case on a 64^3 box (lpmin=896, >= 16+1).
            nek_dir = f"{self.home}/nek/{os.getenv('NEK_DIR', self.nek_case)}"
            self.run_dir = self.prepare_run_dir(nek_dir, ["nek5000"])
            self.app_call = f"{self.run_dir}/nek5000"
            self.app_flags = ""
            self.prepare_nek_input(
                nek_dir,
                int(os.getenv("NEK_STEPS", "100")),
                int(os.getenv("NEK_EVERY", "10")),
            )
        #  ├─ NAS BT-IO
        elif "btio" in self.app:
            # NPB 3.4.3 BT with SUBTYPE=simple: every rank writes its cells of
            # the solution with independent MPI_File_write_at, the 18x case of
            # the GekkoFS paper. Our simple_mpiio.f90 puts each dump in its own
            # file, <BTIO_DIR>/btio.simple.out.<n>, so a finished dump can be
            # flushed; the run reads all of them back to verify the solution.
            btio_class = os.getenv("BTIO_CLASS", "D")
            binary = f"bt.{btio_class}.x.mpi_io_simple"
            self.run_dir = self.prepare_run_dir(f"{self.home}/NPB3.4-MPI/bin", [binary])
            outdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            self.app_call = f"env BTIO_DIR={outdir} {self.run_dir}/{binary}"
            self.app_flags = ""
            self.prepare_btio_input(
                btio_class,
                int(os.getenv("BTIO_STEPS", "0")),
                int(os.getenv("BTIO_EVERY", "5")),
            )
        #  ├─ VPIC-IO (h5bench_write, particle dumps of the VPIC plasma code)
        elif "vpic" in self.app:
            # h5bench's VPIC-IO kernel: every rank writes VPIC_PARTICLES particles
            # (8 float variables, 32 B each) per dump, VPIC_STEPS dumps with
            # VPIC_COMPUTE_S seconds of emulated compute between them. Stock
            # h5bench puts all dumps in one file; our patch (H5BENCH_FILE_PER_TS=1)
            # writes dump i to <outdir>/vpic.h5.<i>, one file per dump as VPIC does.
            self.run_dir = self.prepare_run_dir(
                f"{self.home}/h5bench/build", ["h5bench_write"]
            )
            outdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            self.app_call = f"env H5BENCH_FILE_PER_TS=1 {self.run_dir}/h5bench_write"
            self.app_flags = f"{self.run_dir}/vpic.cfg {outdir}/vpic.h5"
            self.prepare_vpic_input(
                os.getenv("VPIC_PARTICLES", "1M"),
                int(os.getenv("VPIC_STEPS", "20")),
                int(os.getenv("VPIC_COMPUTE_S", "15")),
            )
        #  ├─ MILC (lattice QCD, pure gauge su3_ora)
        elif "milc" in self.app:
            # MILC_SAVES parameter sets, each MILC_TRAJ trajectories and then a
            # save_parallel of the gauge field: every rank seeks and writes its
            # sites into one shared file <outdir>/cfg.<k> (+ a small .info).
            # 288 B per site in single precision: MILC_L=64 -> 4.8 GB a config.
            # MILC_NX overrides the x extent (96/124 nodes, see prepare_milc_input).
            self.run_dir = self.prepare_run_dir(
                f"{self.home}/milc_qcd/pure_gauge", ["su3_ora"]
            )
            outdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            self.app_call = f"{self.run_dir}/su3_ora"
            self.app_flags = f"{self.run_dir}/milc.in"
            self.prepare_milc_input(
                outdir,
                int(os.getenv("MILC_L", "64")),
                int(os.getenv("MILC_SAVES", "10")),
                int(os.getenv("MILC_TRAJ", "20")),
                int(os.getenv("MILC_NX", "0")),
                skip_last=os.getenv("SKIP_LAST_CKPT", "0") == "1",
            )
        #  ├─ NAS kernels without I/O (CG, MG): overhead controls
        elif "npb" in self.app:
            # No checkpoints at all, so FTIO sees nothing and GLASS must cost
            # nothing: any GLASS-vs-PFS gap here is pure harness overhead.
            binary = f"{os.getenv('NPB_KERNEL', 'mg')}.{os.getenv('NPB_CLASS', 'D')}.x"
            self.run_dir = self.prepare_run_dir(f"{self.home}/NPB3.4-MPI/bin", [binary])
            self.app_call = f"{self.run_dir}/{binary}"
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
            # Weak scaling: the lattice grows with the job so bytes-per-rank stay
            # constant (see weak_scale_lattice). The node count is taken from -n,
            # not from app_nodes, so all three modes size identically -- app_nodes
            # is only assigned later, during node allocation.
            # LAMMPS_ATOMS_PER_RANK: the full win vs pfs only holds in a 33-65N band
            # (44585615/44696207), not 41/81/121N -- weak-scaling communication
            # overhead and pfs contention both grow with node count and neither
            # dominates predictably. Per glass-rarely-beats-pfs, growing the
            # checkpoint until pfs is the bottleneck is what turned 168 -> 221 from
            # a loss into a win in the first place; probe whether a bigger
            # atoms/rank (default 169_344) makes the pfs win band-independent.
            atoms_per_rank = int(os.getenv("LAMMPS_ATOMS_PER_RANK", "169344"))
            n = self.weak_scale_lattice((self.nodes - 1) * self.procs_app, atoms_per_rank)
            # Restart writer count. LAMMPS_MP=1 turns on multi-file ("%") restarts
            # (in.ckpt handles the branch), LAMMPS_NF sets the writer-group count
            # -- default one group per app node. lessons.md sec.3: single-writer
            # (mp=0) restart is rank-0-only, so each per-rank record falls under
            # one 512 KB chunk at scale and lands on a single daemon; mp=1 is the
            # structural fix. Tracked as the second LAMMPS config variant.
            lammps_mp = os.getenv("LAMMPS_MP", "0")
            lammps_nf = os.getenv("LAMMPS_NF", str(self.nodes - 1))
            # LAMMPS_IRREGULAR=1: R4 rebuttal ("never evaluates sensitivity to
            # irregular workloads") -- swaps the fixed-period deck for
            # in.ckpt_irregular, which draws each checkpoint gap uniformly from
            # [lo, hi] instead of a constant 'every'. Separate deck, not a flag
            # on in.ckpt, because LAMMPS's `restart N` command only supports a
            # fixed integer period -- the irregular timing needs a loop of
            # run/write_restart pairs instead. LAMMPS_LO/LAMMPS_HI default to a
            # mean of 8 to match LAMMPS_EVERY's default, so total steps/volume
            # stay comparable to the periodic baseline and only the timing
            # differs. Built 2026-09-24, NOT YET smoke-tested on BSC -- do not
            # promote to a full sweep before a short single-node dry run confirms
            # the deck parses and the flush regex still matches its filenames
            # (same `ckpt.restart.*` convention as in.ckpt, so it should, but
            # unverified).
            if os.getenv("LAMMPS_IRREGULAR", "") not in ("", "0"):
                self.app_flags = self.resolve_app_flags(
                    f"-in {self.run_dir}/in.ckpt_irregular -v ckptdir {ckptdir} "
                    f"-v x {n} -v y {n} -v z {n} "
                    f"-v mp {lammps_mp} -v nf {lammps_nf} "
                    f"-v lo {os.getenv('LAMMPS_LO', '3')} "
                    f"-v hi {os.getenv('LAMMPS_HI', '13')} "
                    f"-v nb {os.getenv('LAMMPS_NB', '4')} "
                    f"-v seed {os.getenv('LAMMPS_SEED', '12345')} "
                    f"-v phases {os.getenv('LAMMPS_PHASES', '60')} -v tail 8",
                    ckptdir,
                )
            else:
                self.app_flags = self.resolve_app_flags(
                    f"-in {self.run_dir}/in.ckpt -v ckptdir {ckptdir} "
                    f"-v x {n} -v y {n} -v z {n} "
                    f"-v mp {lammps_mp} -v nf {lammps_nf} "
                    # 60 phases: nsteps = phases*every + tail = 488 = ~60 s at 8
                    # compute nodes. Raised 30 -> 60 (2026-09-09, sweep
                    # 45532561-69): at 30 the app finished in ~30-45 s and glass
                    # lost pfs by 0.72-0.88x across 8-33+1; doubling the run
                    # closed that (glass now wins pfs from 16+1 up, beats gekko
                    # at every N). Per-rank work is fixed by the weak scaling so
                    # wall time stays flat as nodes grow, but the restart does
                    # not: drop phases for very large node counts or the sweep
                    # will outrun stage-out (a past run hit 6.9 TB).
                    # LAMMPS_EVERY/LAMMPS_PHASES let a single submission probe
                    # further without moving the default.
                    # LAMMPS_NB: neighbor-rebuild cadence, an exact divisor of
                    # 'every' (in.ckpt). Defaults to 'every'; a long 'every'
                    # (120, job 48138435) needs a short nb or atoms move too far
                    # between rebuilds ("ERROR: Lost atoms").
                    f"-v every {os.getenv('LAMMPS_EVERY', '8')} "
                    f"-v nb {os.getenv('LAMMPS_NB', os.getenv('LAMMPS_EVERY', '8'))} "
                    f"-v phases {os.getenv('LAMMPS_PHASES', '60')} -v tail 8",
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
                # GLASS paper (paper/dlio/DLIO_20GB): pytorch framework + npz
                # format -> no h5py import, so no libhdf5.so.103 break. The old
                # default resnet50_v100_new_small is framework: tensorflow, and
                # keras eager-imports h5py at startup -> ImportError on every leg.
                # DLIO_20GB won 1.15-1.26x vs pfs in the paper; DLIO_40GB 1.31x.
                # workload = "resnet50_v100_new_small"  # tensorflow -> h5py break
                # workload = "llama_my_7b_zero3"        # zero_stage=3 metadata storm
                workload = "resnet50_my_a100_pytorch"
            # ensure surrounding spaces
            workload = f" workload={workload} "
        #  ├─ S3D-IO
        elif "s3d" in self.app:
            # $HOME-relative like every other app: the old hardcoded Mogon path
            # (/lustre/project/nhr-gekko/shared/...) does not exist on BSC, so
            # -a s3d could never run there even though the binary is installed.
            self.app_call = os.getenv("S3D_BIN", f"{self.home}/S3D-IO/s3d_io.x")
            # A big local subdomain (fixed_grid3d at low PROCS gives each rank a
            # much larger block than weak_scale_grid3d ever did -- e.g. 512 MB/rank
            # at PROCS=8/16+1/800^3) overflows the default 8 MB stack if S3D-IO
            # keeps that block on the stack rather than heap-allocating it, and the
            # GekkoFS interceptor's extra call-stack depth is what tips a marginal
            # allocation over: job 45408816 (PROCS=8, fixed 800^3, 16+1) SIGSEGV'd
            # inside pnetcdf_write on the glass/gekko (intercepted) legs only --
            # the pfs leg, same binary and buffer size, no interception, ran clean.
            # Same class of fix as WRF's ideal.exe (jitsettings.py pre_app_call).
            self.stack_unlimited = True
            # S3D-IO's own output-path CLI arg was a bare "." -- resolved against
            # jit's cwd (the job's $HOME/jit/<jobid>/ dir), which has a quota and
            # is not meant for parallel I/O (same class of bug prepare_run_dir was
            # built for -- see its docstring). Route to scratch instead.
            self.run_dir = self.prepare_run_dir(f"{self.home}/S3D-IO", files=[])
            if not self.app_flags:  # default value if app_flags is not set
                ranks = (self.nodes - 1) * self.procs_app
                # Two sizing modes:
                #  - S3D_GLOBAL_EDGE set  -> fixed_grid3d, the GLASS paper recipe
                #    (paper/s3d-io/serial_800 = 800): a ~62 GB aggregate checkpoint
                #    at every node count, so PFS's collective write is genuinely
                #    slow and GLASS's incremental flush has something to beat. Run
                #    this with PROCS=64 (the paper's -p 64) for the golden-goal
                #    comparison.
                #  - otherwise            -> weak_scale_grid3d via S3D_EDGE_PER_RANK
                #    (default 64 -> ~33.5 MB/rank): per-rank record fixed, aggregate
                #    grows with the job. A scaling study, not a GLASS win at small N.
                global_edge = os.getenv("S3D_GLOBAL_EDGE")
                if global_edge:
                    nx_g, ny_g, nz_g, npx, npy, npz = self.fixed_grid3d(
                        ranks, int(global_edge)
                    )
                else:
                    edge_per_rank = int(os.getenv("S3D_EDGE_PER_RANK", "64"))
                    nx_g, ny_g, nz_g, npx, npy, npz = self.weak_scale_grid3d(
                        ranks, edge_per_rank
                    )
                self.app_flags = f"{nx_g} {ny_g} {nz_g} {npx} {npy} {npz} 0 F ."
        #  ├─ WRF
        elif "wrf" in self.app:
            # em_b_wave_glass is our copy of the idealized baroclinic-wave case
            # (run_hours=48, restart_interval=360). WRF must NOT run inside the
            # mount: it reads each namelist group with a REWIND and that fails
            # through GekkoFS. So the inputs stay on the parallel FS and only
            # rst_outname points into the mount.
            #
            # wrfinput_d01 encodes the domain size, so it MUST be regenerated
            # with ideal.exe whenever e_we/e_sn/e_vert change in namelist.input.
            # Scaling the deck 4x (82x162 -> 328x648) on 2026-08-04 without
            # rerunning ideal.exe left wrf.exe aborting during init in all three
            # modes -- pfs included, which is what proved it was never a GekkoFS
            # problem.
            self.app_call = f"{self.home}/WRF/main/wrf.exe"
            # WRF drops rsl.out.<rank> + rsl.error.<rank> in its cwd: never $HOME.
            self.run_dir = self.prepare_run_dir(
                f"{self.home}/WRF/test/em_b_wave_glass",
                ["namelist.input", "wrfinput_d01", "ideal.exe", "input_jet"],
            )
            self.app_flags = ""
            self.point_wrf_restarts_at(
                self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            )
            # Weak-scale the domain: a fixed e_we/e_sn (unchanged since
            # 2026-08-04) means each rank's restart-file slice shrinks as node
            # count grows -- the same shrinking-per-rank-record bug already
            # fixed for LAMMPS/WarpX/S3D-IO (weak_scale_lattice/_cells/
            # _grid3d). See weak_scale_wrf_grid for the dx/time_step coupling.
            # Opt-in via WRF_WEAK_SCALE_GRID=1 until the whole node-count sweep
            # has been redone under it (see [[one-scaling-rule-per-app]]).
            if os.getenv("WRF_WEAK_SCALE_GRID") == "1":
                ranks = (self.nodes - 1) * self.procs_app
                e_we, e_sn, dx, dy, dt, nproc_x, nproc_y = self.weak_scale_wrf_grid(ranks)
                # nproc_x/nproc_y must NOT be in the namelist yet when ideal.exe
                # runs below: ideal.exe is invoked directly (no srun -n), i.e. as
                # a single serial task, so it needs WRF's default auto
                # decomposition (-1/-1, which resolves to 1x1 for a 1-task run).
                # Writing the explicit ranks-sized nproc_x*nproc_y here made
                # ideal.exe abort ("check comm_start, nest_pes_x, nest_pes_y ...")
                # because 12*8=96 != the 1 actual MPI task it was launched with
                # (jobs 45787185/45787186, 2026-09-12). The explicit decomposition
                # is still needed for wrf.exe's real parallel launch afterward
                # (see weak_scale_wrf_grid for why auto breaks at >64 ranks) --
                # inserted via sed in pre_app_call below, after ideal.exe exits,
                # so it only reaches the namelist wrf.exe reads.
                self.set_wrf_domain(e_we, e_sn, dx, dy, dt)
                # Regenerate wrfinput_d01 for the new domain (ideal.exe reads
                # input_jet, hence its place in the copy list above).
                # ideal.exe SIGABRTs (returncode -11) on the generated namelist
                # above ~232x458 and teardown wipes run_dir before it can be
                # inspected -- copy the namelist + any rsl.error.* out to $HOME
                # first so the crash can be diagnosed offline.
                keep = f"{self.home}/wrf_weakgrid_{self.job_id}"
                self.pre_app_call = (
                    f"cd {self.run_dir} && mkdir -p {keep} && "
                    f"cp namelist.input {keep}/ && ulimit -s unlimited && "
                    f"(./ideal.exe; rc=$?; cp -f rsl.* {keep}/ 2>/dev/null; "
                    f"[ $rc -eq 0 ] && "
                    f'sed -i "/e_sn/a nproc_x                             = {nproc_x}," namelist.input && '
                    f'sed -i "/nproc_x/a nproc_y                             = {nproc_y}," namelist.input; '
                    f"exit $rc)"
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
            # n_cell 320 was tried and reverted: 9N went compute-bound (glass last)
            # and 65N died in the Mercury metadata storm.
            #
            # Files per MultiFab (AMReX_Amr.cpp -> VisMF::SetNOutFiles), default 64.
            # Lowering it is the direct lever on the metadata storm that kills the
            # daemon at 65 nodes, but fewer/larger files also suit Lustre, which
            # weakens gekko < pfs. Leave unset to keep AMReX's default.
            nfiles = os.getenv("CASTRO_CHECKPOINT_NFILES", "")
            nfiles_flag = f"amr.checkpoint_nfiles={nfiles} " if nfiles else ""
            # CASTRO_ASYNC_OUT=1: AMReX writes each checkpoint straight into its
            # final chk dir instead of <chk>.temp + rename. GekkoFS loses a dir's
            # children on rename, so without this every flush finds nothing to
            # copy (11-01 runs: "lost its children to a rename"). nfiles >= ranks
            # keeps one file per rank, so no MPI_THREAD_MULTIPLE is needed.
            async_flag = (
                "amrex.async_out=1 amrex.async_out_nfiles=1000000 "
                if os.getenv("CASTRO_ASYNC_OUT") == "1"
                else ""
            )
            self.app_flags = self.resolve_app_flags(
                f"inputs.3d.sph max_step=1440 amr.check_int=24 amr.plot_int=-1 "
                f"amr.max_level=0 castro.fixed_dt=4e-6 castro.init_shrink=1.0 "
                f"{nfiles_flag}{async_flag}"
                f"amr.check_file={ckptdir}/sedov_3d_sph_chk amr.n_cell = 160 160 160",
                ckptdir,
            )
        #  ├─ WarpX (AMReX)
        elif "warpx" in self.app:
            # AMReX checkpoint directories (chk<step>/...) into the mount,
            # write-only. Size = n_cell (weak-scaled), period = intervals.
            # Force FUSE: the syscall interceptor's flat namespace can't walk
            # AMReX's nested chk<step>/Level_0/ tree with `cp -r` (job 45077746
            # staged 0 bytes, hung to the wall) and the gekko leg died on a UCX
            # ibv_reg_mr failure under the seccomp hooks. FUSE's real POSIX
            # namespace fixes both (45153231 / 45244939). Set WARPX_NO_FUSE=1 to
            # opt back out for a syscall comparison.
            if os.getenv("WARPX_NO_FUSE", "") in ("", "0"):
                self.fuse = True
            self.app_call = f"{self.home}/WarpX/build/bin/warpx.3d"
            self.run_dir = self.prepare_run_dir(f"{self.home}/WarpX/glass", ["inputs"])
            ckptdir = self.gkfs_mntdir if not self.exclude_daemon else self.run_dir
            # Weak-scale n_cell like LAMMPS's lattice (weak_scale_lattice): a fixed
            # n_cell=320 is really a different, shrinking-per-rank workload at every
            # node count, not the same run at a bigger scale -- 9-33N lost to pfs
            # (0.78-0.92x) while 41-121N won, which tracks bytes/rank shrinking as
            # nodes grow, not a property of the app itself. 64000 cells/rank is the
            # 65N/-p8 config (n=320, 512 ranks) that's the one clean, single-run-
            # confirmed full win (44696014) -- anchor every node count to it instead
            # of hand-picking a size per node count.
            cells_per_rank = int(os.getenv("WARPX_CELLS_PER_RANK", "64000"))
            ranks = (self.nodes - 1) * self.procs_app
            n_cell = self.weak_scale_cells(ranks, cells_per_rank)
            # WARPX_INTERVALS=800 -> ~5 checkpoints over max_step=3200, ~24 s
            # apart at 9N. The old value (50 -> a checkpoint every ~1.5 s) was
            # calibrated while the checkpoint was a silent no-op (chk never
            # registered); once it actually wrote, the flush fell behind and
            # the run aborted (job 45134159). Checkpoint period must exceed the
            # flush drain time -- same rule as DLIO_COMPUTE_TIME.
            #
            # A fixed 800 stopped being enough at 124+1 (2026-09-14, job
            # 45852469): n_cell weak-scales (bigger checkpoint at higher N),
            # so copy time grows with node count while the wall-clock gap
            # between checkpoints (intervals steps x time/step) did not. At
            # 96+1 (job 45852468, 768 ranks) the last checkpoint's copy
            # (25.9s) still just barely finished inside the ~24.5s gap
            # (app_time/4) -- the last point where intervals=800 holds. At
            # 124+1 (992 ranks) copy time grew to 39.3s against a ~26.9s gap,
            # missing it -- the last 2 of 5 checkpoints fell through into the
            # end-of-run catch-up drain instead of an incremental flush
            # (stage_out jumped 0.47s -> 45.7s, and glass lost to both gekko
            # and pfs for the first time in this app's sweep). Scale the gap
            # for N beyond that point only, so the already-validated 8+1-96+1
            # range (intervals=800 there) is untouched: linear in ranks (copy
            # time should track total checkpoint bytes, which weak-scale
            # linearly with ranks by construction -- n_cell is the cube root),
            # plus a 1.3x safety margin because the observed 96+1->124+1 copy
            # time growth (1.52x) already ran ahead of the raw rank ratio
            # (1.29x), so a same-ratio scale-up alone would likely still miss.
            # NOT re-verified below 124+1 under this formula -- promoting the
            # weak-grid-style "one setting across the sweep" rule to table 1
            # needs a fresh check at every node count, not just the one that
            # was broken.
            base_ranks_for_intervals = (
                768  # ranks at 96+1, the last point intervals=800 held
            )
            safety_margin = 1.3
            default_intervals = (
                800
                if ranks <= base_ranks_for_intervals
                else round(800 * safety_margin * ranks / base_ranks_for_intervals)
            )
            intervals = int(os.getenv("WARPX_INTERVALS", str(default_intervals)))
            max_step = int(os.getenv("WARPX_MAX_STEP", "3200"))
            # Declare the checkpoint diagnostic in the deck (command-line
            # quoting mangles it) and drop the stock diag1 plotfile so the
            # checkpoint is the only I/O.
            self.set_warpx_checkpoint(ckptdir, intervals)
            # Gate A lever, untested until now (2026-09-25): the opposite
            # direction from the old "raise checkpoint_nfiles to split into
            # many units" plan (that was Gate-B reasoning -- more/smaller
            # flush units -- and conflicts with the already-tested "WarpX 32
            # checkpoints vs 5 -> worse" Gate-A result; also the flush regex
            # matches the whole chk<step> directory as one unit regardless of
            # internal file count, so it wouldn't even change flush
            # granularity). Gate A needs genuine Lustre write CONTENTION, and
            # WRF's single-shared-file restart (confirmed via flush.log) still
            # fails Gate A despite looking collective -- likely because it
            # funnels through a gather-then-serial-write pattern, not true
            # concurrent multi-rank MPI-IO. AMReX's VisMF, when nfiles is LOW
            # relative to rank count, funnels many ranks' data through few
            # actual files via genuine collective MPI-IO coordination (unlike
            # WRF's classic-netCDF gather). Setting nfiles=1 forces maximum
            # collective contention -- the closest analog to QMC's winning
            # topology this session has found for WarpX. WARPX_CHECKPOINT_NFILES
            # unset = AMReX's own default (no behavior change).
            nfiles = os.getenv("WARPX_CHECKPOINT_NFILES", "")
            nfiles_flag = f"amr.checkpoint_nfiles={nfiles} " if nfiles else ""
            self.app_flags = self.resolve_app_flags(
                f"inputs max_step={max_step} "
                f"{nfiles_flag}"
                f"chk.file_prefix={ckptdir}/chk amr.n_cell = "
                f"{n_cell} {n_cell} {n_cell}",
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
            # Run QMC at PROCS=8 -- the config behind every table-1 QMC win.
            # "-p 56" was documented as "current best" but never validated, and it
            # broke two ways: QMCPACK auto-threads from --cpus-per-task (56x56 =
            # 3136 threads/node, ~28x oversubscribed), and 56 ranks/node x 8192
            # walkers/rank put ~68 GB of walker buffers on each node -> the 32+1
            # run stalled 90 min in walker setup (job 45335439, same class as the
            # 16384-walker hang in 44475461). cpus_per_task_app=2 keeps 2 OMP
            # threads/rank (1 starves the GekkoFS RPC-progress thread).
            self.cpus_per_task_app = 2
            # glass.xml's <walkers> is PER RANK. Node memory for the walker
            # buffers is per_rank * omp_threads * 73808 B * ranks_per_node, so at
            # PROCS=8 / 2 threads the safe ceiling is ~32k walkers/rank (enforced
            # inside set_qmc_walkers via per_rank_cap). Pin the aggregate instead
            # (QMC_TOTAL_WALKERS): the per-section config.h5 checkpoint -- what
            # GLASS's incremental flush has to beat gekko's end drain on -- stays
            # a fixed large size at every node count.
            #
            # 2026-09-25/28: promoted default 2M -> 8M after a full 8-point sweep
            # at each candidate. 8M: 7 wins / 1 loss (124+1 loses to gekko by a
            # hair, still crushes pfs 7.1x). 16M: 5 wins / 3 losses -- tested and
            # NOT adopted, worse win-rate. 8M costs ~20-25% more wall-clock than
            # the old 2M-era canonical row at every N from 8+1 up (e.g. 124+1:
            # 386.3s vs 324.9s) but wins more consistently and by a much wider
            # pfs margin -- Ahmad confirmed the tradeoff is worth it. Old default
            # (2M) still available via QMC_TOTAL_WALKERS=2000000 if ever needed.
            ranks = (self.nodes - 1) * self.procs_app
            self.set_qmc_walkers(int(os.getenv("QMC_TOTAL_WALKERS", "8000000")), ranks)
            # QMC_STEPS: compute per <qmc> section, i.e. the gap between two
            # config.h5 checkpoints. At the deck's steps=1 the gap is only
            # 0.44-0.77x the flush time (2026-10-27 workload-fit table), so FTIO
            # has no idle window to flush in. More steps lengthen the gap
            # without changing the checkpoint size. Unset keeps the deck as is.
            if os.getenv("QMC_STEPS"):
                self.set_qmc_steps(int(os.environ["QMC_STEPS"]))
            # QMC_SECTIONS: number of <qmc> sections = resident config.h5 files.
            # At 30 the gekko end drain is one 32-thread wave (11-22 s at every
            # N, 2026-10-28), too little for GLASS to hide. Unset keeps the deck.
            if os.getenv("QMC_SECTIONS"):
                self.set_qmc_sections(int(os.environ["QMC_SECTIONS"]))
            # SKIP_LAST_CKPT=1: the last section writes no config.h5. The app ends
            # right after it, so no compute phase is left to flush it in.
            if os.getenv("SKIP_LAST_CKPT", "0") == "1":
                self.skip_last_qmc_checkpoint()
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
            # DLIO derives checkpoint_only = (not do_train) and do_checkpoint
            # (utils/config.py). In that mode run() skips training entirely and
            # just writes num_checkpoints_write checkpoints spaced by
            # time_between_checkpoints -- the compute->checkpoint square wave we
            # want, with no dataset at all. Forcing train=True instead drags the
            # dataset back in and dies on "Max steps per epoch: 0" as soon as
            # num_files_train < ranks (llama_7b_zero3 ships 8, so anything past
            # 8 ranks). Set DLIO_CHECKPOINT_ONLY=1 for such workloads.
            ckpt_only = os.getenv("DLIO_CHECKPOINT_ONLY", "") not in ("", "0")
            train = "False" if ckpt_only else "True"
            # _checkpoint() raises if fewer checkpoints exist than it wants to
            # read back, and driver apps must only ever write.
            no_read = "++workload.checkpoint.num_checkpoints_read=0 " if ckpt_only else ""
            # Checkpoint-only mode never auto-derives num_checkpoints_write:
            # config.py defaults it to -1 and _checkpoint_write() runs only when
            # it is > 0 (main.py:262), so the compute->checkpoint square wave
            # happens only if we pass it. The llama_* workloads ship 10;
            # resnet50_my_a100_pytorch ships nothing, so job 45534106 wrote ~3
            # checkpoints and FTIO locked no period (P(periodic)=0, glass silently
            # ran as gekko-mode). DLIO_NUM_CHECKPOINTS sets it -- default 20 in
            # ckpt-only mode gives FTIO plenty of cycles to lock on; pair it with
            # DLIO_COMPUTE_TIME (below, defaulted to 50 s here) so each checkpoint
            # fully drains before the next lands.
            n_ckpt = os.getenv("DLIO_NUM_CHECKPOINTS", "20" if ckpt_only else "")
            n_ckpt_flag = (
                f"++workload.checkpoint.num_checkpoints_write={n_ckpt} "
                if n_ckpt not in ("", "0")
                else ""
            )
            # Length of the compute phase between checkpoints, in seconds --
            # _checkpoint_write() passes it straight to framework.compute()
            # (main.py). GLASS can only hide stage-out behind compute if the gap
            # outlasts the drain, and llama_7b_zero3 ships 5 s while one
            # checkpoint needs ~76 s to reach GPFS (65 nodes, job 44301928). At
            # that ratio the flush never finishes before the next checkpoint
            # lands, so it degenerates into a post-app stage-out and the flush
            # only steals daemon bandwidth from the app. Leave unset to keep the
            # workload's own value.
            compute_time = os.getenv("DLIO_COMPUTE_TIME", "50" if ckpt_only else "")
            gap = (
                f"++workload.checkpoint.time_between_checkpoints={compute_time} "
                if compute_time
                else ""
            )
            # llama_7b_zero3's checkpoint is a fixed 4096 hidden_size regardless of
            # -n. Tried scaling hidden_size ~ ranks^0.5 to hold per-rank shard
            # bytes constant (2026-08-19/20) -- it made every node count worse
            # (9N started losing to gekko, 33N's gekko leg crashed on daemon
            # capacity again, 65N collapsed to 0.24x pfs). Reverted to the fixed
            # value; DLIO_SCALE_HIDDEN=1 re-enables the scaled formula for
            # further experiments, off by default.
            ranks = max(1, (self.nodes - 1) * self.procs_app)
            if os.getenv("DLIO_SCALE_HIDDEN", "") not in ("", "0"):
                hidden = max(512, round(4096 * (ranks / 512) ** 0.5 / 128) * 128)
            else:
                hidden = 4096
            ffn_hidden = round(hidden * 11008 / 4096)
            scale = (
                f"++workload.model.transformer.hidden_size={hidden} "
                f"++workload.model.transformer.ffn_hidden_size={ffn_hidden} "
            )
            # Gate A lever, untested until now: read base_checkpointing.py
            # directly (2026-09-24). At zero_stage=0, save_checkpoint()'s gate
            # (`rank_to_checkpoint == my_rank`) makes rank_to_checkpoint 0 for
            # every non-model-parallel rank -- so with model_parallelism=1
            # (our default), ONLY RANK 0 EVER WRITES, model state AND
            # optimizer state both. A true single-writer/N->1 checkpoint, not
            # a smaller shard count -- structurally different from
            # zero_stage=1 (optimizer state sharded across data-parallel
            # ranks, confirmed via flush.log job 43408430:
            # model_states-N.pb-1_temp per rank) and zero_stage=3 (already
            # tested, no effect on Gate A, job 46435305). Neither prior test
            # ever tried genuine single-writer topology. DLIO_ZERO_STAGE
            # overrides it; unset keeps the workload yaml's own value (no
            # behavior change by default).
            zero_stage_override = os.getenv("DLIO_ZERO_STAGE", "")
            zero_stage_flag = (
                f"++workload.parallelism.zero_stage={zero_stage_override} "
                if zero_stage_override
                else ""
            )
            # resnet50_v100_new_small.yaml's num_files_train=25 is the same
            # flat-constant-regardless-of-node-count problem the hidden_size
            # attempt above was for -- except this one's never been tried at
            # all (hidden_size only ever applied to the unused llama_7b_zero3
            # workload). Worse: this repo's own comment on DLIO_CHECKPOINT_ONLY
            # documents train=True with num_files_train < ranks hitting "Max
            # steps per epoch: 0" -- already true past 6 nodes at
            # PROCS_DLIO=4. files_per_rank=1 is a conservative floor (every
            # node count gets at least one file per rank), not a calibrated
            # value -- DLIO_SCALE_FILES=1 opts in, off by default so nothing
            # changes silently. Untested against the sweep.
            if os.getenv("DLIO_SCALE_FILES", "") not in ("", "0"):
                num_files = self.weak_scale_files(ranks, files_per_rank=1)
                files_scale = f"++workload.dataset.num_files_train={num_files} "
            else:
                files_scale = ""
            if self.exclude_daemon:
                self.app_flags = (
                    f"{workload} "
                    f"++workload.workflow.generate_data=False "
                    f"++workload.workflow.train={train} "
                    f"++workload.workflow.checkpoint=True "
                    f"{no_read}"
                    f"{n_ckpt_flag}"
                    f"{gap}"
                    f"{scale}"
                    f"{zero_stage_flag}"
                    f"{files_scale}"
                    f"++workload.dataset.data_folder={self.run_dir}/data "
                    f"++workload.checkpoint.checkpoint_folder={self.run_dir}/checkpoints "
                    f"++workload.output.output_folder={self.run_dir}/hydra_log "
                )
                # self.pre_app_call = f"mpirun -np 8 dlio_benchmark {self.app_flags} ++workload.workflow.generate_data=True ++workload.workflow.train=False"
                # self.pre_app_call = f"mpirun -np $APP_NODES dlio_benchmark {self.app_flags} ++workload.workflow.generate_data=True ++workload.workflow.train=False"
                self.pre_app_call = (
                    ""
                    if ckpt_only  # nothing to generate, there is no dataset
                    else (
                        f"mpirun -np $APP_PROCS_X_NODES dlio_benchmark "
                        f"{workload} "
                        f"++workload.workflow.generate_data=True "
                        f"++workload.workflow.train=False "
                        f"++workload.workflow.checkpoint=False "
                        f"++workload.dataset.data_folder={self.run_dir}/data "
                        f"++workload.checkpoint.checkpoint_folder={self.run_dir}/checkpoints "
                        f"++workload.output.output_folder={self.run_dir}/hydra_log "
                    )
                )
                self.post_app_call = ""
            else:
                # self.run_dir = self.gkfs_mntdir #? don't enable this flag, as the executing node doesn't have this folder
                self.app_flags = (
                    # f"++workload.workflow.generate_data=True ++workload.workflow.train=True ++workload.workflow.checkpoint=True "
                    f"{workload} "
                    f"++workload.workflow.generate_data=False "
                    f"++workload.workflow.train={train} "
                    f"++workload.workflow.checkpoint=True "
                    f"{no_read}"
                    f"{n_ckpt_flag}"
                    f"{gap}"
                    f"{scale}"
                    f"{zero_stage_flag}"
                    f"{files_scale}"
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
                ]
                if not ckpt_only:  # nothing to generate, there is no dataset
                    self.pre_app_call.append(
                        f"mpirun -np $APP_PROCS_X_NODES dlio_benchmark "
                        f"{workload} "
                        f"++workload.workflow.generate_data=True "
                        f"++workload.workflow.train=False "
                        f"++workload.workflow.checkpoint=False "
                        f"++workload.dataset.data_folder={dlio_dir}/data "
                        f"++workload.checkpoint.checkpoint_folder={dlio_dir}/checkpoints "
                        f"++workload.output.output_folder={dlio_dir}/hydra_log "
                    )
                self.post_app_call = ""
        # ├─ Nek5000
        elif "nek" in self.app:
            # Nek reads <case>.re2/.par from, and writes <case>0.f<N> to, the
            # directory on SESSION.NAME's second line (its trailing / matters).
            case = getattr(self, "nek_case", "turbPipe")
            if self.exclude_daemon:
                self.pre_app_call = f"echo -e '{case}\\n{self.run_dir}/input/' > {self.run_dir}/SESSION.NAME"
                self.post_app_call = f"rm {self.run_dir}/input/*.f* || echo true"
            else:
                self.pre_app_call = f"echo -e '{case}\\n{self.gkfs_mntdir}/' > {self.run_dir}/SESSION.NAME"
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
            self.post_app_call = ""
            if not self.exclude_daemon:
                self.pre_app_call = ""
                self.app_flags = self.app_flags.replace("test_run", f"{self.gkfs_mntdir}")
            else:
                # prepare_run_dir only copies the binary; the output subdir
                # (test_run/mpi, relative to run_dir) needs creating on scratch.
                self.pre_app_call = f"mkdir -p {self.run_dir}/test_run/mpi"
        #  ├─ E3SM-IO
        elif "e3sm" in self.app:
            self.post_app_call = ""
            if not self.exclude_daemon:
                self.pre_app_call = ""
                self.app_flags = self.app_flags.replace("test_run", f"{self.gkfs_mntdir}")
            else:
                self.pre_app_call = f"mkdir -p {self.run_dir}/test_run"
        # ├─ wrf
        elif "wrf" in self.app:
            # Deliberately empty by default. The old body ran WRF *inside* the
            # mount, which cannot work: WRF reads each namelist group with a
            # REWIND and that fails through GekkoFS ("ERROR while reading
            # namelist diags" -> FATAL -> MPI_ABORT), while the identical file
            # parses fine on a real FS. It also built the call from the
            # `cdf`/`cpf` cluster aliases and $HOME paths, neither of which
            # exists locally.
            # Restarts go into the mount via point_wrf_restarts_at() instead.
            # WRF_WEAK_SCALE_GRID=1 sets pre_app_call earlier (ideal.exe regen)
            # -- don't clobber it here.
            if not self.pre_app_call:
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

    @staticmethod
    def weak_scale_lattice(ranks: int, atoms_per_rank: int = 169_344) -> int:
        """fcc lattice edge that gives each rank ~`atoms_per_rank` atoms.

        LAMMPS writes its restart from rank 0 only, one record per rank. GekkoFS
        parallelizes a write across as many daemons as that single write spans
        512 KB chunks, so what matters is bytes *per rank*, not total file size.
        With a fixed lattice that shrinks as nodes grow -- at 168^3 over 3584
        ranks each record is 466 KB, under one chunk, so every write hits one
        daemon and GekkoFS loses to the PFS. Holding atoms/rank constant keeps
        each record many chunks wide at any scale.

        Args:
            ranks (int): Total application ranks.
            atoms_per_rank (int): Atoms each rank should own.

        Returns:
            int: Lattice edge n, so the deck has 4*n^3 atoms.
        """
        return max(1, round(((atoms_per_rank * max(1, ranks)) / 4) ** (1 / 3)))

    @staticmethod
    def weak_scale_files(ranks: int, files_per_rank: int = 1) -> int:
        """num_files_train that gives each DLIO rank at least `files_per_rank` files.

        resnet50_v100_new_small.yaml ships num_files_train=25, a flat constant
        never adjusted for node count -- the exact anti-pattern already found
        and fixed for WarpX/LAMMPS (see [[one-scaling-rule-per-app]]). Worse
        here: DLIO_CHECKPOINT_ONLY's own comment documents that train=True
        with num_files_train < ranks hits "Max steps per epoch: 0" and does
        no real training work -- at PROCS_DLIO=4, that's already true past
        6 nodes ((n-1)*4 > 25). Untested whether this explains the
        daemon-capacity-theory failure (more nodes made 81N worse than 65N,
        the opposite of the theory) or is unrelated; files_per_rank=1 is the
        conservative floor that keeps every node count valid, not a
        calibrated "best" value.

        Args:
            ranks (int): Total application ranks.
            files_per_rank (int): Minimum files each rank should have to read.

        Returns:
            int: num_files_train.
        """
        return max(1, files_per_rank * max(1, ranks))

    @staticmethod
    def _factor3(ranks: int) -> tuple[int, int, int]:
        """Balanced 3-way factorization of `ranks` -> (npx, npy, npz), npx>=npy>=npz.

        Peel the largest divisor <= cube root first, then the largest divisor of
        what's left <= sqrt of it. Shared by weak_scale_grid3d and fixed_grid3d.
        """
        ranks = max(1, ranks)
        npz = max(d for d in range(1, ranks + 1) if ranks % d == 0 and d * d * d <= ranks)
        rem = ranks // npz
        npy = max(d for d in range(1, rem + 1) if rem % d == 0 and d * d <= rem)
        npx = rem // npy
        return npx, npy, npz

    @staticmethod
    def weak_scale_grid3d(
        ranks: int, edge_per_rank: int
    ) -> tuple[int, int, int, int, int, int]:
        """S3D-IO process grid + global domain that keeps each rank's subdomain fixed.

        Weak scaling: nx_g/ny_g/nz_g = npx/npy/npz * edge_per_rank, so each rank
        always writes the same subdomain volume regardless of node count and the
        aggregate checkpoint grows with the job. Fine for scaling studies, but at
        small node counts the aggregate is only a few GB and PFS writes it fast
        enough that GLASS's incremental flush has nothing to beat -- for the
        golden-goal comparison use fixed_grid3d (the GLASS paper recipe) instead,
        which pins a large aggregate checkpoint at every node count.

        Args:
            ranks (int): Total application ranks (must equal npx*npy*npz exactly --
                S3D-IO has no notion of idle ranks).
            edge_per_rank (int): Per-rank subdomain edge length.

        Returns:
            tuple[int, int, int, int, int, int]: (nx_g, ny_g, nz_g, npx, npy, npz).
        """
        npx, npy, npz = JitSettings._factor3(ranks)
        return (
            npx * edge_per_rank,
            npy * edge_per_rank,
            npz * edge_per_rank,
            npx,
            npy,
            npz,
        )

    @staticmethod
    def fixed_grid3d(ranks: int, global_edge: int) -> tuple[int, int, int, int, int, int]:
        """S3D-IO process grid + a fixed global domain (the GLASS paper recipe).

        paper/s3d-io/serial_800 held nx_g=ny_g=nz_g=800 while npx*npy*npz grew with
        the job -- the aggregate checkpoint stays ~62 GB at every node count, so
        PFS's collective write is genuinely slow and GLASS beats both PFS and gekko
        at every point 3-33N in the paper. Per-rank work shrinks with scale (strong
        scaling of the grid), which is the price of a fixed large checkpoint and
        exactly what makes it a GLASS benchmark rather than a scaling study.

        Each global edge is rounded down to an exact multiple of its process-grid
        dimension (S3D-IO block-partitions each dimension; a non-exact split is
        undefined behavior), so the realized grid is <= global_edge on each axis.

        Args:
            ranks (int): Total application ranks (== npx*npy*npz exactly).
            global_edge (int): Target global domain edge on each axis.

        Returns:
            tuple[int, int, int, int, int, int]: (nx_g, ny_g, nz_g, npx, npy, npz).
        """
        npx, npy, npz = JitSettings._factor3(ranks)
        return (
            (global_edge // npx) * npx,
            (global_edge // npy) * npy,
            (global_edge // npz) * npz,
            npx,
            npy,
            npz,
        )

    @staticmethod
    def weak_scale_cells(ranks: int, cells_per_rank: int, blocking: int = 32) -> int:
        """AMReX domain edge that gives each rank ~`cells_per_rank` cells.

        Same rationale as `weak_scale_lattice`: a checkpoint's bytes-per-rank
        is what determines whether a GekkoFS write spans enough 512 KB chunks
        to parallelize, so a fixed `amr.n_cell` that doesn't grow with the
        node count is really a different, shrinking-per-rank workload at every
        scale, not the same app run bigger. Rounds to a multiple of `blocking`
        (AMReX's `amr.blocking_factor`) so the domain always decomposes.

        Args:
            ranks (int): Total application ranks.
            cells_per_rank (int): Domain cells each rank should own.
            blocking (int): AMReX blocking factor the edge must be a multiple of.

        Returns:
            int: Domain edge n (n^3 total cells), a multiple of `blocking`.
        """
        raw = (cells_per_rank * max(1, ranks)) ** (1 / 3)
        return max(blocking, round(raw / blocking) * blocking)

    @staticmethod
    def weak_scale_wrf_grid(
        ranks: int,
        base_ranks: int = 8,
        base_e_we: int = 164,
        base_e_sn: int = 324,
        base_dx: float = 25000.0,
        base_dt: float = 150.0,
    ) -> tuple[int, int, float, float, float]:
        """WRF domain size that keeps each rank's restart-file slice constant.

        em_b_wave_glass's e_we/e_sn/dx have been fixed since 2026-08-04 across
        every node count -- the same shrinking-per-rank-record bug already fixed
        for LAMMPS/WarpX/S3D-IO (weak_scale_lattice/_cells/_grid3d): WRF
        decomposes e_we x e_sn across ranks (PROCS=1, so ranks == app nodes), so
        a fixed grid means each rank's restart slice shrinks as nodes grow.

        Scales e_we/e_sn up by sqrt(ranks/base_ranks) to hold grid-points-per-rank
        constant, and dx/dy down by the same factor so the physical domain
        extent (e_we*dx, e_sn*dy) stays exactly what it was at base_ranks --
        growing e_we/e_sn without shrinking dx let the idealized channel's
        y-extent exceed a physical bound and segfaulted ideal.exe (2026-08-04,
        82x162->328x648 test, see [[wrf-scale-dx-with-grid]]). time_step scales
        with dx to hold the CFL ratio (dt/dx) constant -- a finer dx needs a
        proportionally smaller step or the integration blows up.

        Note: because dt shrinks with dx, the step count needed to cover a fixed
        run_hours grows as sqrt(ranks) too, so larger node counts take
        proportionally longer in wall-clock app time under this formula on top
        of running more ranks -- a real cost of weak-scaling this deck, not a
        bug to chase out.

        e_sn is capped at MAX_E_SN (2026-09-11): WRF's stock em_b_wave idealized
        case hardcodes a 64x80 jet-sounding lookup table (module_initialize_ideal.F,
        `nz_jet=64, ny_jet=80`) and indexes its 2nd dimension directly by the
        model's south-north grid index with no bounds check. Any e_sn > 80 reads
        past the table; small overruns silently return garbage (wrong physics,
        no crash), large ones (e.g. e_sn=648, the 32+1 point) walk far enough to
        hit unmapped memory and SIGSEGV in calc_jet_sounding. Root-caused via gdb
        (job 45731615) -- confirmed NOT a GekkoFS issue, ideal.exe never touches
        the mount. Fixing the table itself would mean patching third-party WRF
        Fortran source, which repo policy reserves for an explicit exception;
        capping e_sn here instead keeps every WRF weak-grid run inside the
        table's valid range. Since base_e_sn (324) already exceeds the cap, e_sn
        is now effectively constant at MAX_E_SN across the whole sweep -- only
        e_we still weak-scales. Historical data taken before this cap (any run
        with e_sn > 80, which is every prior weak-grid point including the
        base/16+1 "clean" ones) used an uncapped domain and should be treated as
        a superseded config, not silently compared against post-cap runs.

        Args:
            ranks (int): Total application ranks (WRF runs PROCS=1, so nodes).
            base_ranks (int): Rank count the base_* values were calibrated at.
            base_e_we (int): e_we at base_ranks.
            base_e_sn (int): e_sn at base_ranks (pre-cap history; see above).
            base_dx (float): dx=dy (m) at base_ranks.
            base_dt (float): time_step (s) at base_ranks.

        Also returns an explicit nproc_x/nproc_y MPI decomposition (2026-09-12):
        with e_sn now capped at a constant 80 (see above) while e_we keeps
        growing, WRF's own auto-decomposition (nproc_x=nproc_y=-1) starts
        assigning more than 8 processes along the south-north axis once total
        ranks exceeds ~64 -- e.g. a 96-rank job picks something like an 8x12
        split, giving each rank only 80/12=6.7 south-north grid points. That's
        below WRF's minimum per-patch size and wrf.exe is killed within ~2s of
        starting (job 45780469, 96+1; 45780470, 124+1, both weak-grid,
        2026-09-12 -- confirmed NOT a timeout, "real 1.7s" before the SLURM
        step gets signal-killed, no rsl.error survives teardown to quote
        directly, but the timing/pattern match a decomposition-triggered
        MPI_Abort exactly, and this only appears once ranks exceeds the point
        where auto-decomposition would need >8 south-north processes for a
        fixed 80-wide domain). Fix: pick nproc_y as the largest divisor of
        ranks that keeps e_sn/nproc_y >= MIN_PATCH, and nproc_x = ranks /
        nproc_y -- e.g. 96 ranks -> nproc_y=8 (patch 10), nproc_x=12 (patch
        47); 124 ranks -> nproc_y=4 (patch 20), nproc_x=31 (patch 21). Falls
        back to nproc_y=1 (all-x decomposition) if no divisor fits, which is
        always safe since e_we scales with ranks and e_sn does not.

        Returns:
            tuple[int, int, float, float, float, int, int]: (e_we, e_sn, dx,
                dy, time_step, nproc_x, nproc_y).
        """
        MAX_E_SN = 80  # WRF em_b_wave jet-sounding table width (ny_jet), hard cap
        MIN_PATCH = 10  # WRF's minimum MPI-patch size per horizontal dimension
        eff_ranks = max(1, ranks)
        factor = (eff_ranks / base_ranks) ** 0.5
        e_we = max(base_e_we, round(base_e_we * factor))
        e_sn = min(MAX_E_SN, max(base_e_sn, round(base_e_sn * factor)))
        dx = base_dx / factor
        dt = base_dt / factor
        max_nproc_y = max(1, e_sn // MIN_PATCH)
        nproc_y = 1
        for d in range(min(max_nproc_y, eff_ranks), 0, -1):
            if eff_ranks % d == 0:
                nproc_y = d
                break
        nproc_x = eff_ranks // nproc_y
        return e_we, e_sn, dx, dx, dt, nproc_x, nproc_y

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

    def set_wrf_domain(
        self,
        e_we: int,
        e_sn: int,
        dx: float,
        dy: float,
        time_step: float,
        nproc_x: int | None = None,
        nproc_y: int | None = None,
    ) -> None:
        """Rewrite the &domains grid fields so ideal.exe regenerates a matching wrfinput_d01.

        Same in-place rewrite approach as point_wrf_restarts_at, done in Python
        so it works identically locally and on the cluster. wrfinput_d01 bakes
        in the domain size at ideal.exe-run time, so the namelist must be
        correct *before* the ideal.exe pre_app_call runs, not after.

        nproc_x/nproc_y (2026-09-12) override WRF's default auto-decomposition
        (-1/-1), which starts producing MPI patches narrower than WRF's
        minimum once e_sn is capped at 80 and rank counts climb past ~64 --
        see weak_scale_wrf_grid for why. em_b_wave_glass's namelist.input has
        no existing nproc_x/nproc_y lines, so these are INSERTED (right after
        e_sn) rather than replaced in place.

        Args:
            e_we (int): East-west grid points.
            e_sn (int): South-north grid points.
            dx (float): Grid spacing in x (m).
            dy (float): Grid spacing in y (m).
            time_step (float): Integration step (s).
            nproc_x (int | None): Explicit east-west MPI decomposition, or
                None to leave WRF's default (-1, auto) in place.
            nproc_y (int | None): Explicit south-north MPI decomposition, or
                None to leave WRF's default (-1, auto) in place.
        """
        namelist = os.path.join(self.run_dir, "namelist.input")
        dt = round(time_step)
        replacements = {
            "e_we": f" e_we                                = {e_we},   {e_we},   {e_we},",
            "e_sn": f" e_sn                                = {e_sn},   {e_sn},   {e_sn},",
            "dx": f" dx                                  = {round(dx)},",
            "dy": f" dy                                  = {round(dy)},",
            "time_step": f" time_step                           = {dt},",
        }
        if nproc_x is not None:
            replacements["nproc_x"] = f" nproc_x                             = {nproc_x},"
        if nproc_y is not None:
            replacements["nproc_y"] = f" nproc_y                             = {nproc_y},"
        try:
            with open(namelist) as f:
                lines = f.read().splitlines()
            rewritten = []
            seen = set()
            for line in lines:
                key = line.lstrip().split(None, 1)[0] if line.strip() else ""
                if key in replacements:
                    seen.add(key)
                    rewritten.append(replacements[key])
                else:
                    rewritten.append(line)
            missing = [
                k for k in ("nproc_x", "nproc_y") if k in replacements and k not in seen
            ]
            if missing:
                insert_at = len(rewritten)
                for i, line in enumerate(rewritten):
                    if line.lstrip().split(None, 1)[:1] == ["e_sn"]:
                        insert_at = i + 1
                        break
                for k in missing:
                    rewritten.insert(insert_at, replacements[k])
                    insert_at += 1
            if rewritten != lines:
                with open(namelist, "w") as f:
                    f.write("\n".join(rewritten) + "\n")
        except OSError as e:
            console.print(f"[yellow]Could not set WRF domain in {namelist}: {e}[/]")

    def set_warpx_checkpoint(self, ckptdir: str, intervals: int) -> None:
        """Set the deck to write only a `chk` checkpoint (no `diag1` plotfile).

        `diags_names = "diag1 chk"` on the command line does not survive jit's
        `bash -c` wrapper (the quotes collapse), so it goes in the deck instead.
        `diag1` is dropped so the checkpoint is the only I/O. Scalar overrides
        jit passes on the command line (max_step, n_cell, file_prefix) still win.

        Args:
            ckptdir (str): Directory the checkpoints should be written to.
            intervals (int): Checkpoint every `intervals` steps.
        """
        inputs = os.path.join(self.run_dir, "inputs")
        block = [
            "diagnostics.diags_names = chk",
            f"chk.intervals = {intervals}",
            "chk.diag_type = Full",
            "chk.format = checkpoint",
            f"chk.file_prefix = {ckptdir}/chk",
        ]
        try:
            with open(inputs) as f:
                lines = f.read().splitlines()
            rewritten = [
                line
                for line in lines
                if not line.lstrip().startswith(
                    ("diagnostics.diags_names", "chk.", "diag1.")
                )
            ]
            rewritten += block
            if rewritten != lines:
                with open(inputs, "w") as f:
                    f.write("\n".join(rewritten) + "\n")
            if (
                "_gkfs_mountdir" in ckptdir
                and inputs not in self.update_files_with_gkfs_mntdir
            ):
                self.update_files_with_gkfs_mntdir.append(inputs)
        except OSError as e:
            console.print(f"[yellow]Could not set WarpX checkpoint in {inputs}: {e}[/]")

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

    def set_qmc_walkers(self, total: int, ranks: int, per_rank_cap: int = 24000) -> None:
        """Rewrite every <walkers> in glass.xml to a memory- and total-bounded value.

        glass.xml's walkers value is per MPI rank. Two limits apply:
          * per node: per_rank * omp_threads * 73808 B * ranks_per_node must fit
            RAM, so per_rank is capped (``per_rank_cap``, ~24k for PROCS=8 / 2
            OMP threads -- job 44475461 hung at 16384/rank once thread-cloning is
            counted);
          * aggregate: QMCPACK's walker setup is ~O(total) and serial before the
            first VMC block, so ``total`` bounds the sum (job 45335439 stalled
            90 min at 14.7M; 16+1 ran 7.3M clean).
        Pinning the aggregate keeps the per-section config.h5 checkpoint a
        constant large size at every node count -- the size GLASS's incremental
        flush has to beat gekko's end drain on.

        Args:
            total (int): Target aggregate walker count across all ranks.
            ranks (int): Total application MPI ranks.
            per_rank_cap (int): Hard ceiling on walkers per rank (node-RAM bound).
        """
        per_rank = min(per_rank_cap, max(64, total // max(1, ranks)))
        xml = os.path.join(self.run_dir, "glass.xml")
        try:
            with open(xml) as f:
                text = f.read()
            rewritten = re.sub(
                r'(<parameter name="walkers"\s*>\s*)\d+(\s*</parameter>)',
                rf"\g<1>{per_rank}\g<2>",
                text,
            )
            if rewritten != text:
                with open(xml, "w") as f:
                    f.write(rewritten)
        except OSError as e:
            console.print(f"[yellow]Could not set QMC walkers in {xml}: {e}[/]")

    def set_qmc_steps(self, steps: int) -> None:
        """Rewrite every <qmc> section's steps in glass.xml to ``steps``.

        Each section writes one config.h5 at its end, so the steps set the
        compute time between two checkpoints while the checkpoint size (set
        by the walkers) stays the same.

        Args:
            steps (int): VMC steps per block in every section.
        """
        xml = os.path.join(self.run_dir, "glass.xml")
        try:
            with open(xml) as f:
                text = f.read()
            rewritten = re.sub(
                r'(<parameter name="steps"\s*>\s*)\d+(\s*</parameter>)',
                rf"\g<1>{max(1, steps)}\g<2>",
                text,
            )
            if rewritten != text:
                with open(xml, "w") as f:
                    f.write(rewritten)
        except OSError as e:
            console.print(f"[yellow]Could not set QMC steps in {xml}: {e}[/]")

    def set_qmc_sections(self, sections: int) -> None:
        """Resize glass.xml to ``sections`` <qmc> sections.

        Each section writes its own config.h5, so the section count is the
        number of checkpoint files left on GekkoFS at the end. The first
        section (the one with the warm-up) is kept; the second is the template
        for every further one.

        Args:
            sections (int): Total number of <qmc> sections (at least 2).
        """
        xml = os.path.join(self.run_dir, "glass.xml")
        try:
            with open(xml) as f:
                text = f.read()
            found = list(re.finditer(r"[ \t]*<qmc\b.*?</qmc>\n?", text, re.S))
            if len(found) < 2:
                console.print(f"[yellow]Need >= 2 <qmc> sections in {xml}[/]")
                return
            body = found[0].group(0) + found[1].group(0) * (max(2, sections) - 1)
            rewritten = text[: found[0].start()] + body + text[found[-1].end() :]
            if rewritten != text:
                with open(xml, "w") as f:
                    f.write(rewritten)
        except OSError as e:
            console.print(f"[yellow]Could not set QMC sections in {xml}: {e}[/]")

    def skip_last_qmc_checkpoint(self) -> None:
        """Set checkpoint="-1" (no config.h5) on the last <qmc> section of glass.xml."""
        xml = os.path.join(self.run_dir, "glass.xml")
        try:
            with open(xml) as f:
                text = f.read()
            last = list(re.finditer(r'<qmc\b[^>]*?checkpoint="[^"]*"', text))
            if not last:
                console.print(f"[yellow]No <qmc checkpoint=...> in {xml}[/]")
                return
            m = last[-1]
            tag = re.sub(r'checkpoint="[^"]*"', 'checkpoint="-1"', m.group(0))
            with open(xml, "w") as f:
                f.write(text[: m.start()] + tag + text[m.end() :])
        except OSError as e:
            console.print(f"[yellow]Could not skip the last QMC checkpoint: {e}[/]")

    def prepare_nek_input(self, nek_dir: str, steps: int, every: int) -> None:
        """Copy the Nek5000 mesh, map and par into run_dir/input and set the run length.

        run_dir/input is the stage-in dir on the GekkoFS legs and the session
        dir on the PFS leg, so all three legs read the same files.

        Args:
            nek_dir (str): Where <case>.re2, <case>.ma2 (genmap) and <case>.par live.
            steps (int): Time steps in the run (numSteps).
            every (int): Steps between two field dumps (writeInterval).
        """
        case = self.nek_case
        input_dir = os.path.join(self.run_dir, "input")
        try:
            os.makedirs(input_dir, exist_ok=True)
            # this Nek5000 build has no parRSB, so it needs the genmap partition
            for ext in ("re2", "ma2"):
                shutil.copy2(os.path.join(nek_dir, f"{case}.{ext}"), input_dir)
            with open(os.path.join(nek_dir, f"{case}.par")) as f:
                text = f.read()
            text = re.sub(r"(?m)^numSteps\s*=.*$", f"numSteps = {max(1, steps)}", text)
            text = re.sub(
                r"(?m)^writeInterval\s*=.*$", f"writeInterval = {max(1, every)}", text
            )
            with open(os.path.join(input_dir, f"{case}.par"), "w") as f:
                f.write(text)
        except OSError as e:
            console.print(
                f"[yellow]Could not prepare the Nek5000 input in {input_dir}: {e}[/]"
            )

    def prepare_btio_input(self, btio_class: str, steps: int, every: int) -> None:
        """Write run_dir/inputbt.data for a BT-IO run.

        BT reads it from its cwd; without it the class defaults apply and the
        dump interval falls back to 5.

        Args:
            btio_class (str): NPB class the binary was built for (C or D).
            steps (int): Time steps; 0 keeps the class default (needed to verify).
            every (int): Steps between two dumps (wr_interval).
        """
        grid, niter, dt = {"C": (162, 200, "0.0001d0"), "D": (408, 250, "0.00002d0")}[
            btio_class
        ]
        text = (
            f"{steps if steps > 0 else niter}\n{dt}\n{grid} {grid} {grid}\n"
            f"{max(1, every)} 0\n0 1000000\n"
        )
        try:
            with open(os.path.join(self.run_dir, "inputbt.data"), "w") as f:
                f.write(text)
        except OSError as e:
            console.print(f"[yellow]Could not write the BT-IO input: {e}[/]")

    def prepare_milc_input(
        self,
        outdir: str,
        lat: int,
        saves: int,
        traj: int,
        nx: int = 0,
        skip_last: bool = False,
    ) -> None:
        """Write run_dir/milc.in for su3_ora: `saves` sets on an L^4 lattice.

        su3_ora loops over parameter sets until EOF; each set runs `traj`
        trajectories and saves the gauge field to <outdir>/cfg.<k>, so every
        set is one checkpoint. The first set starts fresh, the rest continue.

        Args:
            outdir (str): Directory the configurations are saved to.
            lat (int): Lattice extent in every direction.
            saves (int): Number of parameter sets, i.e. saved configurations.
            traj (int): Trajectories per set (the compute between two saves).
            nx (int): x extent if not `lat` (0). MILC splits the lattice by the
                prime factors of the rank count, so 96 or 124 nodes need an x
                extent divisible by 3 or 31; nx = nodes keeps 64+1's sites/rank.
            skip_last (bool): The last set saves nothing (`forget`): the app ends
                right after it, so no compute phase is left to flush it in.
        """
        nx = nx or lat
        text = f"prompt 0\nnx {nx}\nny {lat}\nnz {lat}\nnt {lat}\niseed 8352\n"
        for k in range(saves):
            text += (
                f"\nwarms 0\ntrajecs {traj}\ntraj_between_meas {traj}\nbeta 6.0\n"
                f"steps_per_trajectory 4\nqhb_steps 4\n"
                f"{'fresh' if k == 0 else 'continue'}\nno_gauge_fix\n"
                + (
                    "forget\n"
                    if skip_last and k == saves - 1
                    else f"save_parallel {outdir}/cfg.{k}\n"
                )
            )
        with open(os.path.join(self.run_dir, "milc.in"), "w") as f:
            f.write(text)

    def prepare_vpic_input(self, particles: str, steps: int, compute_s: int) -> None:
        """Write run_dir/vpic.cfg for h5bench_write (VPIC-IO, contiguous 1D).

        Args:
            particles (str): Particles per rank and dump, h5bench units (e.g. 1M).
            steps (int): Number of dumps.
            compute_s (int): Emulated compute between two dumps, in seconds.
        """
        text = (
            "MEM_PATTERN=CONTIG\nFILE_PATTERN=CONTIG\n"
            f"TIMESTEPS={steps}\nDELAYED_CLOSE_TIMESTEPS=0\n"
            "COLLECTIVE_DATA=YES\nCOLLECTIVE_METADATA=YES\n"
            f"EMULATED_COMPUTE_TIME_PER_TIMESTEP={compute_s} s\n"
            f"NUM_DIMS=1\nDIM_1={particles}\nDIM_2=1\nDIM_3=1\n"
        )
        with open(os.path.join(self.run_dir, "vpic.cfg"), "w") as f:
            f.write(text)

    def select_regexes(self) -> None:
        """Pick the flush / stage-out / stage-in patterns for the current app.

        Keyed off ``app_call``, so this must run *after* the cluster and local
        blocks have finalized it. Running it earlier picks the pattern of
        whatever app_call happened to hold at the time, which silently left
        castro/warpx/qmc with an empty flush regex -- a flush that stages
        nothing at all.
        """
        # ├─ NAS BT-IO
        if "mpi_io_simple" in self.app_call:
            # one file per dump; a dump is complete once its file is closed
            self.regex_flush_match = ".*/btio\\.simple\\.out\\.\\d+$"
            self.regex_stage_out_match = ".*"
            self.regex_stage_in_match = ".*"
        # ├─ Nek5000
        elif "nek" in self.app_call:
            case = getattr(self, "nek_case", "turbPipe")
            # Anchored: Open MPI's per-rank chan0.f<N>.data.<r> / .metadata.<r>
            # temp files matched the bare prefix, so one flush
            # queued ~3600 of them and FTIO was still busy with them at app end
            # (BSC 48871819: GLASS stage-out 134 s vs GekkoFS 17 s).
            self.regex_flush_match = f".*/[a-zA-Z0-9]*{case}0\\.f\\d+$"
            self.regex_stage_out_match = f".*/[a-zA-Z0-9]*{case}0\\.f\\d+$"
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
            # rootdir until the post-app copy hits EDQUOT. That is exactly what
            # happened once: in.ckpt was switched from multi-file restarts
            # (ckpt.restart.<step>.<idx>) to single-writer ones
            # (ckpt.restart.<step>) without updating this regex, so every FTIO
            # trigger staged 0 items for a whole run (BSC 43752428: "Staging 0
            # item(s)" on every call). Match BOTH namings so in.ckpt's -v mp
            # knob can flip writer count without touching the regex again.
            # (Multi-file "%" restarts also once segfaulted under the libc
            # LD_PRELOAD intercept, BSC 43561676; --use_syscall is now the
            # default interception mode and does not hit that path.)
            self.regex_flush_match = ".*/ckpt\\.restart\\.\\d+(\\.(\\d+|base))?$"
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
            # 2026-09-22: two regex attempts here (both now REVERTED, see
            # remember.md for the full trail) tried to make this pattern also
            # match AMReX's transient <chk>.temp/ staging name, on the theory
            # that the trigger only ever scans the mount during that window.
            # Both were wrong, and the second was actively harmful: reading
            # posix_control.py's copy_file_and_unlink()/_orphaned_children()
            # shows GekkoFS-specific recovery machinery ALREADY EXISTS for
            # exactly this rename, but built the other way around --
            # `rename` re-keys only the directory entry, so after AMReX moves
            # <chk>.temp/ to <chk>/, a readdir of <chk> lists the children but
            # every stat on them misses (their metadata is still under the old
            # name); the fallback expects `item` to be the FINAL name and
            # retries with `.temp` appended. Matching `.temp` in the regex
            # itself fed that fallback an already-`.temp`-suffixed item, so
            # when the real rename had already happened by execute-time it
            # tried `.temp.temp` and found nothing -- exactly the 396
            # "had an error" / "No such file or directory" failures seen in
            # job 46345881. Reverted to the bare pattern below, consistent
            # with what copy_file_and_unlink() actually expects.
            # STILL UNRESOLVED: why job 46252936 (which ran this exact bare
            # pattern, before either regex experiment) got 0 flushes despite
            # FTIO locking a period -- not re-diagnosed this session, needs a
            # live look (debug-level trigger log or a manual mid-run mount
            # listing), not another blind regex edit. See remember.md.
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
            # QMCPACK rolls one config file per <qmc> section,
            # <mnt>/glass_heg.s<series>.config.h5, overwritten every block.
            self.regex_flush_match = ".*/glass_heg\\.s\\d+\\.config\\.h5$"
            self.regex_stage_out_match = ".*/glass_heg\\.s\\d+\\.config\\.h5$"
            self.regex_stage_in_match = ".*"
        # ├─ HACC-IO
        elif "HACC" in self.app_call:
            # One file per rank and checkpoint: <mnt>/mpi[.<loop>]-PartNNNNNNNN-of-
            # NNNNNNNN.data (.<loop> only with HACC_STEP_FILES=1).
            self.regex_flush_match = ".*/mpi(\\.\\d+)?-Part\\d+-of-\\d+\\.data$"
            self.regex_stage_out_match = ".*/mpi(\\.\\d+)?-Part\\d+-of-\\d+\\.data$"
            self.regex_stage_in_match = ".*"
        # ├─ VPIC-IO
        elif "h5bench_write" in self.app_call:
            # One file per dump: <mnt>/vpic.h5.<i>; vpic.h5 itself stays empty.
            self.regex_flush_match = ".*/vpic\\.h5\\.\\d+$"
            self.regex_stage_out_match = ".*"
            self.regex_stage_in_match = ".*"
        # ├─ MILC
        elif "su3_ora" in self.app_call:
            # One shared file per saved configuration: <mnt>/cfg.<k> + cfg.<k>.info.
            self.regex_flush_match = ".*/cfg\\.\\d+(\\.info)?$"
            self.regex_stage_out_match = ".*/cfg\\.\\d+(\\.info)?$"
            self.regex_stage_in_match = ".*"
        # ├─ E3SM-IO
        elif "e3sm_io" in self.app_call:
            # One history file pair per checkpoint: <mnt>/e3sm.<k>_h0.nc, _h1.nc,
            # with blob one subfile per node: <mnt>/e3sm.<k>_h0.nc.<NNNN>.
            self.regex_flush_match = ".*/e3sm\\.\\d+_h[01]\\.nc(\\.\\d{4})?$"
            self.regex_stage_out_match = ".*/e3sm\\.\\d+_h[01]\\.nc(\\.\\d{4})?$"
            self.regex_stage_in_match = ".*"
        # └─ Other
        else:
            self.regex_flush_match = ""
            self.regex_stage_out_match = ".*"
            self.regex_stage_in_match = ".*"

        self.regex_match = self.regex_flush_match
