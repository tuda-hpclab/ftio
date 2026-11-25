"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Okt 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

from ftio.api.gekkoFs.jit.execute_and_wait import execute_block
from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.setup_helper import (
    create_hostfile,
    flaged_call,
    jit_print,
)


def init_gekko(settings: JitSettings) -> None:
    """Creates GekkoFs hostfile and directories (root and mount)

    Args:
        settings (JitSettings): Jit settings
    """
    if not settings.exclude_daemon:
        create_hostfile(settings)
        calls = []
        # set debug flag
        # if settings.cluster:
        #     # Create directories
        #     calls.append(
        #         f"srun --jobid={settings.job_id} {settings.app_nodes_command} --disable-status -N {settings.app_nodes} "
        #         f"--ntasks={settings.app_nodes} --cpus-per-task={settings.procs_daemon} --ntasks-per-node=1 --overcommit --overlap "
        #         f"--oversubscribe --mem=0 mkdir -p {settings.gkfs_mntdir}"
        #             )
        #     calls.append(
        #         f"srun --jobid={settings.job_id} {settings.app_nodes_command} --disable-status -N {settings.app_nodes} "
        #         f"--ntasks={settings.app_nodes} --cpus-per-task={settings.procs_daemon} --ntasks-per-node=1 --overcommit --overlap "
        #         f"--oversubscribe --mem=0 mkdir -p {settings.gkfs_rootdir}"
        #             )
        # else:
        #     calls.append(f"mkdir -p {settings.gkfs_mntdir}")
        #     calls.append(f"mkdir -p {settings.gkfs_rootdir}")
        # Proactively unmount any stale FUSE mount before creating dirs.
        # Needed when a previous JIT run was hard-killed and left the mount
        # point in a zombie state (daemon dead but kernel mount still registered).
        unmount_call = flaged_call(
            settings,
            f"fusermount -uz {settings.gkfs_mntdir} 2>/dev/null || umount -l {settings.gkfs_mntdir} 2>/dev/null || true",
            nodes=settings.app_nodes,
            procs_per_node=1,
            exclude=["ftio", "demon", "proxy", "cargo"],
        )
        execute_block(unmount_call, raise_exception=False, dry_run=settings.dry_run)
        calls.append(
            f"mkdir -p {settings.gkfs_mntdir} 2>/dev/null || [ -d {settings.gkfs_mntdir} ]"
        )
        calls.append(
            f"mkdir -p {settings.gkfs_rootdir} 2>/dev/null || [ -d {settings.gkfs_rootdir} ]"
        )
        jit_print("[cyan]Creating directories[/]")
        for call in calls:
            tmp = flaged_call(
                settings,
                call,
                nodes=settings.app_nodes,
                procs_per_node=1,
                exclude=["ftio", "demon", "proxy", "cargo"],
            )
            _ = execute_block(tmp)
