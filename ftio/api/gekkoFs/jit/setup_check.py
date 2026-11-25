"""
This file provides functions to check the setup for the JIT script, including verifying
configuration files, generating test scripts, and executing them to ensure the environment
is correctly configured.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import os
import time
from datetime import datetime

from rich.console import Console

from ftio.api.gekkoFs.jit.execute_and_wait import execute_block
from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.setup_helper import jit_print

console = Console()


def check_setup(settings: JitSettings):
    """
    Checks the setup for the JIT script by verifying configuration files, generating
    and executing a test script, and ensuring the environment is correctly configured.

    Args:
        settings (JitSettings): The settings object containing configuration details
                                for the JIT environment.

    Returns:
        None
    """

    if not settings.exclude_all:
        # Display MPI hostfile
        if settings.cluster:
            mpi_hostfile_path = os.path.expanduser(f"{settings.mpi_hostfile}")
            with open(mpi_hostfile_path) as file:
                mpi_hostfile_content = file.read()
            console.print(f"[cyan]MPI hostfile:\n{mpi_hostfile_content}[/]")

        # Display GekkoFS hostfile
        gekkofs_hostfile = settings.gkfs_hostfile
        with open(gekkofs_hostfile) as file:
            gekkofs_hostfile_content = file.read()
        console.print(f"[cyan]Geko hostfile:\n{gekkofs_hostfile_content}[/]")

        # ls_command = f"LD_PRELOAD={settings.gkfs_intercept} LIBGKFS_HOSTS_FILE={gekkofs_hostfile} ls {settings.gkfs_mntdir}"
        # files = subprocess.check_output(ls_command, shell=True).decode()
        # console.print(f"[cyan]geko_ls {gkfs_mntdir}: \n{files}[/]")

        if settings.cluster and not settings.exclude_daemon:
            additional_arguments = ""
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            file = create_test_file("test.sh" + timestamp, settings)
            if settings.use_mpirun:
                # if not settings.exclude_ftio:
                #     additional_arguments += f"-x LIBGKFS_METRICS_IP_PORT={settings.address_ftio}:{settings.port_ftio} -x LIBGKFS_ENABLE_METRICS=on "
                if not settings.exclude_proxy:
                    additional_arguments += (
                        f"-x LIBGKFS_PROXY_PID_FILE={settings.gkfs_proxyfile} "
                    )
                if not settings.exclude_daemon:
                    log_modules = "all" if settings.debug_lvl > 1 else "errors"
                    additional_arguments += (
                        f"-x LIBGKFS_LOG={log_modules} "
                        f"-x LIBGKFS_LOG_OUTPUT={settings.gkfs_client_log} "
                        f"-x LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile} "
                    )
                    if settings.preload_via_export:
                        additional_arguments += (
                            f"-x LD_PRELOAD={settings.gkfs_intercept} "
                        )

                test_invocation = file
                if (
                    not settings.preload_via_export
                    and not settings.exclude_daemon
                    and settings.gkfs_intercept
                ):
                    test_invocation = (
                        f'bash -c "LD_PRELOAD={settings.gkfs_intercept} {file}"'
                    )
                call = (
                    f" mpiexec -np {settings.app_nodes} --oversubscribe "
                    f"--hostfile {settings.mpi_hostfile} --map-by node "
                    f"{additional_arguments} "
                    f"{test_invocation}"
                )
            else:
                # if not settings.exclude_ftio:
                #     additional_arguments += f"LIBGKFS_ENABLE_METRICS=on,LIBGKFS_METRICS_IP_PORT={settings.address_ftio}:{settings.port_ftio},"
                if not settings.exclude_proxy:
                    additional_arguments += (
                        f"LIBGKFS_PROXY_PID_FILE={settings.gkfs_proxyfile},"
                    )
                if not settings.exclude_daemon:
                    log_modules = "all" if settings.debug_lvl > 1 else "errors"
                    additional_arguments += (
                        f'LIBGKFS_LOG="{log_modules}",'
                        f"LIBGKFS_LOG_OUTPUT={settings.gkfs_client_log},"
                        f"LIBGKFS_HOSTS_FILE={settings.gkfs_hostfile},"
                    )
                    if settings.preload_via_export:
                        additional_arguments += f"LD_PRELOAD={settings.gkfs_intercept},"

                test_invocation = file
                if (
                    not settings.preload_via_export
                    and not settings.exclude_daemon
                    and settings.gkfs_intercept
                ):
                    test_invocation = (
                        f'bash -c "LD_PRELOAD={settings.gkfs_intercept} {file}"'
                    )
                call = (
                    f" srun --export=ALL,{additional_arguments}LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')} "
                    f"--jobid={settings.job_id} {settings.app_nodes_command} --disable-status "
                    f"-N {settings.app_nodes} --ntasks={settings.app_nodes} "
                    f"--cpus-per-task=1 --ntasks-per-node=1 "
                    f"--overcommit --overlap --oversubscribe --mem=0 "
                    f"{test_invocation} "
                )
            # test script
            jit_print("[cyan]Checking test file")
            try:
                out = execute_block(call, False)
                console.print(f"{out}")
                # remove the created file
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                jit_print(f"[red] Error running test script:\n{e}")
    time.sleep(1)


def create_test_file(name: str, settings: JitSettings) -> str:
    """
    Creates a shell script to test the JIT environment by performing basic operations
    such as listing and checking the status of the GekkoFS mount directory.

    Args:
        name (str): The name of the shell script file to be created.
        settings (JitSettings): The settings object containing configuration details
                                for the JIT environment.

    Returns:
        str: The path to the created shell script file.
    """
    # Define the content of the shell script
    script_content = "#!/bin/bash\n"
    script_content += "myhostname=$(hostname)\n"
    script_content += f"statcall=$(stat {settings.gkfs_mntdir})\n"
    if settings.app_nodes == 1:
        script_content += f"lscall=$(ls -lR {settings.gkfs_mntdir})\n"
    else:
        script_content += f"lscall=$(ls {settings.gkfs_mntdir})\n"
    script_content += """
    echo -e "Hello I am ${myhostname} and stat output: \\n${statcall}\\n The directory contains:\\n${lscall}\\n"
    """

    # Write the content to a file called tet.sh
    file_path = os.path.join(os.getcwd(), name)
    with open(file_path, "w") as file:
        file.write(script_content)

    os.chmod(file_path, 0o755)
    return file_path
