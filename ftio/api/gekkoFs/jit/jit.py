"""
This file provides functions to execute the JIT script, including setting up the environment,
allocating resources, starting necessary services, and managing the workflow from staging
data in to staging out. It also handles pre- and post-application calls.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Aug 2024

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import sys
import traceback

from rich.console import Console

from ftio.api.gekkoFs.jit.jitsettings import JitSettings
from ftio.api.gekkoFs.jit.jittime import JitTime
from ftio.api.gekkoFs.jit.setup_core import (
    post_call,
    pre_call,
    stage_in,
    stage_out,
    start_application,
    start_cargo,
    start_ftio,
    start_fuse,
    start_gekko_daemon,
    start_gekko_proxy,
)
from ftio.api.gekkoFs.jit.setup_helper import (  # set_env,
    allocate,
    cancel_jit_jobs,
    check_port,
    clear_bandwidth,
    get_address_cargo,
    get_address_ftio,
    hard_kill,
    install_signal_handlers,
    log_dir,
    log_execution,
    parse_options,
    print_settings,
    save_bandwidth,
    save_hosts_file,
    set_dir_gekko,
    snapshot_directory,
    soft_kill,
)

console = Console()


def main() -> None:
    """Executes the JIT script by setting up the environment,
    allocating resources, starting necessary services,
    and managing the workflow from staging in to staging out.
    """
    console.print("\n\n[bold green]################ JIT ################[/]\n")
    settings = JitSettings()

    # Handle --help before the try/finally so the kill sequence never runs.
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        parse_options(settings, sys.argv[1:])  # argparse prints help and exits here

    runtime = JitTime()
    error = 0

    try:
        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        parse_options(settings, sys.argv[1:])

        if not settings.tmp_dir:
            raise RuntimeError(
                "STAGE_DIR is not set. "
                "Either define it in the environment or specify it in this file (jitsettings)."
            )

        # Register cleanup for SIGINT/SIGTERM/SIGHUP. timeout(1) and schedulers
        # send SIGTERM; without a handler the interpreter dies before the
        # finally-block cleanup runs, orphaning the daemon and predictor pool.
        install_signal_handlers(settings)

        cancel_jit_jobs(settings)
        allocate(settings)

        log_dir(settings)
        log_execution(settings)

        get_address_ftio(settings)
        # Before any service is launched: the daemon, FUSE and the app are all
        # handed address_ftio:port_ftio, so the port has to be settled first.
        check_port(settings)
        get_address_cargo(settings)
        set_dir_gekko(settings)

        print_settings(settings)

        # ------------------------------------------------------------------
        # Start services
        # ------------------------------------------------------------------
        clear_bandwidth(settings)
        start_gekko_daemon(settings)
        start_gekko_proxy(settings)
        start_cargo(settings)
        start_ftio(settings)
        start_fuse(settings)

        # ------------------------------------------------------------------
        # Workflow execution
        # ------------------------------------------------------------------
        pre_call(settings)
        stage_in(settings, runtime)

        # Signal predictor_jit to start processing predictions now that the
        # application is about to launch (pre-application I/O is excluded).
        if not settings.exclude_ftio and settings.app_start_file:
            open(settings.app_start_file, "w").close()

        start_application(settings, runtime)
        stage_out(settings, runtime)
        post_call(settings)

        # ------------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------------
        runtime.print_and_save_time(settings)
        save_bandwidth(settings)
        log_execution(settings)
        save_hosts_file(settings)
        snapshot_directory(settings)

    except Exception as e:
        console.print(f"[bold red]JIT failed:[/] {e}\n")
        traceback.print_exc()
        error = 1

    finally:
        # Cleanup
        soft_kill(settings)
        hard_kill(settings)

    if error == 0:
        console.print("[bold green]############### JIT completed ###############[/]")
    else:
        console.print("[bold red]################# JIT failed #################[/]")

    sys.exit(error)


if __name__ == "__main__":
    main()
