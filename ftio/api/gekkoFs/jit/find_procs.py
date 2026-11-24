"""
Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.1.0
Date: Mai 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

import argparse
import math

from rich.console import Console

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Calculate problem_size and dim based on procs per node and total nodes."
)

# All arguments are now flags
parser.add_argument(
    "-p",
    "--procs_per_node",
    type=int,
    help="Number of processes per node (e.g., 64)",
    required=True,
)
parser.add_argument(
    "-n",
    "--nodes",
    type=int,
    nargs="+",
    help="List of total number of nodes (e.g., 19 20 21)",
    required=True,
)
parser.add_argument(
    "-s",
    "--scale_problem",
    type=int,
    help="Scale problem factor (e.g., 100)",
    default=100,
)
parser.add_argument(
    "-w",
    "--weak_scaling",
    action="store_true",
    help="Enable weak scaling (problem size constant)",
)

args = parser.parse_args()

# Set up the rich console for printing
CONSOLE = Console()
LARGE_CALL = ""
ADJUSTED_PROBLEM_SIZE = None  # Will store the single problem size for all dims
DIMS = []  # To store all dims
PROCS_PER_NODE = []  # To store all dims
NODES = []  # To store all


# Function to calculate the Least Common Multiple (LCM) of a list of numbers
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


# Loop over each value of node (can handle list of nodes)
for node in args.nodes:
    # Calculate initial total processes
    tmp_procs_per_node = args.procs_per_node
    total_procs = tmp_procs_per_node * node
    sol = False  # Track if we adjusted procs_per_node

    while True:
        # Calculate the cube root and check if it's an integer
        dim = round(total_procs ** (1 / 3))
        if dim**3 == total_procs:
            sol = True  # Mark that adjustment was successful
            break  # Found a perfect cube
        elif tmp_procs_per_node > 1:
            tmp_procs_per_node -= 1  # Decrease procs_per_node by 1
            total_procs = tmp_procs_per_node * node  # Recompute total_procs
        else:
            CONSOLE.print(
                f"[bold yellow]Cannot adjust procs_per_node further for {node} nodes. Skipping.[/bold yellow]"
            )
            sol = False
            break  # Break out of the loop when we can't adjust further

    if sol:
        DIMS.append(dim)  # Store this dim value
        PROCS_PER_NODE.append(tmp_procs_per_node)
        NODES.append(node)
    else:
        continue  # Skip the rest of the loop and move to the next node

    # If weak scaling is enabled, the problem size remains constant
    if args.weak_scaling:
        # Calculate the problem_size as dim * scale_problem for the first node (we use the same for all)
        problem_size = dim * args.scale_problem

        # Store this adjusted problem size for all nodes (to propagate)
        if len(args.nodes) > 1:
            if not ADJUSTED_PROBLEM_SIZE:
                ADJUSTED_PROBLEM_SIZE = problem_size

            # Adjust problem size to be divisible by the LCM of all dims
            if DIMS:
                lcm_value = DIMS[0]
                for d in DIMS[1:]:
                    lcm_value = lcm(lcm_value, d)

                # Adjust problem_size to be divisible by the LCM of all dims
                if lcm_value != 0 and ADJUSTED_PROBLEM_SIZE % lcm_value != 0:
                    ADJUSTED_PROBLEM_SIZE += lcm_value - ADJUSTED_PROBLEM_SIZE % lcm_value

    else:
        # Calculate problem_size as dim * scale_problem (this is based on the first `dim` value)
        problem_size = dim * args.scale_problem

    # Print the result using rich console
    CONSOLE.print(f"\n[bold green]total nodes[/bold green]: {node}")
    CONSOLE.print(f"[bold green]procs per node[/bold green]: {tmp_procs_per_node}")
    CONSOLE.print(f"[bold green]problem_size[/bold green]: {problem_size}")
    CONSOLE.print(f"[bold green]dims[/bold green]: {dim}")
    CONSOLE.print(
        f"[bold cyan]Total Procs (dim * dim * dim)[/bold cyan]: {dim * dim * dim} == {total_procs}"
    )

    # Generate calls for each dim, but use the same `adjusted_problem_size` for all dims
    call = (
        f"jit -n {node+1} -j $JOBID -p {tmp_procs_per_node} -u -s -a s3d --app-flags "
        f"'{problem_size} {problem_size} {problem_size} {dim} {dim} {dim} 0 F .'"
    )
    CONSOLE.print(f"[bold cyan]S3D-IO call for dim {dim}:[/bold cyan]: {call}")

    CONSOLE.print("\n[bold cyan]S3D-IO all calls:[/bold cyan]:")
    print(f"{call} ; {call} -x ; {call} -e ftio ; ")

    # Add to large_call if there are multiple nodes
    if len(args.nodes) > 1:
        # LARGE_CALL += f"{call} ; {call} -x ; {call} -e ftio ; "
        LARGE_CALL += f"{call} -x ; {call} -e cargo ; "

# Print the global call if multiple nodes are given
if len(args.nodes) > 1:
    CONSOLE.print("\n[bold cyan]S3D-IO global calls:[/bold cyan]:")
    print(f"{LARGE_CALL}")

# If weak scaling is enabled, output the updated problem_size in all calls
if args.weak_scaling and len(NODES) > 1:
    CONSOLE.print(
        "\n[bold magenta]Updated S3D-IO calls with adjusted problem_size:[/bold magenta]:"
    )
    txt = ""
    for i, dim in enumerate(DIMS):
        updated_call = (
            f"jit -n {NODES[i]+1} -j $JOBID -p {PROCS_PER_NODE[i]}  -s -a s3d --app-flags "
            f"'{ADJUSTED_PROBLEM_SIZE} {ADJUSTED_PROBLEM_SIZE} {ADJUSTED_PROBLEM_SIZE} {dim} {dim} {dim} 0 F .'"
        )
        txt += f"{updated_call} ; {updated_call} -x ; {updated_call} -e ftio ; "
        # txt += f"{updated_call} -e cargo ; {updated_call} -x  ; "
    print(txt)


# 50
# jit -n 19 -j $JOBID  -p 12 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' ; jit -n 19 -j $JOBID  -p 12 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' -x ; jit -n 19 -j $JOBID  -p 12 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' -e ftio ; jit -n 17 -j $JOBID  -p 32 -u -s -a s3d --app-flags '400 400 400 8 8 8 0 F .' ; jit -n 17 -j $JOBID  -p 32 -u -s -a s3d --app-flags '400 400 400 8 8 8 0 F .' -x ; jit -n 17 -j $JOBID  -p 32 -u -s -a s3d --app-flags '400 400 400 8 8 8 0 F .' -e ftio ; jit -n 13 -j $JOBID  -p 18 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' ; jit -n 13 -j $JOBID  -p 18 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' -x ; jit -n 13 -j $JOBID  -p 18 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' -e ftio ; jit -n 10 -j $JOBID  -p 24 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' ; jit -n 10 -j $JOBID  -p 24 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' -x ; jit -n 10 -j $JOBID  -p 24 -u -s -a s3d --app-flags '300 300 300 6 6 6 0 F .' -e ftio ; jit -n 9 -j $JOBID  -p 64 -u -s -a s3d --app-flags '400 400 400 8 8 8 0 F .' ; jit -n 9 -j $JOBID  -p 64 -u -s -a s3d --app-flags '400 400 400 8 8 8 0 F .' -x ; jit -n 9 -j $JOBID  -p 64 -u -s -a s3d --app-flags '400 400 400 8 8 8 0 F .' -e ftio ;

# python3 find_procs.py -p 64 -n 18 16 9 8  -s 100

# jit -n 19 -j $JOBID  -p 12 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' ; jit -n 19 -j $JOBID  -p 12 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' -x ; jit -n 19 -j $JOBID  -p 12 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' -e ftio ; jit -n 17 -j $JOBID  -p 32 -u -s -a s3d --app-flags '800 800 800 8 8 8 0 F .' ; jit -n 17 -j $JOBID  -p 32 -u -s -a s3d --app-flags '800 800 800 8 8 8 0 F .' -x ; jit -n 17 -j $JOBID  -p 32 -u -s -a s3d --app-flags '800 800 800 8 8 8 0 F .' -e ftio ; jit -n 13 -j $JOBID  -p 18 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' ; jit -n 13 -j $JOBID  -p 18 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' -x ; jit -n 13 -j $JOBID  -p 18 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' -e ftio ; jit -n 10 -j $JOBID  -p 24 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' ; jit -n 10 -j $JOBID  -p 24 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' -x ; jit -n 10 -j $JOBID  -p 24 -u -s -a s3d --app-flags '600 600 600 6 6 6 0 F .' -e ftio ; jit -n 8 -j $JOBID  -p 49 -u -s -a s3d --app-flags '700 700 700 7 7 7 0 F .' ; jit -n 8 -j $JOBID  -p 49 -u -s -a s3d --app-flags '700 700 700 7 7 7 0 F .' -x ; jit -n 8 -j $JOBID  -p 49 -u -s -a s3d --app-flags '700 700 700 7 7 7 0 F .' -e ftio ;
