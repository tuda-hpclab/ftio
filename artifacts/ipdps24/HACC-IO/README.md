# HACC-IO

Either [extract](#extract-the-provided-trace-file) or [generate](#generate-the-experiment-on-your-system) the trace files as described below. 
Afterward, perform the [offline evaluation](#ftio-offline-evaluation) on the trace file or set up and execute the [online evaluation](#ftio-online-evaluation).

## Extract the provided trace file
Download and extract the file data.zip as described [here](/artifacts/ipdps24/README.md#extracting-the-data-set).
Once extracted, you can find the HACC-IO trace under `data/application_traces/HACC-IO`.

<p align="right"><a href="#hacc-io">⬆</a></p>


## Generate the experiment on your system:
Download and extract the repository of the modified HACC-IO version from [here](https://github.com/A-Tarraf/hacc-io). 

<p align="right"><a href="#hacc-io">⬆</a></p>



## FTIO: Offline Evaluation 
If you didn't install `ftio`, first check out version 0.0.1 as described [here](/artifacts/ipdps24/README.md#ftio-version), then install it as described [here](https://github.com/tuda-hpclab/ftio#installation).
The instructions below assume that the trace file is called `3072.jsonl`, which after [extracting](#extract-the-provided-trace-file) the trace file, should be located in `data/application_traces/HACC-IO`.

To get the dominant frequency with `ftio` simply call:
```sh
ftio 3072.jsonl  -v -e no  
```
Remove `-e no` if you want to obtain figures (Fig 12 and 14 from the paper).


<p align="right"><a href="#hacc-io">⬆</a></p>


## FTIO: Online Evaluation
- [Install TMIO](https://github.com/tuda-hpclab/tmio#installation) 
- If you didn't install `ftio`, first check out version 0.0.1 as described [here](/artifacts/ipdps24/README.md#ftio-version), then install it as described [here](https://github.com/tuda-hpclab/ftio#installation). If you don't want to reproduce the experiment from the paper but rather just try FTIO online, then you can skip checking out to a particular version. 

In case you are interested in the original version of the benchmark, it is [here](https://github.com/glennklockwood/hacc-io). For our experiments, we modified the code such that it is executed in a loop.
Navigate to the folder where the source code of the _modified_ version of HACC-IO is presented in `data/application_traces/HACC-IO/src` or clone it from here:

```bash
git clone https://github.com/A-Tarraf/hacc-io.git
```

Modify the provided Makefile to point to your TMIO git repository:

```bash
⋮
TMIO_REPO= /d/github/TMIO #modify this line
⋮
```

A provided Makefile takes care of including TMIO into the HACC-IO source code. For that, execute:
```bash
make run_with_include_static PROCS=8
```
Once completed, you should see a file called x.jsonl, where x stands for the number of processes, which for the code above is 8. 

To run the code on your cluster, use the provided `sbatch.sh` script and modify it as necessary. Depending on the number of processes you specified in this script, an x.jsonl file is generated just as previously. FTIO (more specifically `predictor`) should monitor this file by executing, for example, on the login node of your system:
```bash
predictor  path_to_file/x.jsonl  -f 10  
```
execute this call and then submit your `sbatch.sh` script with:
```bash
sbatch sbatch.sh
```
`ftio` is called internally in `predictor` whenever new traces are appended to the monitored fill x.jsonl. This yields online predictions during the runtime of HACC-IO. Additionally add the `-w` flag to the call of `predictor`, in case you want to utilize the automatic time window adaption as we did in the paper. With the `-x 3` flag, you can trigger the time window adaption after three hits, which is the default behavior.

<p align="right"><a href="#ior">⬆</a></p>