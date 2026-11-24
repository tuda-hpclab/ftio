# Artifacts Reproducibility

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17713783.svg)](https://doi.org/10.5281/zenodo.17713783)


Below, we describe how to reproduce the experiments in the Paper entitled:
"Improving I/O Phase Predictions in FTIO Using Hybrid Wavelet-Fourier Analysis," which was published at the Frontiers in High Performance Computing Journal for the call High Performance Big Data Systems. 

Before you start, first set up FTIO with the [version](#ftio-version) used in the journal.

This document contains the following topics

- [Artifacts Reproducibility](#artifacts-reproducibility)
	- [Prerequisites](#prerequisites)
		- [FTIO Version](#ftio-version)
		- [Extracting the Data Set:](#extracting-the-data-set)
	- [Artifacts](#artifacts)
		- [1. Running Example and Overhead](#1-running-example-and-overhead)
		- [2. Case Studies](#2-case-studies)
			- [2.1. LAMMPS](#21-lammps)
			- [2.2 Nek5000](#22-nek5000)
			- [2.3 Modified HACC-IO](#23-modified-hacc-io)
		- [3. Limitations of FTIO](#3-limitations-of-ftio)
		- [4. Use Case: I/O Scheduling](#4-use-case-io-scheduling)
	- [Citation](#citation)

## Prerequisites 
Before you start, there are two prerequisites:
1. Install the correct [FTIO version](#ftio-version) 
2. Depending on what you want to test, you need to [download and extract](#extracting-the-data-set) the data set from [Zenodo](https://doi.org/10.5281/zenodo.17713783).

### FTIO Version

For all the cases below, `ftio` first needs to be installed (see [Installation](https://github.com/tuda-hpclab/ftio?tab=readme-ov-file#installation)). We used `ftio` version 0.0.7 for the experiment in the paper. To get this version, simply execute the following code:
```sh
git checkout v0.0.7 
```


### Extracting the Data Set:
download the zip file from [here](https://zenodo.org/records/17713784) or using wget in a bash terminal:
```sh
wget https://zenodo.org/records/17713784/files/data.zip?download=1
```
Next, unzip the file
```sh
unzip data.zip
```
This extracts the necessary traces and experiments:

```sh
data
├── dlio
│   ├── profiles
│   │   └── 48116802-0.profile
│   └── traces
│       └── 48116802-0.trace
└── hacc-io
    ├── 47969612_n96_HACC_ASYNC_IO.out
    ├── 47969614_n192_HACC_ASYNC_IO.out
    ├── 47969620_n384_HACC_ASYNC_IO.out
    ├── 47969623_n768_HACC_ASYNC_IO.out
    ├── 47969634_n1536_HACC_ASYNC_IO.out
    ├── 47970248_n3072_HACC_ASYNC_IO.out
    ├── 47970493_n4608_HACC_ASYNC_IO.out
    ├── 47970668_n6144_HACC_ASYNC_IO.out
    ├── 47970840_n9216_HACC_ASYNC_IO.out
    ├── Job47969612_96.msgpack
    ├── Job47969614_192.msgpack
    ├── Job47969620_384.msgpack
    ├── Job47969623_768.msgpack
    ├── Job47969634_1536.msgpack
    ├── Job47970248_3072.msgpack
    ├── Job47970493_4608.msgpack
    ├── Job47970668_6144.msgpack
    ├── Job47970840_9216.msgpack
    └── sbatch.sh

```

## Artifacts
### 1. Signal Reconstruction with Multiple Frequencies
In Section 4.1 of the paper "Improving I/O Phase Predictions in FTIO Using Hybrid Wavelet-Fourier Analysis," FTIO's ability to improve signal characterization was demonstrated based on HACC-IO with 384 processes (4 nodes on Lichtenberg). 
As a first step, either [extract the traces](#extracting-the-data-set) used in the paper or [recreate it](#hacc-io). With the trace named `Job47969620_384.msgpack`, simply execute `ftio` with: 
```bash
ftio Job47969620_384.msgpack -n 10
```
The `-n` flag specifies the number of frequencies to reconstruct. In the paper this number was set to 2,3,5, and 10 to create Figure 1.  

### 2. Fourier Fitting
In Section 6.2 of the paper "Improving I/O Phase Predictions in FTIO Using Hybrid Wavelet-Fourier Analysis," FTIO's ability to improve signal characterization was demonstrated based on HACC-IO with 9216 processes using Fourier fitting. 
Similar to [the previous section](Signal Reconstruction with Multiple Frequencies) the trace is first needed. For that, either [extract the traces](#extracting-the-data-set) used in the paper or [recreate it](#hacc-io). \
With the trace named `Job47970840_9216.msgpack`, simply execute `ftio` with:
```bash
ftio Job47970840_9216.msgpack -f 1 -n 10 --fourier_fit   -v  
```

Once the call completes, FTIO prints the following:
```bash
❯ ftio Job47970840_9216.msgpack -f 1 -n 10 --fourier_fit   -e no -v
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                              Ftio                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Author:  Ahmad Tarraf
Date:    2025-11-25
Version:  0.0.7
License:  BSD

Current file: Job47970840_9216.msgpack


Data imported in: 3.65 s
Frequency Analysis: DFT
Mode: write_sync
Executing: Discretization

╭─ Discretization ──────────────────╮
│ Time window: 180.28 s             │
│ Frequency step: 5.547e-03 Hz      │
│ Sampling frequency:  1.000e+00 Hz │
│ Expected samples: 180             │
│ Abstraction error: 2.72942e-03    │
╰───────────────────────────────────╯

Discretization finished: 0.321 s
Executing: DFT + Z-score + None

╭─ Fourier Fit ────────────────────────────╮
│ maxfev:50000                             │
│ Fourier fit improvement:                 │
│ MSE reduced from 1.416e+21 to 9.755e+20  │
│ --> 31.11% improvement                   │
╰──────────────────────────────────────────╯
╭─ DFT ─────────────────────────────────────────────────────────────────────────────────╮
│ Ranks: 9216                                                                           │
│ Start time: 6.18 s                                                                    │
│ End time: 186.46 s                                                                    │
│ Total bytes: 1.40e+13 bytes                                                           │
│ Ignored bytes: 0.00e+00 bytes                                                         │
│                                                                                       │
│ ╭─ Z-score ─────────────────────────────────────────────────────────────────────────╮ │
│ │ Spectrum: Power spectrum                                                          │ │
│ │ mean: 1.111e-02                                                                   │ │
│ │ std: 4.258e-02                                                                    │ │
│ │ Frequencies with Z-score > 3 -> 2 candidates                                      │ │
│ │          + Z > Z_max*80.0% > 3 -> 1 candidates                                    │ │
│ │ Dominant frequency at: 5.556e-02 Hz (T = 18.000 s, k = 10) -> confidence: 78.531% │ │
│ ╰───────────────────────────────────────────────────────────────────────────────────╯ │
│                                                                                       │
│                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────╯

DFT + Z-score finished: 0.839 s
Prediction results:
Frequency: 5.556e-02 Hz-> 18.0 s
Confidence: 78.53 %

Top 10 Frequencies:
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Freq (Hz) ┃ Conf. (%) ┃ Amplitude ┃        Phi ┃                          Cosine Wave ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1.803e-03 │    100.00 │ 7.650e+12 │  4.873e+00 │ 7.65e+12*cos(2π*1.80e-03*t+4.87e+00) │
│ 5.229e-02 │    100.00 │ 7.271e+12 │  2.287e+00 │ 7.27e+12*cos(2π*5.23e-02*t+2.29e+00) │
│ 4.288e-02 │    100.00 │ 3.121e+12 │ -1.913e+00 │ 3.12e+12*cos(2π*4.29e-02*t-1.91e+00) │
│ 4.894e-02 │    100.00 │ 8.966e+12 │  1.144e+00 │ 8.97e+12*cos(2π*4.89e-02*t+1.14e+00) │
│ 6.630e-03 │    100.00 │ 3.666e+12 │ -1.524e+00 │ 3.67e+12*cos(2π*6.63e-03*t-1.52e+00) │
│ 1.608e-02 │    100.00 │ 3.293e+12 │ -6.146e-01 │ 3.29e+12*cos(2π*1.61e-02*t-6.15e-01) │
│ 2.612e-02 │    100.00 │ 2.228e+12 │ -6.164e-01 │ 2.23e+12*cos(2π*2.61e-02*t-6.16e-01) │
│ 8.095e-03 │    100.00 │ 1.843e+12 │  5.008e+00 │ 1.84e+12*cos(2π*8.09e-03*t+5.01e+00) │
│ 9.410e-02 │    100.00 │ 1.438e+12 │ -1.603e+00 │ 1.44e+12*cos(2π*9.41e-02*t-1.60e+00) │
│ 5.821e-02 │    100.00 │ 3.312e+12 │  4.181e+00 │ 3.31e+12*cos(2π*5.82e-02*t+4.18e+00) │
└───────────┴───────────┴───────────┴────────────┴──────────────────────────────────────┘
Carterization up to 10 frequencies: 
7.65e+12*cos(2π*1.80e-03*t+4.87e+00) + 7.27e+12*cos(2π*5.23e-02*t+2.29e+00) + 3.12e+12*cos(2π*4.29e-02*t-1.91e+00) + 
8.97e+12*cos(2π*4.89e-02*t+1.14e+00) + 3.67e+12*cos(2π*6.63e-03*t-1.52e+00) + 3.29e+12*cos(2π*1.61e-02*t-6.15e-01) + 
2.23e+12*cos(2π*2.61e-02*t-6.16e-01) + 1.84e+12*cos(2π*8.09e-03*t+5.01e+00) + 1.44e+12*cos(2π*9.41e-02*t-1.60e+00) + 
3.31e+12*cos(2π*5.82e-02*t+4.18e+00)

Total elapsed time: 4.820 s
```
Note that the table provided by FTIO shows the top 10 frequencies from Fourier fitting. To get the once from DFT, simply remove the `--fourier_fit` flags (not in the Paper).

To create the figure 2 from the paper, add the `-e` flag with matplotlib:
```bash
ftio Job47970840_9216.msgpack -f 1 -n 10 --fourier_fit  -e mat -v  
```

<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>

### 3. Combining DWT with DFT
TO demonstrate the advantage of combining DWT with DFT, we conducted two experiments in the paper "Improving I/O Phase Predictions in FTIO Using Hybrid Wavelet-Fourier Analysis" based on: 
1. [HACC-IO](#dwt-x-dtw-hacc-io) with 96 nodes and 96 ranks each 
2. [DLIO](#dwt-x-dtw-dlio) on two nodes with 96 ranks each  
The two experiments are described in the next two sections.

#### DWT X DTW: HACC-IO
In this example, the same trace as in [Fourier Fitting](#2-fourier-fitting) is used. To get the trace, either 
[extract the traces](#extracting-the-data-set) used in the paper or [recreate it](#hacc-io). With `Job47970840_9216.msgpack` 
at hand, execute:
```bash
time ftio Job47970840_9216.msgpack  -tr "wave_disc" -le 2 -e no -v  
```
This prints the following to the console:
```bash
❯ time ftio Job47970840_9216.msgpack  -tr "wave_disc" -le 2 -e no 
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                              Ftio                                                              │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Author:  Ahmad Tarraf
Date:    2025-11-25
Version:  0.0.7
License:  BSD

Current file: Job47970840_9216.msgpack

Correlation between 0.12 and 1.0 from 9.88s to 72.08s
Correlation between 0.12 and 1.0 from 106.38s to 168.28s
Prediction results:
Frequency: 5.552e-02 Hz-> 18.01 s
Confidence: 69.93 %
Valid time segments:
- [9.88,72.08] sec
- [106.38,168.28] sec
```
For the paper, we used the `time` call to get the execution time. 
To generate the figure 3, we execute this call with the `-e mat` argument:
```bash
ftio Job47970840_9216.msgpack  -tr "wave_disc" -le 2 -e mat 
```

#### DWT X DTW: DLIO
For this experiment, we used the Metric Proxy to gather the traces rather than TMIO. To get the trace, either extract 
in from [extract the traces](#extracting-the-data-set), which is the traces used in the paper, or [recreate it](#dlio). 
In either case, the starting point in a directory containing a folder named `traces`, hosting the DLIO job, which is 
`48116802-0.trace` in our case.  First start the proxy in the current folder to access the data through the HTTP endpoints:
```bash
proxy_v2 -r http://localhost:1337 -t . -S 100
```
In a different terminal, execute the following command to run ftio on the trace:
```bash
proxy_ftio --proxy -j 48116802-0 -m "total___strace___size___write" -e mat -f 10 -tr "wave_disc" -le 2 
```
This call will generate figure 4 from the paper. In particular, in runs FTIO on the job name `48116802`, extracting the 
metric write in bytes captured using strace, and using the wavelet transform with 2 levels on the signal sampled with 
a frequency of 10 Hz. 
To measure the runtime, execute 
```bash
time proxy_ftio --proxy -j 48116802-0 -m "total___strace___size___write" -e no -f 10 -tr "wave_disc" -le 2 
```

## Traces
In this section, we describe how to recreate the traces from the Paper. This includes [HACC-IO](#hacc-io) and [DLIO](#dlio).

### HACC-IO
The traces from the paper are provided in the dataset (see [Extracting the Data Set](#extracting-the-data-set)). In case the users are interested in reproducing the traces, first HACC-IO needs to be compiled with TMIO. Hence, the following steps are required: 
#### Download and Extract HACC-IO 
Download and extract the repository of the modified HACC-IO version from [here](https://github.com/A-Tarraf/hacc-io).
#### Download and install TMIO:
The next step is to download and install TMIO and make sure that it works. For that, execute the following steps:
```bash
git clone https://github.com/tuda-hpclab/tmio.git
git checkout v0.0.2
cd TMIO/build
make msgpack_library 
```
#### Compile HACC-IO with TMIO
Navigate to the HACC-IO directory and compile the application with TMIO:
`cd your_hacc_io_directory`
Next, modify the Makefile to include the TMIO directory: replace `TMIO_DIR = /d/github/TMIO` with `TMIO_DIR = /path/to/TMIO`.
The HACC-IO version from the GItHub repository uses Async I/O. To use Sync I/O, the changes below need to be made to `testHACC_Async_IO.cxx`. Also make sure that the correct modes are set:
```c++
//? set interface for I/O (read & write)
//? ******************************************
//rst->Set_POSIX_IO_Interface();
rst->Set_Sync_MPI_IO_Interface();
//rst->Set_Async_MPI_IO_Interface();

//...

//? set Header mode:
rst->Set_Sync_MPI_IO_Header();
//rst->Set_Async_MPI_IO_Header();

//? set File mode
//? ******************************************
rst->SetMPIIOSharedFilePointer();
// rst->SetMPIOIndepFilePointer();
rst->SetFileDistribution(GLEAN_SINGLE_FILE);
// rst->SetFileDistribution(GLEAN_FILE_PER_RANK);
```
Finally, compile HACC-IO with TMIO:
```bash
make run_msgpack
```
This should create the executable `HACC_ASYNC_IO` and run the application with 8 ranks locally in a test directory.
Next, the sbatch script `sbatch.sh` can be used to run the application on Lichtenberg cluster:
```bash
#!/bin/bash
#SBATCH -J HACC_ASYNC_IO
#SBATCH -e ./%x.err
#SBATCH -o ./%x.out

## uses LB 2 phase I:
#SBATCH -C i01
#SBATCH -n 384 #384 #9216
#SBATCH -c 1
#SBATCH --mem-per-cpu=3800 
#SBATCH -t 00:15:00      
#SBATCH -A projectXXXX

module pure
module load gcc/11 openmpi python cuda cmake hdf5 
mkdir -p /work/scratch/user_name/test

# actual run
time srun ./HACC_ASYNC_IO 5000000 /work/scratch/user_name/test/mpi

# Ouput log
EXITSTATUS=$?
echo “Job $SLURM_JOB_ID has finished at $(date).”
exit $EXITSTATUS
```
The above sbatch script can be used to run the HACC-IO application on Lichtenberg with 384 ranks. The output will be stored in the directory `/work/scratch/user_name/test/mpi`. 
In the paper, another experiment was executed with 9216 ranks (section 6.2). 
To create the trace, simply replace `#SBATCH -n 384` with `#SBATCH -n 9216`.  
Finally, submit the sbatch script via `sbatch sbatch.sh`
After the job finishes, the trace can be found in the current directory. It names should reflect the number of ranks used (e.g., `384.msgpack`). This trace can be simply used with `ftio` as described under [Signal Reconstruction with Multiple Frequencies](#1-signal-reconstruction-with-multiple-frequencies) or [Fourier Fitting](#2-fourier-fitting).

<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>

### DLIO
To run the evaluation with `ftio`, we first need to gather the trace. Unlike the case with [HACC-IO](#hacc-io), the experiments
in this part used the metric proxy. In the following, we describe how to [set it up] and [run the experiments](#run-the-experiments).

#### Install the Metric Proxy
Download the [metric proxy](https://github.com/A-Tarraf/metric-proxy) and follow the instructions to install it. In short, 
you should execute these steps:
```bash
git clone https://github.com/A-Tarraf/proxy_v2.git
cd proxy_v2
# For our case, we installed the proxy in a tool directory:
./install.sh /work/projects/projectXXX/tools
export TOOLS_BIN=/work/projects/projectXXXX/tools/bin
export PATH=$TOOLS_BIN:$PATH 
# Alternatively, you can install it in your home directory:
#./install.sh $HOME/metric-proxy
# export PATH=$HOME/metric-proxy/bin:$PATH
```
#### Install DLIO
Follow the instructions to [install DLIO](https://github.com/argonne-lcf/dlio_benchmark). In short, execute the following steps:
```
git clone https://github.com/argonne-lcf/dlio_benchmark
cd dlio_benchmark/
pip install -e .
```
Next, since we installed DLIO with the `-e` flag, we can simply place a custom workload script in the `workloads` directory.
```
cd ./dlio_benchmark/configs/workload
touch resnet50_my_a100_pytorch.yaml
```
Add the following content to `resnet50_my_a100_pytorch.yaml`
```yaml
model:
  name: resnet50
  type: cnn
  model_size: 499153191
  num_layers: 50
  model_datatype: fp16
  optimizer_datatype: fp32
  layer_parameters: [1024, 2048]
  parallelism:
    pipeline: 1
    tensor: 1
    zero_stage: 3

framework: pytorch

workflow:
  generate_data: True
  train: True
  checkpoint: True

dataset:
  num_files_train: 1024
  num_samples_per_file: 100 #1251
  record_length_bytes: 114660.07
  record_length_bytes_resize: 150528
  data_folder: data/resnet50
  format: npz #png #csv #npz

train:
  computation_time: 0.435
  epochs: 10

reader:
  data_loader: pytorch
  read_threads: 8
  computation_threads: 8
  batch_size: 400
  dont_use_mmap: True

checkpoint:
  checkpoint_folder: checkpoints/resnet50_my_a100_pytorch
  checkpoint_after_epoch: 1
  epochs_between_checkpoints: 1
  type: all_ranks

metric:
  au: 0.90
```

#### Run the Experiments
With the proxy set up, the experiments can be executed. The first steps involves running the root proxy at the login nodes:
```bash
cd ~/projectXXXX/results/dlio_new 
# export the variables 
export SRUN=/opt/slurm/current/bin/srun
export TOOLS_BIN=/work/projects/projectXXXX/tools/bin
export TOOLS_LIB=/work/projects/projectXXXX/tools/lib
export PATH=$TOOLS_BIN:$PATH
export LD_LIBRARY_PATH=$TOOLS_LIB:$LD_LIBRARY_PATH

# Finally start the root proxy on the login node
proxy_v2 -t . -S 100  -m 128
```
Since we do not need to collect the metrics before the accrual application was executed, we waited until the job in the 
next step started before starting the root proxy. 

In a different terminal, we submitted the following sbatch script:
```bash
#!/bin/bash
#SBATCH -J dlio_benchmark
#SBATCH --mail-type=NONE   #BEGIN, END, ALL, or NONE
#SBATCH -e ./%x.err
#SBATCH -o ./%x.out
## uses LB 2 phase I:
#SBATCH -C i01
#SBATCH -n 96 
#SBATCH -c 2
#SBATCH --mem-per-cpu=3800  
# -------------------------------
module purge
module load gcc/11 openmpi python cuda cmake hdf5 

# Select workload
export WORKLOAD=resnet50_large_a100_pytorch
export WORKLOAD=resnet50_my_a100_pytorch
export ROOT_PROXY=logc0002 #name of node where the root proxy is running

#### proxy application side
export DFTRACER_INC_METADATA=0
export DFTRACER_ENABLE=0
#export DLIO_LOG_LEVEL="warning"
export DLIO_LOG_LEVEL="info"
export PATH=$TOOLS_BIN:$PATH
export LD_LIBRARY_PATH=$TOOLS_LIB:$LD_LIBRARY_PATH
export SRUN=/opt/slurm/current/bin/srun
# Total CPUs per task/node as specified via `-c`
TOTAL_CPUS=${SLURM_CPUS_PER_TASK:-3}
CPUS_PROXY=1
CPUS_REMAINING=$((TOTAL_CPUS - CPUS_PROXY))
$SRUN --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} --ntasks-per-node=1 --cpus-per-task=${CPUS_PROXY} --overlap proxy_v2 -i -r http://${ROOT_PROXY}:1337 &
$SRUN --cpus-per-task=${CPUS_REMAINING} proxy_run -- dlio_benchmark workload=$WORKLOAD ++workload.workflow.generate_data=True ++workload.workflow.train=True ++workload.workflow.checkpoint=True

EXITCODE=$?
exit $EXITCODE
```

While the job is running, the results can be examined online in the root proxy. Furthermore, with the latest Proxy version
(under development branch), FTIO can also be applied online. However, for this paper, we skipped both aspects. After the 
job completes, the results (traces and profiles) can be found in the directory where the root proxy was started.
<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>


## Citation
The paper citation is available [here](/README.md#citation). You can cite the [data set](https://doi.org/10.5281/zenodo.10670270) as:
```
@dataset{tarraf_2025_17713783,
  author       = {Tarraf, Ahmad and
                  Wolf, Felix},
  title        = {Improving I/O Phase Predictions in FTIO Using
                   Hybrid Wavelet-Fourier Analysis [Data Set]
                  },
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17713783},
  url          = {https://doi.org/10.5281/zenodo.17713783},
}

```

