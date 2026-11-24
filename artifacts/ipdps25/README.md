# Artifacts Reproducibility

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14965920.svg)](https://doi.org/10.5281/zenodo.14965920)


Below, we describe how to reproduce the FTIO experiments related to periodic behavior from the Paper entitled:
"A Deep Look Into the Temporal I/O Behavior of HPC Applications," which was published at the IPDPS 2025

Before you start, first set up [FTIO](#setup).
The experiments are divided into four parts:



## Prerequisites 
Before you start, there are two prerequisites:
1. Install [FTIO](#setup) 
2. Depending on what you want to test, you need to [download and extract](#extracting-the-data-set) the data set from [Zenodo](https://doi.org/10.5281/zenodo.14965920).

### Setup

We used `ftio` version 0.0.4 for the experiment in the paper. To get this version, simply execute the following code:
```sh
git checkout v0.0.4
```
Afterward, you can install FTIO on your system (see [Installation](https://github.com/tuda-hpclab/ftio?tab=readme-ov-file#installation)). There are no issues or compatibility problems with newer FTIO versions. You can also simply use the latest FTIO version.

For the experiments, we used the parallel trace analysis in FTIO. After installation, you can check that it works with the following command:

```bash

# After installation this should work
parallel_trace_analysis   -h 
# or execute the python script manually
<FTIO_repo>/ftio/api/trace_analysis/parallel_trace_analysis.py
```
This displays the following message
```bash
Usage:  parallel_trace_analysis  <dir>

-n <str>: filter according if they contain the name
--time-step <float>: specifies the value of the implicit time steps between the samples
-o <str>: Output dir location
-j <bool>: Enables JSON search
-v <bool>: verbose
-s <bool>: save the FTIO result for each file in a file that contains the name _freq_
All ftio options (see ftio -h)
```


### Extracting the Data Set:
download the zip file from [here](https://doi.org/10.5281/zenodo.14965920) or using wget in a bash terminal:
```sh
wget https://zenodo.org/records/14965920/files/data.zip?download=1
```
Next, unzip the file
```sh
unzip data.zip
```
This extracts the needed traces and experiments:

```sh
data
├── clusterwise-master.tar.gz
├── data.tar.xz
├── plafrim_signal.tar.gz
└── sdumont_signal_1.tar.gz
```

## Artifacts
Bellow we describe the procedure to recreate the results from Section IV 
"Are Applications Periodic?" from the paper "A Deep Look Into the Temporal I/O Behavior of HPC Applications"

### Sdumont
extract `sdumont_signal_1.tar.gz` from the provided [Zenodo dataset](#extracting-the-data-set) and navigate to the extracted folder:
```bash
# Extract
tar -xf sdumont_signal_1.tar.gz
# Navigate to the folder 
cd sdumont_signal_1/sdumont
```
Now simply execute the parallel trace analysis script with the desired processes. We used 30 processes for the analysis:

```bash
parallel_trace_analysis  . -n sdumont --time-step 15 -p 30
```
The `--time-step 15` is required as it indicates the time step between the entries in the CSV file. 

### PLatfrim
extract `plafrim_signal.tar.gz` from the provided [Zenodo dataset](#extracting-the-data-set) and navigate to the extracted folder:
```bash
# Extract
tar -xf plafrim_signal.tar.gz
# Navigate to the folder 
cd plafrim_signal/plafrim
```

Now simply execute the parallel trace analysis script with the desired processes. We used 30 processes for the analysis alongside the default settings:

```bash
parallel_trace_analysis  . -p 30
```

### For Intrepid
You can also pass ftio arguments to the call. Bellow for example, we set the 
sampling frequency to 10 (default) using `-f 10`. Other arguments like the location
of the output results can also be provided with `-o <path>`:
```bash
parallel_trace_analysis  . -j -p 30 -f 10 -o ~/tmp -n intrepid
```


### For Blue waters
You can acquire the data from: [https://bluewaters.ncsa.illinois.edu/data-sets](https://bluewaters.ncsa.illinois.edu/data-sets). Afterward simply executed: 
```bash
parallel_trace_analysis  . -j -p 30 -f 10 -n blue_waters
```



You can cite the [data set](https://doi.org/10.5281/zenodo.14965920) as:
## Citation
```
 @dataset{boito_2025_14965920,
  author       = {Boito, Francieli Zanon and
                  Teylo, Luan and
                  Popov, Mihail and
                  Jolivel, Theo and
                  Tessier, François and
                  Luettgau, Jakob and
                  Monniot, Julien and
                  Tarraf, Ahmad and
                  Carneiro, André Ramos and
                  Osthoff, Carla},
  title        = {A Deep Look Into the Temporal I/O Behavior of HPC
                   Applications [Dataset]
                  },
  month        = mar,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14965920},
  url          = {https://doi.org/10.5281/zenodo.14965920},
}
```

