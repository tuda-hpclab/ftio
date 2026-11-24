# Artifacts Reproducibility

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10670270.svg)](https://doi.org/10.5281/zenodo.10670270)


Below, we describe how to reproduce the experiments in the Paper entitled:
"Capturing Periodic I/O Using Frequency Techniques," which was published at the IPDPS 2024

Before you start, first set up the correct [FTIO version](#ftio-version).
The experiments are divided into four parts:

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
2. Depending on what you want to test, you need to [download and extract](#extracting-the-data-set) the data set from [Zenodo](https://doi.org/10.5281/zenodo.10670270).

### FTIO Version

For all the cases below, `ftio` first needs to be installed (see [Installation](https://github.com/tuda-hpclab/ftio?tab=readme-ov-file#installation)). We used `ftio` version 0.0.1 for the experiment in the paper. To get this version, simply execute the following code:
```sh
git checkout v0.0.1 
```

### Extracting the Data Set:
download the zip file from [here](https://doi.org/10.5281/zenodo.10670270) or using wget in a bash terminal:
```sh
wget https://zenodo.org/records/10670271/files/data.zip?download=1
```
Next, unzip the file
```sh
unzip data.zip
```
This extracts the needed traces and experiments:

```sh
data
├── application_traces
│   ├── HACC-IO
│   ├── IOR
│   ├── LAMMPS
│   ├── NEK5000
│   └── README.md
├── exps_with_synthetic_traces
├── iosets-ftio-experiments
└── README.md
```

## Artifacts

### 1. Running Example and Overhead
Throughout Section II of the paper "Capturing Periodic I/O Using Frequency Techniques", IOR was used to daemonstrate the approach. Moreover, In Section III-C the overhead of our approach was examined using IOR with a varying number of ranks (from 1 to 10752 ranks).

- Follow the instructions provided in the [IOR/README.md](/artifacts/ipdps24/IOR/README.md) to set up and run IOR or execute `ftio` on the provided traces.  
- Instructions regarding the overhead are provided [here](/artifacts/ipdps24/IOR/README.md#tracing-library-overhead).

<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>

### 2. Case Studies
Bellow instructions are provided on how to recreate the case studies from Section III-B. As mentioned [above](#ftio-version), `ftio` version 0.0.1 was used for all examples here. 

#### 2.1. LAMMPS

This experiment was executed on the Lichtenberg cluster with 3072 ranks. 
Follow the instructions provided in the [LAMMPS/README.md](/artifacts/ipdps24/LAMMPS/README.md) to reconstruct this experiment. 

<!-- The provided [tar archive](/LAMMPS/lammps.tar.gz) contains not only the result from our -->
<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>


#### 2.2 Nek5000
The trace can be downloaded from [here](https://hpcioanalysis.zdv.uni-mainz.de/trace/64ed13e0f9a07cf8244e45cc).
After downloading, instructions on how to reproduce the results are provided in the [NEK5000/README](/artifacts/ipdps24/NEK5000/README.md).

<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>

#### 2.3 Modified HACC-IO
Navigate to [HACC-IO/README](/artifacts/ipdps24/HACC-IO/README.md) and follow the provided instructions. 


<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>

### 3. Limitations of FTIO

The details for this experiment are provided here: <https://gitlab.inria.fr/hpc_io/ftio_paper_exps_with_synthetic_traces>
<br> 
Alternatively, the details are also provided in the README.md file in the folder `exps_with_synthetic_traces` from the [Zenodo](https://doi.org/10.5281/zenodo.10670270) dataset.

<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>


### 4. Use Case: I/O Scheduling
The details for this experiment are provided here: <https://gitlab.inria.fr/hpc_io/iosets-ftio-experiments>
<br> 
Alternatively, the details are also provided in the README.md file in the folder `iosets_ftio_experiments` from the [Zenodo](https://doi.org/10.5281/zenodo.10670270) dataset.

<p align="right"><a href="#artifacts-reproducibility">⬆</a></p>


## Citation
The paper citation is available [here](/README.md#citation). You can cite the [data set](https://doi.org/10.5281/zenodo.10670270) as:
```
 @dataset{tarraf_2024_10670270,
  author       = {Tarraf, Ahmad and
                  Bandet, Alexis and
                  Boito, Francieli and
                  Pallez, Guillaume and
                  Wolf, Felix},
  title        = {Capturing Periodic I/O Using Frequency Techniques},
  month        = Feb,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10670270},
  url          = {https://doi.org/10.5281/zenodo.10670270}
}
```

