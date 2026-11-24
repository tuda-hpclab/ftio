# IOR

Either [extract](#extract-the-provided-trace-file) or [generate](#generate-the-experiment-on-your-system) the trace files as described below. 
Afterward, [run ftio](#run-ftio) on the trace file.

## Extract the provided trace files
Download and extract the file data.zip as described [here](/artifacts/ipdps24/README.md#extracting-the-data-set).
Once extracted, you can find the IOR trace under `data/application_traces/IOR`.

<p align="right"><a href="#ior">⬆</a></p>

## Generate the experiment on your system:

Install IOR as described [here](https://ior.readthedocs.io/en/latest/userDoc/install.html)

For our experiments on the Lichtenberg cluster, we used the following sbatch script (`sbatch.s`):
```sh
#!/bin/bash
#SBATCH -J ior
##SBATCH --mail-type=END
#SBATCH -e %x.err
#SBATCH -o %x.out
#SBATCH -n 9216
#SBATCH --mem-per-cpu=3800   
#SBATCH -t 00:10:00
#SBATCH -t 00:30:00

export LD_PRELOAD=./libtmio.so  
srun ./ior  -N ${SLURM_NPROCS} -t 2m -b 10m -s 2 -i 8 -a MPIIO
```
Modify this script if needed according to your system specifications.
Next simply submit the sbatch script via:
```sh
sbatch sbatch.sh
```

Once the simulation is completed, the file `9216.json` should be generated in the current folder.
Next, run `ftio` on the trace file as described [here](#run-ftio).

<p align="right"><a href="#ior">⬆</a></p>



## Run FTIO
If you didn't install `ftio`, first check out version 0.0.1 as described [here](/artifacts/ipdps24/README.md#ftio-version), then install it as described [here](https://github.com/tuda-hpclab/ftio#installation).
The instructions below assume that the trace file is called `9216.json`, which after [extracting](#extract-the-provided-trace-file) the trace file, should be located in `data/application_traces/IOR`.

To get the dominant frequency with `ftio` simply call:
```sh
ftio 9216.json  -e no  
```
Remove `-e no` if you want to obtain figures.

Now lower the threshold from 0.8 (default) to 0.45:
```sh
ftio 9216.json  -e no  -t 0.45
```

To gain further confidence, execute ftio with autocorrelation (`-c`):
```sh
ftio 9216.json  -e no  -t 0.45 -c 
```

<p align="right"><a href="#ior">⬆</a></p>


## Tracing Library Overhead

### Detection Mode:
Download and extract the file data.zip as described [here](/artifacts/ipdps24/README.md#extracting-the-data-set).

Navigate to `data/application_traces/IOR/overhead/detection` and call [`ioplot`](https://github.com/tuda-hpclab/ftio/blob/main/docs/tools.md#ioplot):

```sh
cd data/application_traces/IOR/overhead/detection
ioplot 
```

Now examine the plots in your browser by selecting "Time". If your browser does not open automatically, open the `file io_results/time.html` or `file io_results/main.html` and select time.

<p align="right"><a href="#ior">⬆</a></p>

### Prediction Mode:

Download and extract the file data.zip

Navigate to `data/application_traces/IOR/overhead/prediction` and call [`ioplot`](https://github.com/tuda-hpclab/ftio/blob/main/docs/tools.md#ioplot):

```sh
cd data/application_traces/IOR/overhead/detection
ioplot run_*
```
Now examine the plots in your browser by selecting "Time". If your browser does not open automatically, open the `file io_results/time.html` or `file io_results/main.html` and select time.

<p align="right"><a href="#ior">⬆</a></p>
