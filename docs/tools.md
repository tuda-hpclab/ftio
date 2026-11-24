# Tools
Aside from [`ftio`](https://github.com/tuda-hpclab/ftio#usage) or [`predictor`](https://github.com/tuda-hpclab/ftio/blob/main/docs/approach.md#online-prediction), the FTIO repository offers several more tools for various purposes: 

- [`ioplot`](#ioplot) generates interactive plots in HTML
- [`ioparse`](#parse) parses and merges several traces to an [Extra-P](https://github.com/extra-p/extrap)-supported file format
- [`convert_trace`](#convert_trace) converts an old [TMIO](https://github.com/tuda-hpclab/tmio) trace to the latest version

The tools are automatically generated when installing FTIO, for example, by calling `make install` as described [here](https://github.com/tuda-hpclab/ftio#installation)

## ioplot
This tool generates interactive plots using Plotly. Supported file formats are described [here](./file_formats.md).
```bash
usage: ioplot [-h] [-m MODE] [-s SOURCE] [-r RENDER] [-z ZOOM]
              [-nt] [-e ENGINE]
              [--n_shown_samples N_SHOWN_SAMPLES] [--merge_plots]
              [--no_disp] [--sum] [--no_sum] [--avr] [--no_avr]
              [--ind] [--no_ind] [-cf CUSTOM_FILE] [-x DXT_MODE]
              [-l LIMIT]
              files [files ...]
```

Example plotting a single file:
```bash 
ioplot 1536.json
```

Example for plotting an entire folder with several json or jsonl files:
```bash
ioplot .
# or 
ioplot *.json
```

## ioparse
Parses and merges several traces generated with [FTIO](https://github.com/tuda-hpclab/ftio) or [TMIO](https://github.com/tuda-hpclab/tmio) to an [Extra-P](https://github.com/extra-p/extrap) supported file format. Supported file formats are described [here](./file_formats.md). Usage:
```bash
usage: ioparse [-h] [-m MODE] [-s SOURCE] [--scale] [--sum]
               [--no_sum] [--avr] [--no_avr] [--ind] [--no_ind]
               [-cf CUSTOM_FILE] [-x DXT_MODE] [-l LIMIT]
               files [files ...]
```

Example:
```bash
ioparse .
```


## convert_trace
Starting with version v.0.0.2, TMIO changed the default units to bytes rather than MB in the json and jsonl files. With this script, old traces obtained with TMIO v0.0.1 can be updated to support the latest versions of the other tools here. Usage:
```bash
usage: convert_trace [-h] [--outfile [OUTFILE]] filename
```

Example: 

```bash 
convert_trace 9216.json
#or explicit specify the output
convert_trace 9216.json -o file_name.json
```


## parallel_trace_analysis
Kindly see this link for more details:
https://github.com/tuda-hpclab/ftio/tree/main/artifacts/ipdps25

Example:
``` bash
parallel_trace_analysis  . -j -p 30 -f 10 -o ~/tmp -n name
```