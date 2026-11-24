# 🌐 FTIO HTTP Server Interface

This document describes the FTIO HTTP server which wraps the FTIO CLI functionality to accept POST requests and return
JSON results. It supports all FTIO CLI arguments.

## 🚀 Starting the Server

You can start the server with default settings or specify a custom port and host address.
Note that this call is only avilable if you installed FTIO
for [development]https://github.com/tuda-hpclab/ftio/tree/development?tab=readme-ov-file#automated-installation-developer-environment-setup).
Afterwards, you can execute the command:

```bash
server_ftio
```

---

## Usage

Running `server_ftio` starts the server by running the `main_cli()`
function [server_ftio.py](https://github.com/tuda-hpclab/ftio/blob/development/ftio/util/server_ftio.py).
By default, it listens on `127.0.0.1` and port `5000`, but you can specify custom address and port via environment
variables or command-line
options.

```bash
server_ftio --port 8080 --host 0.0.0.0

server_ftio -h
usage: server_ftio [-h] [--port PORT] [--host HOST]

FTIO HTTP Server

options:
  -h, --help   show this help message and exit
  --port PORT  Port to run the server on
  --host HOST  Host address to bind to
```

Once running, you can send requests using `curl`. Send a POST request with your FTIO arguments in the body using `curl`:

```bash
curl -X POST http://localhost:5000/ftio --data "--file data.txt --mode auto"
```

The server response in JSON. For instance send a POST request for working on the file `Job47969634_1536.msgpack`:

```bash
curl -X POST http://localhost:5000/ftio --data "Job47969634_1536.msgpack -f 10 -e none"

# The server response with
{
"amp":[13901960943653.285],"candidates":[],"conf":[1.0],
"dominant_freq":[0.13986013986013987],"freq":10.0,"n_samples":572,
"phi":[-0.594723495382335],"ranks":1536,"source":"dft",
"t_end":59.535650343,"t_start":2.239342796,"top_freqs":{},
"total_bytes":2334921326592
}
```

