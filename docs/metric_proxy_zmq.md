## FTIO ZMQ Server for Metric Proxy
This document describes the custom FTIO zmq server which the [Metric Proxy](https://github.com/A-Tarraf/proxy_v2/tree/main) utilizes to get ftio predictions.
### Functionality
The FTIO zmq server provides a communication interface for the Metric Proxy, handling requests for FTIO predictions and returning results over ZeroMQ. It deserializes incoming messages, executes prediction tasks (parallel or sequential), returns the serialized results to Metric Proxy, responds to control messages like "ping" and "New Address" and shuts down automatically after being idle for too long.
### Call Tree
This is the immediate call tree of proxy_zmq.py within FTIO.
```
ftio/api/metric_proxy/proxy_zmq.py::main()
└── proxy_zmq.py::handle_request()
    ├── metric_proxy/parse_proxy.py::filter_metrics()
    └── metric_proxy/parallel_proxy.py::execute_parallel() # (or execute() if disable_parallel is true)
        └── ftio/prediction/tasks.py::ftio_metric_task_save()
            └── tasks.py::ftio_metric_task()
                ├── ftio/parse/args.py::parse_args()
                └── ftio/cli/ftio_core.py::core()
```

### Limitations
[ftio_metric_task_save()](https://github.com/tuda-hpclab/ftio/blob/development/ftio/prediction/tasks.py) uses a dictionary to store results so that MessagePack can serialize the data and send it to Metric Proxy where MessagePack can then deserialize it with little maintenance required. If a new class such as [Prediction](https://github.com/tuda-hpclab/ftio/blob/development/ftio/freq/prediction.py) would be used instead, MessagePack would require custom implementations for serialization and deserialization for both the FTIO and the Metric Proxy side. Changes to Prediction would then require both custom implementations to be updated and maintained as well.