(multi-node)=

# Multi-Node

Lightly**Train** supports multi-node training. This means that you can train your model on multiple machines at the same time. This can be useful if you have a large dataset and want to speed up the training process.

Lightly**Train** builds upon PyTorch Lightning and thus supports the same multi-node training features.

```{note}
Multi-node training is an advanced topic. You should be familiar with multi-node training before using it with Lightly**Train**.
```

## Choosing the Multi-Node Training Approach

There are different ways to start a multi-node training job:

1. Using a [general-purpose cluster](https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_1.html). If you are new to multi-node training, we recommend this method as the easiest way to get started.
1. Using [TorchRun (TorchElastic)](https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_2.html).
1. Using a [Slurm-managed cluster](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html).
1. Multi-node setups provided by your infrastructure provider.

## Setting Up the Multi-Node Training Environment

In all cases, you need to set up the multi-node training environment first. This means:

- Choose one of the supported multi-node training methods and set it up by following its documentation.
- Ensure that all nodes have access to the same network. This is necessary for the nodes to communicate with each other. You can test this by pinging the IP addresses and ports of the other nodes from each node.
- Ensure that all nodes use the same software stack with exactly the same versions: NVIDIA drivers, CUDA, cuDNN, Python, PyTorch, PyTorch Lightning, Lightly**Train**, etc.
- Ensure that the dataset defined by the `data` argument is available on all nodes. This can be done by using a shared file system or by copying the dataset to all nodes. Note that the dataset must be exactly the same.
- When using Lightly**Train** inside a Docker container, make sure that port forwarding between the container and host is set up correctly.

## Starting the Multi-Node Training Job with Lightly**Train**

Once you have set up the multi-node training environment, you can easily use Lightly**Train** for multi-node training. Just call Lightly**Train** on each node the same way you call other multi-node training jobs. The only difference is that you need to set the `num_nodes` argument to the number of nodes you want to use.

See the examples below for how to start a multi-node training job with Lightly**Train** when using a general-purpose cluster.

### General-Purpose Cluster

If you are using a general-purpose cluster, you can start a multi-node training job as shown below.
Set the `WORLD_SIZE=2` environment variable and the `num_nodes=2` argument to train on two nodes.

````{tab} Python
Create a Python script that calls `lightly_train.train()` and add `num_nodes=2` to the
arguments:

```python
# my_train_script.py
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        num_nodes=2,
    )
```

Then call your Python script on each node from the CLI as follows:
```bash
# On node 1
MASTER_PORT=50027 MASTER_ADDR=123.45.67.89 WORLD_SIZE=2 NODE_RANK=0 python my_train_script.py
# On node 2
MASTER_PORT=50027 MASTER_ADDR=123.45.67.89 WORLD_SIZE=2 NODE_RANK=1 python my_train_script.py
```
````

````{tab} Command Line
```bash
# On node 1
MASTER_PORT=50027 MASTER_ADDR=123.45.67.89 WORLD_SIZE=2 NODE_RANK=0 lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" num_nodes=2
# On node 2
MASTER_PORT=50027 MASTER_ADDR=123.45.67.89 WORLD_SIZE=2 NODE_RANK=1 lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" num_nodes=2
```
````
