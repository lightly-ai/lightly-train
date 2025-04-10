(multi-gpu)=

# Multi-GPU

Set the `accelerator` and `devices` arguments to train on a single machine (node) with
multiple GPUs. By default, **Lightly Train** uses all available GPUs on the current
node for training. The following example shows how to train with two GPUs:

````{tab} Python

```{important}
Always run your code inside an `if __name__ == "__main__":` block when using multiple
GPUs!
```

```python
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        accelerator="gpu",      # Accelerator type
        devices=2,              # Number of GPUs
    )
```
````

````{tab} Command Line
```bash
lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50" accelerator="gpu" devices=2
```
````

```{tip}
Set `devices=[1, 3]` to train on GPUs 1 and 3 specifically. When using the command line
interface, set `devices="[1,3]"` instead.
```

(multi-gpu-adjusting-parameters)=

## Adjusting Parameters

Parameters such as the batch size (`batch_size`), learning rate (`optim_args.lr`), and
the number of dataloader workers (`num_workers`) are automatically adjusted
based on the number of GPUs. You do not need to modify these parameters manually when
changing the number of GPUs.

```{tip}
The batch size (`batch_size`) is the global batch size across all GPUs. Setting
`batch_size=256` with `devices=2` will result in a batch size of 128 per GPU.
```

```{tip}
The number of workers (`num_workers`) is the number of dataloader workers per GPU.
Setting `num_workers=8` with `devices=2` results in 16 dataloader workers in total.
The total number of dataloader workers should not exceed the number of CPU cores on the
node to avoid training slowdowns.
```

## SLURM

Use the following setup to train on a SLURM-managed cluster with multiple GPUs:

````{tab} Python
Create a SLURM script (`my_train_slurm.sh`) that looks as follows:

```bash
#!/bin/bash -l

#SBATCH --nodes=1               # Number of nodes
#SBATCH --gres=gpu:2            # Number of GPUs
#SBATCH --ntasks-per-node=2     # Must match the number of GPUs
#SBATCH --cpus-per-task=12      # Number of CPU cores per GPU; must be larger than the 
                                # number of dataloader workers (num_workers) if num_workers 
                                # is set in the training function. Otherwise, num_workers is 
                                # automatically set to cpus-per-task - 1.
#SBATCH --mem=0                 # Use all available memory

# IMPORTANT: Do not set --ntasks as it is automatically inferred from --ntasks-per-node.

# Activate your virtual environment.
# The command might differ depending on your setup.
# For conda environments, use `conda activate my_env`.
source .venv/bin/activate

# On your cluster you might need to set the network interface:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Might need to load the latest CUDA version:
# module load NCCL/2.4.7-1-cuda.10.0

# Run the training script.
srun python my_train_script.py      
```

Then create a Python script (`my_train_script.py`) that calls `lightly_train.train()`:

```python
# my_train_script.py
import lightly_train

if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment",
        data="my_data_dir",
        model="torchvision/resnet50",
        # The following arguments are automatically set based on the SLURM configuration:
        # accelerator="gpu",
        # devices=2,
        # num_workers=11,
    )
```

Finally, submit the training job to the SLURM cluster with:
```bash
sbatch my_train_slurm.sh
```
````

````{tab} Command Line
Create a SLURM script (`my_train_slurm.sh`) that looks as follows:

```bash
#!/bin/bash -l

#SBATCH --nodes=1               # Number of nodes
#SBATCH --gres=gpu:2            # Number of GPUs
#SBATCH --ntasks-per-node=2     # Must match the number of GPUs
#SBATCH --cpus-per-task=12      # Number of CPU cores per GPU; must be larger than the 
                                # number of dataloader workers (num_workers) if num_workers 
                                # is set in the training function. Otherwise, num_workers is 
                                # automatically set to cpus-per-task - 1.
#SBATCH --mem=0                 # Use all available memory

# IMPORTANT: Do not set --ntasks as it is automatically inferred from --ntasks-per-node.

# Activate your virtual environment.
# The command might differ depending on your setup.
# For conda environments, use `conda activate my_env`.
source .venv/bin/activate

# On your cluster you might need to set the network interface:
# export NCCL_SOCKET_IFNAME=^docker0,lo

# Might need to load the latest CUDA version:
# module load NCCL/2.4.7-1-cuda.10.0

# Start the training.
srun lightly-train train out="out/my_experiment" data="my_data_dir" model="torchvision/resnet50"

# The following arguments are automatically set based on the SLURM configuration:
# accelerator="gpu"
# devices=2
# num_workers=11
```

Then submit the training job to the SLURM cluster with:
```bash
sbatch my_train_slurm.sh
```
````
