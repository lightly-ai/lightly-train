import os
from pathlib import Path

import wandb


def init_wandb(args, name=None):
    wandb_dir = args.wandb_dir
    Path(wandb_dir).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir

    # Load the API key
    with open(args.wandb_key_path, "r") as f:
        wandb_key = f.readlines()[0].strip()

    # Login
    wandb.login(key=wandb_key)

    # Infer the run name.
    if name is None:
        name = str(args.out).split("/")[-1]

    # If
    wandb_id_path = os.path.join(args.out, "wandb_id")
    if not os.path.exists(wandb_id_path):
        id = wandb.util.generate_id()
        with open(wandb_id_path, "w") as f:
            f.write(id)
        wandb.init(
            id=id,
            project=args.wandb_project,
            name=name,
            resume="allow",
            dir=args.out,
            mode=args.wandb_mode,
        )
        wandb.config.update(args)
    else:
        with open(wandb_id_path, "r") as f:
            id = f.read()
        wandb.init(
            id=id,
            project=args.wandb_project,
            name=name,
            resume="allow",
            dir=args.out,
            mode=args.wandb_mode,
        )