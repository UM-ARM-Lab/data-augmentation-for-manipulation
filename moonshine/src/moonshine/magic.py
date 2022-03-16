import logging
import os

import torch


def wandb_lightning_magic():
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    os.environ["WANDB_SILENT"] = "true"
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
