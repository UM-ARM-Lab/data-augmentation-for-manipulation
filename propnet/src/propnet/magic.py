import logging
import os

import torch


def wand_lightning_magic():
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["WANDB_SILENT"] = "true"
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
