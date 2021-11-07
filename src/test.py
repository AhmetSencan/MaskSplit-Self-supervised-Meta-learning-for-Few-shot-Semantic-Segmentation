from pytorch_lightning import Trainer
from pytorch_lightning.utilities import seed
from util import load_cfg_from_cfg_file
import argparse
from lightning_models import simpleNetwork
from dataset.dataset import get_test_loader
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', type=str, required=True, help='config file')
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config)
    return cfg


if __name__ == "__main__":
    opt = parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(202)
    random.seed(202)
    torch.manual_seed(202)
    torch.cuda.manual_seed(202)
    seed.seed_everything(202)
    torch.cuda.manual_seed_all(202)

    model = simpleNetwork.SimpleNetwork.load_from_checkpoint(checkpoint_path=opt.ckpt_used,
                                                             map_location="cpu",
                                                             visualize=opt.visualize).eval()
    trainer = Trainer(gpus=opt.gpus, benchmark=True)

    for i in range(5):
        trainer.test(model, test_dataloaders=[get_test_loader(opt)])
