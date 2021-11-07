
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from util import load_cfg_from_cfg_file
import argparse
from lightning_models import simpleNetwork
from dataset.dataset import get_train_loader, get_test_loader, copy_paste_loader, get_multi_test_loader
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', type=str, required=True, help='config file')
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config)
    return cfg


if __name__ == "__main__":
    opt = parse_args()
    model = simpleNetwork.SimpleNetwork(opt)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name=opt.model_name)

    if opt.strategy == "unsupervised_all":
        cp = []
        for i in range(4):
            path = os.path.join(opt.ckpt_path, opt.model_name) + "_" + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            cp.append(ModelCheckpoint(dirpath=path,
                                      verbose=True,
                                      save_top_k=2,
                                      save_weights_only=False,
                                      mode='max',
                                      period=1,
                                      monitor="val_miou_" + str(i),
                                      ))
        trainer = Trainer(max_epochs=opt.epochs,
                          gpus=opt.gpus, profiler=True,
                          benchmark=True,
                          callbacks=cp,
                          )
    else:
        cp = ModelCheckpoint(dirpath=os.path.join(opt.ckpt_path, opt.model_name),
                             verbose=True,
                             save_top_k=2,
                             save_weights_only=False,
                             mode='max',
                             period=1,
                             monitor="val_miou",
                             )
        if not os.path.exists(os.path.join(opt.ckpt_path, opt.model_name)):
            os.makedirs(os.path.join(opt.ckpt_path, opt.model_name))
        trainer = Trainer(max_epochs=opt.epochs,
                          gpus=opt.gpus, profiler=True,
                          benchmark=True,
                          callbacks=[cp],
                          logger=logger
                          )
    
    if opt.strategy == "unsupervised_fbf":
        trainer.fit(model, copy_paste_loader(opt), get_test_loader(opt))
    elif opt.strategy == "supervised":
        trainer.fit(model, get_train_loader(opt), get_test_loader(opt))
    elif opt.strategy == "unsupervised_all":
        assert opt.use_all_classes is True
        trainer.fit(model, copy_paste_loader(opt), get_multi_test_loader(opt))
