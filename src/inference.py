
from util import load_cfg_from_cfg_file
import argparse
from lightning_models import simpleNetwork
from dataset.dataset import get_test_loader
import os
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--sup_im_path", type=str, required=True, help="support image path")
    parser.add_argument("--query_im_path", type=str, required=True, help="query image path")
    parser.add_argument("--sup_mask_path", type=str, required=True, help="support mask path")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    parser.add_argument("--image_size", type=int, default=400) 
    args = parser.parse_args()
    return args

def get_orig_images(opt):

    sup_im = cv2.imread(opt.sup_im_path, cv2.IMREAD_COLOR)
    sup_im = cv2.cvtColor(sup_im, cv2.COLOR_BGR2RGB)
    smask = cv2.imread(opt.sup_mask_path, cv2.IMREAD_GRAYSCALE)
    query = cv2.imread(opt.query_im_path, cv2.IMREAD_COLOR)
    query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
    test_transform = A.Compose([
                A.Resize(height=opt.image_size,width=opt.image_size),
                ])
    transformed_sup = test_transform(image=sup_im, mask=smask)
    sup_im, smask = transformed_sup["image"], transformed_sup["mask"]
    tranformed_query = test_transform(image=query)
    query = tranformed_query["image"]
    return sup_im, smask, query

def read_preprocess(opt):

        sup_im = cv2.imread(opt.sup_im_path, cv2.IMREAD_COLOR)
        sup_im = cv2.cvtColor(sup_im, cv2.COLOR_BGR2RGB)
        sup_im = np.float32(sup_im)
        smask = cv2.imread(opt.sup_mask_path, cv2.IMREAD_GRAYSCALE)
        query = cv2.imread(opt.query_im_path, cv2.IMREAD_COLOR)
        query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        query = np.float32(query)

        test_transform = A.Compose([
                        A.Resize(height=opt.image_size,width=opt.image_size),

                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()
                        ])
        transformed_sup = test_transform(image=sup_im, mask=smask)
        sup_im, smask = transformed_sup["image"], transformed_sup["mask"]
        tranformed_query = test_transform(image=query)
        query = tranformed_query["image"]
        return sup_im.unsqueeze(0).unsqueeze(0), smask.unsqueeze(0).unsqueeze(0), query.unsqueeze(0)

if __name__ == "__main__":
    opt = get_parser()

    model = simpleNetwork.SimpleNetwork.load_from_checkpoint(checkpoint_path=opt.checkpoint,
                                                             map_location="cpu",
                                                             ).eval()
    sup_im, smask, query = read_preprocess(opt)

    if opt.use_cuda:
        model = model.cuda()
        device = model.device
        sup_im, smask, query = sup_im.to(device), smask.to(device), query.to(device)
    
    preds = model(sup_im, smask, query)
    preds = preds = torch.argmax(preds, dim=1).squeeze(0).detach().cpu().numpy()
    if opt.visualize:

        figure, ax = plt.subplots(nrows=1, ncols=4)
        sup_im, smask, query = get_orig_images(opt)
        ax[0].imshow(sup_im)
        ax[0].set_title("support")
        ax[1].imshow(smask, cmap="gray")
        ax[1].set_title("support mask")
        ax[2].imshow(query)
        ax[2].set_title("query")
        ax[3].imshow(preds, cmap="gray")
        ax[3].set_title("prediction")
        plt.tight_layout()
        plt.savefig("inference.png")

    

    




