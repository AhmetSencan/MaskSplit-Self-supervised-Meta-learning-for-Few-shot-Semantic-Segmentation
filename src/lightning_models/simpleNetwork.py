import torch
from torch import nn
import pytorch_lightning as pl
from util import intersectionAndUnionGPU, batch_intersectionAndUnionGPU
from torch.nn import functional as F
from model.resnet import resnet50, resnet101
from visu import make_episode_visualization


def masked_global_pooling(mask, Fs):
    # mask size = nway, kshot, 1, 56, 56
    mask = mask.float()
    mask = F.interpolate(mask, size=(Fs.shape[-2], Fs.shape[-1]))
    expanded_mask = mask.expand_as(Fs)
    masked_dog = Fs * expanded_mask
    out = torch.sum(masked_dog, dim=[-1, -2]) / (expanded_mask.sum(dim=[-1, -2]) + 1e-5)
    out = out.unsqueeze(-1)
    out = out.unsqueeze(-1)
    out = out.expand_as(Fs)
    return out


class SimpleNetwork(pl.LightningModule):
    def __init__(self, hparams, visualize=False):
        super(SimpleNetwork, self).__init__()
        print(hparams)
        self.save_hyperparameters()
        self.args = hparams
        self.args.visualize = self.hparams.visualize
        if self.args.arch == 'resnet':
            if self.args.layers == 50:
                resnet = resnet50(pretrained=self.args.pretrained)
            else:
                resnet = resnet101(pretrained=self.args.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                        resnet.conv2, resnet.bn2, resnet.relu,
                                        resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            self.feature_res = (50, 50)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.projection1 = nn.Sequential(nn.Conv2d(in_channels=512,
                                                   out_channels=128,
                                                   kernel_size=(3, 3),
                                                   padding=(1, 1)), nn.GroupNorm(4, 128), nn.ReLU())
        self.projection2 = nn.Sequential(nn.Conv2d(in_channels=1024,
                                                   out_channels=128,
                                                   kernel_size=(3, 3),
                                                   padding=(1, 1)), nn.GroupNorm(4, 128), nn.ReLU())
        self.projection3 = nn.Sequential(nn.Conv2d(in_channels=2048,
                                                   out_channels=128,
                                                   kernel_size=(3, 3),
                                                   padding=(1, 1)), nn.GroupNorm(4, 128), nn.ReLU())

        self.dense_conv = nn.Sequential(nn.Conv2d(in_channels=768,
                                                  out_channels=128,
                                                  kernel_size=(3, 3),
                                                  padding=(1, 1)), nn.GroupNorm(4, 128), nn.ReLU())

        if not self.args.use_all_classes:
            self.val_class_IoU = [ClassIoUNew(self.args.num_classes_val)]
        else:
            self.val_class_IoU = [ClassIoUNew(self.args.num_classes_val),
                                  ClassIoUNew(self.args.num_classes_val),
                                  ClassIoUNew(self.args.num_classes_val),
                                  ClassIoUNew(self.args.num_classes_val)]

        self.decoder = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 128),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 128),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 64),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 64),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 32),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 32),
                                     nn.ReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(32, 16, (3, 3), padding=(1, 1), bias=True),
                                     nn.GroupNorm(4, 16),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 2, (1, 1), bias=False))

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, support: torch.Tensor, smask: torch.Tensor, query: torch.Tensor):
        # Support Feature Extraction
        support = support.squeeze(1)
        with torch.no_grad():
            fs = self.layer0(support)
            fs = self.layer1(fs)
            fs_2 = self.layer2(fs)
            fs_3 = self.layer3(fs_2)
            fs_4 = self.layer4(fs_3)

            # Query Feature Extraction
            fq = self.layer0(query)
            fq = self.layer1(fq)
            fq_2 = self.layer2(fq)
            fq_3 = self.layer3(fq_2)
            fq_4 = self.layer4(fq_3)
        
        fq_2 = self.projection1(fq_2)
        fs_2 = self.projection1(fs_2)
        fs_3 = self.projection2(fs_3)
        fq_3 = self.projection2(fq_3)
        fq_4 = self.projection3(fq_4)
        fs_4 = self.projection3(fs_4)
        smask[smask == 255] = 0

        fq_2 = torch.cat([fq_2, masked_global_pooling(smask, fs_2)], dim=1)
        fq_3 = torch.cat([fq_3, masked_global_pooling(smask, fs_3)], dim=1)
        fq_4 = torch.cat([fq_4, masked_global_pooling(smask, fs_4)], dim=1)

        fq = torch.cat([fq_2, fq_3, fq_4], dim=1)
        final = self.dense_conv(fq)
        result = self.decoder(final)
        return result

    def training_step(self, batch, batch_nb):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch
        y_hat = self.forward(spprt_imgs, spprt_labels, qry_img)
        target = target.long()
        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        area_intersection, area_union, area_target = intersectionAndUnionGPU(preds, target, 2)
        miou = (area_intersection.float() / area_union.float())[1]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_miou', miou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_nb=0):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch
        y_hat = self.forward(spprt_imgs, spprt_labels, qry_img)
        target = target.long()
        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        area_intersection, area_union, area_target = intersectionAndUnionGPU(preds, target, 2)
        miou = (area_intersection.float() / area_union.float())[1]
        self.log("val_miou_old", miou, on_epoch=True, prog_bar=True, logger=True)
        y_hat = y_hat.unsqueeze(1)
        target = target.unsqueeze(1)
        intersection, union, _ = batch_intersectionAndUnionGPU(y_hat, target, 2)

        self.val_class_IoU[dataset_nb].update(intersection, union, subcls_list)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        if len(self.val_class_IoU) == 1:
            val_miou = self.val_class_IoU[0].compute()
            self.log('val_miou', val_miou, prog_bar=True, logger=True)
            self.val_class_IoU[0].reset()
        else:
            for i, calculator in enumerate(self.val_class_IoU):
                val_miou = calculator.compute()
                calculator.reset()
                self.log("val_miou_" + str(i), val_miou, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx, dataset_nb=0):
        qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, image_paths = batch
        target = target.long()
        y_hat = self.forward(spprt_imgs, spprt_labels, qry_img)
        loss = self.criterion(y_hat, target)
        preds = torch.argmax(y_hat, dim=1)
        area_intersection, area_union, area_target = intersectionAndUnionGPU(preds, target, 2)
        miou = (area_intersection.float() / area_union.float())[1]
        self.log("test_miou_old", miou, on_epoch=True, prog_bar=True, logger=True)
        y_hat = y_hat.unsqueeze(1)
        target = target.unsqueeze(1)
        intersection, union, _ = batch_intersectionAndUnionGPU(y_hat, target, 2)

        self.val_class_IoU[dataset_nb].update(intersection, union, subcls_list)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        if self.args.visualize:
            for i in range(len(qry_img)):
                path = image_paths[0][i].split(".")[0].split("/")[-1]+"_"+str(dataset_nb)+"_"+str(batch_idx)+"_"+str(i)
                make_episode_visualization(spprt_imgs[i].cpu().numpy(),
                                           qry_img[i].cpu().numpy(),
                                           spprt_labels[i].cpu().numpy(),
                                           target[i, 0].cpu().numpy(),
                                           y_hat[i].cpu().numpy(),
                                           path)

    def on_test_epoch_end(self) -> None:
        test_miou = self.val_class_IoU[0].compute()
        self.log('test_miou', test_miou, prog_bar=True, logger=True)
        self.val_class_IoU[0].reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return [optimizer]


class ClassIoUNew:
    def __init__(self, class_size):
        self.class_size = class_size
        self.cls_iou = torch.zeros(self.class_size)
        self.cls_counts = torch.zeros(self.class_size)

    def update(self, intersection: torch.Tensor, union: torch.Tensor, classes: torch.Tensor):  # , batch_nb):
        for i, task_cls in enumerate(classes[0]):
            iou_score = intersection[i, 0, 1] / union[i, 0, 1]
            if union[i, 0, 1] != 0 and not torch.isnan(iou_score) and not torch.isinf(iou_score):
                self.cls_iou[(task_cls - 1) % self.class_size] += iou_score
                self.cls_counts[(task_cls - 1) % self.class_size] += 1

    def compute(self):
        print(self.cls_iou, self.cls_counts)
        return torch.mean(self.cls_iou[self.cls_counts != 0] / self.cls_counts[self.cls_counts != 0])

    def reset(self):
        self.cls_iou = torch.zeros(self.class_size)
        self.cls_counts = torch.zeros(self.class_size)
