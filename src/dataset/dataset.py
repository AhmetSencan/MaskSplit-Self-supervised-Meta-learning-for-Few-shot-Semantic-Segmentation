import cv2
import numpy as np
from torch.utils.data import Dataset
from .utils import make_dataset, make_dataset2
from .classes import get_split_classes, filter_classes
import torch
import random
import argparse
from typing import List
from torch.utils.data import RandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_loader(args: argparse.Namespace,
                     return_paths: bool = False) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a standard loader (not episodic)
    """
    assert args.train_split in [0, 1, 2, 3]
    train_transform = A.Compose([
                        A.RandomResizedCrop(scale=(0.5, 1.2),
                                            height=args.image_size,
                                            width=args.image_size),
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                        A.Rotate(p=0.5),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()
                      ])

    split_classes = get_split_classes(args)
    class_list = split_classes[args.train_name][args.train_split]['train']

    # ===================== Build loader =====================
    train_data = EpisodicData(transform=train_transform,
                              class_list=class_list,
                              data_list_path=args.train_list,
                              args=args)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    return train_loader


def get_test_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
        Build the episodic validation loader.
    """
    assert args.test_split in [0, 1, 2, 3, -1, 'default']
    val_transform = A.Compose([
                        A.Resize(height=args.image_size, width=args.image_size),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()
                      ])
    split_classes = get_split_classes(args)

    # ===================== Filter out classes seen during training =====================
    if args.test_name == 'default':
        test_name = args.train_name
        test_split = args.train_split
    else:
        test_name = args.test_name
        test_split = args.test_split
    class_list = filter_classes(args.train_name, args.train_split, test_name, test_split, split_classes)

    # ===================== Build loader =====================
    val_data = EpisodicData(transform=val_transform,
                            class_list=class_list,
                            data_list_path=args.val_list,
                            args=args)
    val_sampler = RandomSampler(val_data, replacement=True, num_samples=2500)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)
    return val_loader


def get_multi_test_loader(args: argparse.Namespace):
    """
        Build the episodic validation loader.
    """
    assert args.test_split in [0, 1, 2, 3, -1, 'default']
    val_transform = A.Compose([
                        A.Resize(height=args.image_size, width=args.image_size),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()
                      ])

    # ===================== Filter out classes seen during training =====================
    if args.test_name == 'default':
        test_name = args.train_name
    else:
        test_name = args.test_name

    loaders = []
    for i in range(4):
        args.train_split = i
        split_classes = get_split_classes(args)
        class_list = filter_classes(args.train_name, i, test_name, i, split_classes)
        val_data = EpisodicData(transform=val_transform,
                                class_list=class_list,
                                data_list_path=args.val_list,
                                args=args)
        val_sampler = RandomSampler(val_data, replacement=True, num_samples=2500)
        loaders.append(torch.utils.data.DataLoader(val_data,
                                                   batch_size=32,
                                                   shuffle=False,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=val_sampler))
    return loaders


class EpisodicData(Dataset):
    def __init__(self,
                 transform,
                 class_list: List[int],
                 data_list_path: str,
                 args: argparse.Namespace):

        self.shot = args.shot
        self.data_root = args.data_root
        self.class_list = class_list
        self.data_list, self.sub_class_file_list = make_dataset(args.data_root, data_list_path, self.class_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # ========= Read query image + Chose class =========================
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        for c in label_class:
            if c in self.class_list:  # current list of classes to try
                new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        # == From classes in query image, chose one randomly ===

        class_chosen = np.random.choice(label_class)
        new_label = np.zeros_like(label)
        ignore_pix = np.where(label == 255)
        target_pix = np.where(label == class_chosen)
        new_label[ignore_pix] = 255
        new_label[target_pix] = 1
        label = new_label

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        # ========= Build support ==============================================

        # == First, randomly choose indexes of support images  =
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []

        shot = self.shot

        for k in range(shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or
                  support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = [self.class_list.index(class_chosen) + 1]

        # == Second, read support images and masks  ============
        for k in range(shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1
            support_label[ignore_pix[0], ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " +
                                    support_image_path + " " + support_label_path + "\n"))
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == shot and len(support_image_list) == shot

        # == Forward images through transforms =================

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            qry_img, target = transformed["image"], transformed["mask"]

            for k in range(shot):
                transformed_sup = self.transform(image=support_image_list[k], mask=support_label_list[k])
                support_image_list[k], support_label_list[k] = transformed_sup["image"], transformed_sup["mask"]
                support_image_list[k] = support_image_list[k].unsqueeze(0)
                support_label_list[k] = support_label_list[k].unsqueeze(0)

        # == Reshape properly ==================================
        spprt_imgs = torch.cat(support_image_list, 0)
        spprt_labels = torch.cat(support_label_list, 0)

        return qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, [image_path]


def copy_paste_loader(args: argparse.Namespace,
                      normalize: bool = True) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a standard loader (not episodic)
    """
    assert args.train_split in [0, 1, 2, 3]

    if args.sup_aug:
        augs_sup = A.Compose([A.RandomResizedCrop(scale=(0.5, 1.2),
                                                  height=args.image_size,
                                                  width=args.image_size),
                              A.HorizontalFlip(),
                              A.VerticalFlip(),
                              A.ColorJitter(),
                              A.GaussianBlur(p=0.5),
                              A.ToGray(p=0.4),
                              A.Rotate(p=0.5),
                              A.Resize(height=args.image_size, width=args.image_size)])
    else:
        augs_sup = A.Compose([A.Resize(height=args.image_size, width=args.image_size)])
    if args.query_aug:
        augs_query = A.Compose([A.RandomResizedCrop(scale=(0.5, 1.2),
                                                    height=args.image_size,
                                                    width=args.image_size),
                                A.HorizontalFlip(),
                                A.VerticalFlip(),
                                A.ColorJitter(),
                                A.GaussianBlur(p=0.5),
                                A.ToGray(p=0.4),
                                A.Rotate(p=0.5)])
    else:
        augs_query = A.Compose([A.Resize(height=args.image_size, width=args.image_size)])

    split_classes = get_split_classes(args)
    if args.use_all_classes:
        class_list = split_classes[args.train_name][args.train_split]['unsup']
    else:
        class_list = split_classes[args.train_name][args.train_split]['train']

    # ===================== Build loader =====================
    train_data = EpisodicDataMaskSplit(transform=augs_query,
                                       sup_transform=augs_sup,
                                       class_list=class_list,
                                       data_list_path=args.train_list,
                                       args=args,
                                       normalize=normalize)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True
                                               )
    return train_loader


class EpisodicDataMaskSplit(Dataset):
    def __init__(self,
                 transform,
                 sup_transform,
                 class_list: List[int],
                 data_list_path: str,
                 args: argparse.Namespace,
                 normalize=True):

        self.shot = args.shot
        self.data_root = args.data_root
        self.class_list = class_list
        self.data_list = make_dataset2(args.data_root, data_list_path, self.class_list)
        self.transform = transform
        self.sup_transform = sup_transform
        self.normalize = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    ToTensorV2()]) if normalize else None
        self.resizedcrop = A.RandomResizedCrop(scale=(0.8, 1.2),
                                               height=args.image_size,
                                               width=args.image_size)
        self.args = args

    def __len__(self):
        return len(self.data_list)

    def masksplit(self, image, label, vertical=True):
        masked_pixels_on_columns = np.sum(label, axis=0 if vertical else 1)
        half_the_masked_pixels = np.sum(label) / 2
        pixel_count_cumulative_sum = np.cumsum(masked_pixels_on_columns)
        comask = abs(pixel_count_cumulative_sum - half_the_masked_pixels)
        half_index = np.argmin(comask)

        shift_val = np.random.randint(low=self.args.vcrop_range[0], high=self.args.vcrop_range[1])

        indices = np.where(label == 1)
        if vertical:
            y_max = image.shape[0]-1
            half_first = (half_index + shift_val, 0)
            half_second = (half_index - shift_val, y_max)
        else:
            x_max = image.shape[1] - 1
            half_first = (0, half_index - shift_val)
            half_second = (x_max, half_index + shift_val)

        y_coords = indices[0]
        x_coords = indices[1]
        perp_vec = (half_second[1] - half_first[1], half_second[0] - half_first[0])

        lower_half = np.where(((x_coords - half_second[0])*perp_vec[0] + (y_coords - half_second[1])*perp_vec[1]) < 0)
        upper_half = np.where(((x_coords - half_second[0])*perp_vec[0] + (y_coords - half_second[1])*perp_vec[1]) > 0)
        
        lower_half_indices = (indices[0][lower_half], indices[1][lower_half])
        upper_half_indices = (indices[0][upper_half], indices[1][upper_half])

        split_p = np.random.rand()
        q_label = np.zeros(label.shape)
        s_label = np.zeros(label.shape)
        if split_p > 0.5 or (not self.args.alternate):
            s_indices, q_indices = lower_half_indices, upper_half_indices
        else:
            s_indices, q_indices = upper_half_indices, lower_half_indices
        s_label[s_indices] = 1
        q_label[q_indices] = 1
        if self.args.vcrop_ignore_support:
            q_label[s_indices] = 255
        else:
            q_label[s_indices] = 1
        return q_label, s_label

    def __getitem__(self, index):
        # ========= Read query image + Chose class =========================
        image_path, label_path, saliency_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
        label[label != 255] = 0
        label[label == 255] = 1

        vsplit_prob = np.random.rand()
        hsplit_prob = 1-vsplit_prob
        if vsplit_prob < self.args.vsplit_prob and self.args.vsplit:
            q_label, s_label = self.masksplit(image, label, vertical=True)
        elif hsplit_prob <= self.args.hsplit_prob and self.args.hsplit:
            q_label, s_label = self.masksplit(image, label, vertical=False)
        else:
            s_label = np.copy(label)
            q_label = np.copy(label)
        if (q_label.sum() < 2*16*16) or (s_label.sum() < 2*16*16):
            return self.__getitem__(np.random.randint(low=0, high=self.__len__()))

        transformed_query = self.transform(image=image, mask=q_label)  
        transformed_support = self.sup_transform(image=image, mask=s_label)  
        qry_img, target = transformed_query["image"], transformed_query["mask"]

        spprt_img, spprt_mask = transformed_support["image"], transformed_support["mask"]

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        if self.normalize is not None:  # variables are torch tensor
            augmented_image_normalized = self.normalize(image=qry_img, mask=target)
            qry_img, target = augmented_image_normalized["image"], augmented_image_normalized["mask"]

            sup_augmented_image_normalized = self.normalize(image=spprt_img, mask=spprt_mask)
            spprt_img, spprt_mask = sup_augmented_image_normalized["image"], sup_augmented_image_normalized["mask"]

            qry_img = qry_img.float()
            target = target.long()
            spprt_img = spprt_img.float().unsqueeze(0)
            spprt_mask = spprt_mask.long().unsqueeze(0)

        subcls_list = []
        support_image_path_list = []
        return qry_img, target, spprt_img, spprt_mask, subcls_list, support_image_path_list, [image_path]
