import os
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset
from util.config import Config
from torch.utils.data import DataLoader
from .transform import Simple2DTransform

class MergedNiiDataset(Dataset):
    def __init__(self, split='train', config=None, selected_modalities=None, transform=None, is_val=False,num_cls = 3):
        """
        Dataset for loading merged npy files per modality.

        """
        self.split = split
        self.cfg = config if config else Config()
        self.modalities = selected_modalities if selected_modalities else self.cfg.modalities
        self.transform = transform
        self.is_val = is_val
        self.num_cls = num_cls  

        self.data = {}
        self.length = None

        # === Load .npy files using memory mapping ===
        for mod in self.modalities:
            path = os.path.join(self.cfg.merge_dir, f"{split}_merged_{mod}.npy")
            self.data[mod] = np.load(path, mmap_mode='r')

        self.total_slices = self.data[self.modalities[0]].shape[2]

    def __len__(self):
        return self.total_slices

    def __getitem__(self, index):
        """
        Extract 2D slice from each modality at given Z index.
        """
        # Input: Stack selected modalities as channels
        input_modalities = [mod for mod in self.modalities if mod != 'seg']
        image = np.stack([self.data[mod][:, :, index] for mod in input_modalities], axis=0)

        # Label slice
        label = None
        if 'seg' in self.modalities:
            label = self.data['seg'][:, :, index].astype(np.int8)

        # Normalize image to [0, 1] per slice
        image = (image - image.min()) / (image.max() - image.min() + 1e-5)

        sample = {'image': image.astype(np.float32)}

        if label is not None:
            # label mapping： 1 for NCR, 2 for ED, 4 for ET, and 0 for everything else.
            label[label == 4] = 3
            sample['label'] = label

        # === Transformations ===
        if not self.is_val and self.transform:
            sample = self.transform(sample)

        sample['image'] = torch.from_numpy(sample['image']).float()

        if 'label' in sample:
            label_tensor = torch.from_numpy(sample['label']).long()
            sample['label'] = F.one_hot(label_tensor, self.num_cls).permute(2, 0, 1).float()

        return sample


if __name__=="__main__":
    # # ---------------------------------------------------------------------------- #
    # #                          find the number of classes                          #
    # # ---------------------------------------------------------------------------- #
    # seg_data = np.load("data_prep/merged_nii/train_merged_seg.npy", mmap_mode='r')
    # print("Label max value:", seg_data.max())
    # print("Unique labels:", np.unique(seg_data))
    # exit()
    # ---------------------------------------------------------------------------- #
    #                             example to load data                             #
    # ---------------------------------------------------------------------------- #
    cfg = Config()

    trainsrc = "train"  # "eval"/"test"
    is_val = False
    batch_size =2
    num_workers = 2
    
    transform2d = Simple2DTransform(flip_prob=0.5)

    # === Dataset ===
    train_set = MergedNiiDataset(
        split=trainsrc,
        config=cfg,
        selected_modalities=['t2','seg'],
        transform=transform2d,
        is_val=is_val,
        num_cls = cfg.num_cls
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


    for batch in train_loader:
        images = batch['image']      # Shape: [B, C, H, W]
        labels = batch['label']      # Shape: [B, num_cls, H, W]
        print(images.shape, labels.shape)  # torch.Size([2, 1, 192, 192]) torch.Size([2, 4, 192, 192])
        break