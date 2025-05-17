import os
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset
from util.config import Config
from torch.utils.data import DataLoader
from .transform import Simple2DTransform
import matplotlib.pyplot as plt



class MergedNiiDataset(Dataset):
    def __init__(self, split='train', config=None, selected_modalities=None, transform=None, is_val=False,num_cls = 4,suffix='npy'):
        """
        Dataset for loading merged npy files per modality.

        """
        self.split = split
        self.cfg = config if config else Config()
        self.modalities = selected_modalities if selected_modalities else self.cfg.modalities
        self.transform = transform
        self.is_val = is_val
        self.num_cls = num_cls  
        self.suffix =suffix
        self.data = {}
        self.length = None

        # === Load .npy files using memory mapping ===
        if suffix =="npy":
            for mod in self.modalities:
                path = os.path.join(self.cfg.merge_dir, f"{split}_merged_{mod}.npy")
                self.data[mod] = np.load(path, mmap_mode='r')
        elif suffix =="gz":
            for mod in self.modalities:
                path = os.path.join(self.cfg.merge_dir, f"{split}_merged_{mod}.nii.gz")
                self.data[mod] = nib.load(path).get_fdata()            
        self.total_slices = self.data[self.modalities[0]].shape[2]

    def __len__(self):
        return self.total_slices

    def __getitem__(self, index):
        """
        Extract 2D slice from each modality at given Z index.
        """
        # Input: Stack selected modalities as channels
        input_modalities = [mod for mod in self.modalities if mod != 'seg']
        image = np.stack([self.data[mod][:, :, index] for mod in input_modalities], axis=0)  # [C, H, W]
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)

        image = torch.from_numpy(image).float()
        
        if 'seg' in self.modalities:
            label = self.data['seg'][:, :, index].astype(np.int8)
            label[label == 4] = 3
            label_tensor = torch.from_numpy(label).long()  # shape: [H, W]
            mask = F.one_hot(label_tensor, num_classes=self.num_cls).permute(2, 0, 1).float()  # [C, H, W]
        else:
            mask = torch.zeros(self.num_cls, *image.shape[1:], dtype=torch.float32)
            
        if self.transform and not self.is_val:
            # save random state so that if more elaborate transforms are used
            # the same transform will be applied to both the mask and the img
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return {'image': image, 'label': mask}



        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 1, 1)
        # plt.imshow(image[0], cmap='gray')
        # plt.title(f'NIfTI [{self.suffix}]')
        # plt.axis('off')
        # plt.savefig(self.suffix+'.png', dpi=150, bbox_inches='tight')
        # plt.close()

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

    trainsrc = "test"  # "eval"/"test"
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
        num_cls = cfg.num_cls,
        suffix = 'npy'
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