import nibabel as nib
import numpy as np
import os
from tqdm import tqdm 
from util.config import Config

def merge_modalities_to_single_nii(gz_root, split_type, modality_list, output_dir):
    """
    Merge all cropped nii.gz files of each modality into a single stacked nii.gz file along Z-axis.

    """
    split_path = os.path.join(gz_root, split_type)
    os.makedirs(output_dir, exist_ok=True)

    for mod in modality_list:
        print(f"Merging modality: {mod} in {split_type} set...")
        mod_files = sorted([f for f in os.listdir(split_path) if f.endswith(f"{mod}.nii.gz")])

        merged_data = []
        for f_name in tqdm(mod_files):
            img = nib.load(os.path.join(split_path, f_name)).get_fdata()
            merged_data.append(img)

        # Stack along Z-axis (axis=2)
        stacked = np.concatenate(merged_data, axis=2).astype(np.float32)

        # # Save merged file
        # merged_nii = nib.Nifti1Image(stacked, affine=np.eye(4))
        # nib.save(merged_nii, os.path.join(output_dir, f"{split_type}_merged_{mod}.nii.gz"))
        # print(f"Saved {split_type}_merged_{mod}.nii.gz with shape {stacked.shape}")
        np.save(os.path.join(output_dir, f"{split_type}_merged_{mod}.npy"), stacked)


if __name__=="__main__":
    cfg = Config()
    merge_modalities_to_single_nii(
        gz_root=cfg.gz_dir,
        split_type="test",
        modality_list=cfg.modalities,
        output_dir="./data_prep/merged_nii"
    )
    