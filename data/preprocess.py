# data/preprocessing.py

import os, random, numpy as np, nibabel as nib
from util.config import Config

class Preprocessor:
    """
    Handles dataset splitting, cropping based on segmentation, and saving in desired formats.
    """
    def __init__(self, config):
        self.cfg = config
        random.seed(self.cfg.seed)

        # Create directories if paths are set
        for path in [self.cfg.split_dir, self.cfg.crop_info_dir, self.cfg.gz_dir,self.cfg.npy_dir]:
            if path: os.makedirs(path, exist_ok=True)

    def split_dataset(self):
        """
        Split dataset according to split_type and split_ratio.
        """
        cases = [d for d in os.listdir(self.cfg.original_dir) if d.startswith("BraTS")]
        random.shuffle(cases)
        total = len(cases)

        if self.cfg.split_type == "all":
            ratio_sum = sum(self.cfg.split_ratio)
            n_train = int(total * self.cfg.split_ratio[0] / ratio_sum)
            n_eval = int(total * self.cfg.split_ratio[1] / ratio_sum)
            splits = {
                "train": cases[:n_train], 
                "eval": cases[n_train:n_train + n_eval], 
                "test": cases[n_train + n_eval:]}
        elif isinstance(self.cfg.split_type, float):
            n_select = int(total * self.cfg.split_type)
            splits = {"train": cases[:n_select], "test": cases[n_select:]}


        for split, ids in splits.items():
            with open(os.path.join(self.cfg.split_dir, f"{split}.txt"), 'w') as f:
                f.write('\n'.join(ids))
            print(f"{split}: {len(ids)} cases saved.")

    def get_crop_region(self, seg):
        """
        Determine cropping coordinates based on segmentation mask.
        - Z-axis: based on non-zero label region with extension.
        - X/Y-axis: center crop if enabled.
        """
        z_idx = np.any(np.any(seg > 0, axis=0), axis=0).nonzero()[0]
        if len(z_idx) == 0: return None  # Skip if no labels in seg

        start_z = max(0, z_idx[0] - self.cfg.extend_size)
        end_z = min(seg.shape[2], z_idx[-1] + self.cfg.extend_size + 1)

        if self.cfg.center_crop_xy:
            x_c, y_c = np.array(seg.shape[:2]) // 2
            crop_x = [max(0, x_c - self.cfg.crop_size_xy[0]//2), x_c + self.cfg.crop_size_xy[0]//2]
            crop_y = [max(0, y_c - self.cfg.crop_size_xy[1]//2), y_c + self.cfg.crop_size_xy[1]//2]
        else:
            crop_x, crop_y = [0, seg.shape[0]], [0, seg.shape[1]]

        return {"crop_x": crop_x, "crop_y": crop_y, "crop_z": [start_z, end_z]}

    def process_case(self, case_id, split_type):
        """

        - Read -> Crop -> Save (npy and/or nii.gz based on config)
        """
        path = os.path.join(self.cfg.original_dir, case_id)
        seg = nib.load(os.path.join(path, f"{case_id}_seg.nii.gz")).get_fdata()

        crop_info = self.get_crop_region(seg)
        if not crop_info: return

        # Save crop_info
        info_dir = os.path.join(self.cfg.crop_info_dir, f"{split_type}_crop_info")
        os.makedirs(info_dir, exist_ok=True)
        np.save(os.path.join(info_dir, f"{case_id}_crop_info.npy"), crop_info)

        slices = (slice(*crop_info["crop_x"]), slice(*crop_info["crop_y"]), slice(*crop_info["crop_z"]))

        for mod in self.cfg.modalities:
            mod_path = os.path.join(path, f"{case_id}_{mod}.nii.gz")
            img = nib.load(mod_path).get_fdata()[slices].astype(self.cfg.dtype)

            # Save as .npy if configured
            if self.cfg.npy_dir:
                npy_path = os.path.join(self.cfg.npy_dir, split_type)
                os.makedirs(npy_path, exist_ok=True)
                np.save(os.path.join(npy_path, f"{case_id}_{mod}.npy"), img)

            # Save as .nii.gz if configured
            if self.cfg.gz_dir:
                gz_path = os.path.join(self.cfg.gz_dir, split_type)
                os.makedirs(gz_path, exist_ok=True)
                nib.save(nib.Nifti1Image(img, affine=np.eye(4)),
                         os.path.join(gz_path, f"{case_id}_{mod}.nii.gz"))

    def process_split(self, split_type="train"):
        """
        Process all cases in a given split (train/eval/test).
        """
        split_txt = os.path.join(self.cfg.split_dir, f"{split_type}.txt")
        if not os.path.exists(split_txt):
            print(f"{split_txt} not found."); return

        with open(split_txt, 'r') as f:
            ids = f.read().splitlines()

        for case_id in ids:
            self.process_case(case_id, split_type)
        print(f"{split_type} processing completed.")


if __name__ == "__main__":
    cfg = Config()                           
    preprocessor = Preprocessor(cfg)          

    preprocessor.split_dataset()              
         
    # preprocessor.process_split("eval")  
    preprocessor.process_split("test")  
    preprocessor.process_split("train")       
