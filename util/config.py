class Config:
    def __init__(self,task="Brats"):

        self.task = task

        # Data source & saving paths
        self.original_dir = "/Volumes/macdata/project/BraTS/archive/BraTS2021_Training_Data"
        self.split_dir = "./data_prep/split_txt"    # For saving .txt
        self.crop_info_dir = "./data_prep/crop_info_dir"       
        self.gz_dir = "./data_prep/gz_data"       # For saving .nii.gz cropped data
        self.npy_dir = None # For saving .npy cropped data. If None or "", npy won't be saved

        self.merge_dir = "data_prep/merged_nii"

        # Split strategy
        self.split_type = "all"          # Options: "all", float (0~1), "custom_train", "custom_eval"
        self.split_ratio = (7, 1, 2)     # Used when split_type == "all"

        # Modalities control
        self.modalities = ['flair', 't1', 't1ce', 't2', 'seg']  # Modalities to process

        # Cropping parameters
        self.extend_size = 5             # Z-axis: slices to extend
        self.center_crop_xy = True       # Whether to crop X, Y dimensions
        self.crop_size_xy = (192, 192)   # Target size if center_crop_xy is True

        # Others
        self.num_cls = 4  #number of classes 
        self.seed = 42
        self.dtype = "float32"