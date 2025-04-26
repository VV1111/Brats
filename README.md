# Preprocessing: preprocess.py
## Step 1: Split the dataset.

Use function ```preprocessor.split_dataset()``` to split the dataset into training, validation, and test sets.

Example command:

```python
python -m data.preprocess
```

⚡ Note: The split results have already been generated and are available in data_prep/split_txt/.

## Step 2: Crop and select slices.

1️⃣ crop and save: preprocess.py
```python
preprocessor.process_split("train")
```
crops the data based on the split results and saves each case as separate .nii.gz files.

2️⃣ Merge Slices: ```merge.py```
merges the cropped .nii.gz files from individual cases into a single .npy or .nii.gz file.

⚡ Note: the merged training set can only be saved as .npy (saving as .nii.gz is not feasible).

## Step 4: Data Loading Example: dataloader.py
A basic example of how to read and load the processed data is provided in dataloader.py.


## Configuration: util/config.py
All relevant parameter settings are located in util/config.py.

You can adjust paths such as:
```
original_dir = "/path/to/your/data"
```