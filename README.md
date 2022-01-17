# Deepfake detection

## Installation

Install Anaconda environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
```

Activate Anaconda environment: 
```bash
conda activate deepfake-detection
```

## Running the code
`preprocess/crop_face.py` - Creates directories with cropped images - run with train and test MODE option.
`src/generate_validation_ds.py` - Generates validation dataset from created testing set.
`src/generate_patch_pairs.py` - Generates pathc pairs.
`src/train.py` - Used for training the Xception model.
`src/train_res.py` - Used for training the ResNet-18 model.
`src/test.py` - Used for testing the trained models.
