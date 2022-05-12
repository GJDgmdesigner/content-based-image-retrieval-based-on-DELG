Please Add your dataset link here

# How to prepare your dataset
Download the GLDv2 from https://github.com/cvdfoundation/google-landmark

Put all content in the `landmark` folder.

Put trian/test/index images in the `trian/test/index` folder.

Put metadata annotation in the `metadata` folder.

Then run process_train_clean.py code to generate train and val annotation file.

Then create a symbolic link named `landmark_train` to the `train` folder of `landmark` folder.