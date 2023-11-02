# Preprocessing

## Description

This module aims at pre-processing pictures and conducting OCR.

Possible pre-processing steps include:

1. Classifying the images (multi-label images VS single-label images)
2. Cropping  multi-label images 
3. Rotating images
4. Detecting background colour?
5. Applying OCR (google vision/pytesseract) on the images



## Structure


File `crop_seg.py` is responsible for .



To run the file use the following command:

    `python rotation.py -i input_images -o rotated_images`
 
 Parameters:
 
 -i (input_image_dir): path to the folder with the images
 
 -o (output_image_dir, default = user's current working directory):
  the path to the output folder, where rotated images are saved
  
  

File `rotation.py` is responsible for rotating the images.

The angle, the image must be rotated to, is predicted by a pretrained PyTorch model. 
The angle may have the following values: 0° (predicted that the image's orientation is already correct),
90°, 180°, 270°. 

To run the file use the following command:

    `python rotation.py -i input_images -o rotated_images`
 
 Parameters:
 
 -i (input_image_dir): path to the folder with the images
 
 -o (output_image_dir, default = user's current working directory):
  the path to the output folder, where rotated images are saved
  
 