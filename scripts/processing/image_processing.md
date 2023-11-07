# Preprocessing

## Description

This part of the package aims at pre-processing pictures and conducting OCR.

Possible pre-processing steps include:

1. Classifying the images (multi-label images VS single-label images)
2. Cropping  multi-label images 
3. Rotating images
4. Detecting background colour?
5. Applying OCR (google vision/pytesseract) on the images



## Structure


#`crop_seg.py` 
is responsible for cropping images using a pretrained model and is able to 
assign classes using object detection.

It uses a model that has to be trained first with the object detection package detecto. More information on how to train such a model as well as an example notebook can be found in the documenting repository for this package.

Input:
The exprected input are pictures with multiple labels (examples can be seen in the documentation), similar to the ones with which the model is trained. The model can handle some degree of deviation from the test set, but the more it deviates the higher the error rate. It should be noted that these pictures should be in the jpg format. 

Output:
Single label pictures. There are supposed to be one single label image per label on a multilabel pictures. The single label images have filenmaes based on their original pictures with the following format: 
`"{old filename}_label_{image class}_{occurence of label with class}.jpg"*` 

To run the file use the following command:

    `python rotation.py -i input_images -o rotated_images`
 
 Parameters:
 
 -i (input_image_dir): path to the folder with the images
 
 -o (output_image_dir, default = user's current working directory):
  the path to the output folder, where rotated images are saved
  
  

# `rotation.py` 
is responsible for rotating the images.
The angle, the image must be rotated to, is predicted by a pretrained PyTorch model. 
The angle may have the following values: 0° (predicted that the image's orientation is already correct),
90°, 180°, 270°. 

Input:
This script works the best with images provided by picturae (since it was trained on them). With other images it might do more wrong rotations than correct and is therefore not really recommended in this case.  



To run the file use the following command:

    `python rotation.py -i input_images -o rotated_images`
 
 Parameters:
 
 -i (input_image_dir): path to the folder with the images. These need to be single label images.
 
 -o (output_image_dir, default = user's current working directory):
  the path to the output folder, where rotated images are saved
  
  
# `image_classifier.py`

This script is designed to simplify the process of image classification using pre-trained TensorFlow classifier models. 
This script is particularly useful for tasks that involve predicting classes for images and efficiently organizing them based on these predictions.
Executes the `tensorflow_classifier.py` module.

Key Features:

 1. Command-Line Usage: Users can execute the script from the command line with options to specify the classifier model, input image directory, and output directory for saving results.
The command `-h` or `--help` displays a usage message and a list of available command-line options, along with brief explanations for each option.

2. Model Selection: The script supports three pre-defined classifier models, each tailored to a specific classification task. Users can choose the appropriate model for their image classification needs (e.g., distinguishing between 'nuri' and 'not_nuri' (1), 'handwritten' and 'printed' (2), or 'multi' and 'single' labels (3)).

3. Automatic Class Selection: Based on the chosen model, the script automatically selects the class labels associated with that model. This simplifies the process of predicting image classes, as users don't need to manually specify class names.

4. Predictions and Organization:  After parsing command-line arguments and selecting the model and class names, the script proceeds to load the selected model, predict classes for the images in the provided directory, and organize the images into separate directories according to their predicted classes.

5. Customizable Output Directory: Users have the option to specify an output directory for saving both the results (in CSV format) and the classified images. The default output directory is set to the current working directory.
      



To utilize the script, execute it from the command line as follows:


    python image_classifier.py [-h] -m <model_number> -j <path_to_jpgs> -o <path_to_outputs>

  

# `tesseract_ocr.py`

uses the tesseract python wrapper "pytesseract" to perform ocr on a set of pictures. It does this by first performing a set of preprocessing steps in order to make the images more uniform and thereby improving the results greatly.
The preprocessing steps are: 


Input:

The input should be single label images. Also angles of  the texts should be very small, otherwise tessseract is not able to recognise them.

Output:
Outputs a json file with entries for every single label image, that have the following format:

`{"ID": "<filename>", "text": "<ocr transcript>"}`


# `vision_api.py`
Performs the Google Vision OCR on the segmented labels by calling the API and returns it as a json file. This service costs money as you have to use the google cloud api. For this a google cloud account is needed as well as a json file containing the credentials

Input:
The input should be single label pictures, that are in the right rotation. Preprocessing is not needed, since google vision runs its own image preprocessing routine on the server. 
Also a path to the google credentials json file needs to be passed as an argument. Information about how to generate the crededential json can be found here: 
https://developers.google.com/workspace/guides/create-credentials

Output:
Outputs a json file with entries for every single label image, that have the following format:

`{"ID": "<filename>", "text": "<ocr transcript>"}`

