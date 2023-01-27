label_preprocessing 0.0.2
===================================================================

*A Python package for the Berlin Natural History Museum*

.. contents ::

Introduction
------------
This package contains three segmentation models and functions to handle
segmenting and performing an OCR on entomology specimen labels. Its installation also includes 
scripts designed for classifying the labels during segmentation and preprocessing the images before applying the OCR.


Installation
------------
1. Clone python-mfnb from https://gitlab.com/preuss.leonardo/python-label_processing.git .

   `git clone https://gitlab.com/preuss.leonardo/python-label_processing.git`

2. cd in label_processing-main

   `cd <path to python-label_processing-main>`
   
3. Install with pip, which will automatically fetch the requirements if
   you don't have it already.

   `pip install .`


Modules
-------
*  segmentation_cropping
   Module containing all functions concerning the application of the segmentation
   models, the classification of the cropped labels and the use of the predicted coordinates for cropping the labels.  


*  ocr_pytesseract
   Module containing the Pytesseract OCR parameters to be performed on the _cropped jpg outputs.


Scripts
-------
For usage information, run any of these scripts with the option --help.

   `<script> -h`

*OCR2data.py*
   Execute the label_preprocessing.segmentation_cropping and label_preprocessing.ocr_pytesseract modules.

**Takes as inputs:**
   - the path to the inputs (jpg_dir)
   - the model used for the segmentation (model)
   - the classes used for the model (classes)
   - the path to the directory in which the resulting crops, the csv and ocr outputs will be stored (out_dir)

**Outputs:**
   - the labels in the pictures are segmented and cropped out of the picture, becoming their own file named after their jpg of origin and class.
   - the segmentation outputs are also saved as a dataframe (filename, class, prediction score, coordinates).
   - the ocr outputs after performing on the cropped images.


Input preparation
-----------------
**The modules are best to be performed on jpg images of labels from entomology databases such as:**
   - `AntWeb`_
   - `Bees&Bytes`_
   - LEP_PHIL - pictures of specimens from the Philippines (by Théo Leger)
   - `Atlas of Living Australia`_


**In terms of data acquisition, the following standards are recommended to optimize the outputs:**

- the pictures quality should be standardized and uniform as much as possible, preferably using macro photography, the .jpg format and    300 DPI resolution.
- if there are multiple labels in one picture, they should be clearly separated from one another without overlapping. The text in the label should be aligned horizontally.
- if possible, the specimen shouldn't be present in the picture with the labels.
- if the labels in the different pictures are similar (same colours and/or same nature/content), they should always be placed the same way at the same spot from one picture to another. *ex: label with location always bottom right, collection number top left, taxonomy top right etc...*
- a black background like in LEP_PHIL is prefered, but a white background is also acceptable.


.. _AntWeb: https://www.antweb.org/
.. _Bees&Bytes: https://www.zooniverse.org/projects/mfnberlin/bees-and-bytes  
.. _Atlas of Living Australia: https://www.ala.org.au/
