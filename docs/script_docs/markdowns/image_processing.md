# Image-Processing

## Description
This section of the package is designed for the processing of images and the execution of Optical Character Recognition (OCR). 
Key processing steps encompass:

1. Image classification, distinguishing between multi-label and single-label images, as well as images containing a QR-Code or not.

2. Cropping multi-label images to transform them into single-label images.

3. Rotation of images to optimize the OCR output.

4. Application of OCR, utilizing tools such as Google Vision or Pytesseract, to extract text from the images.

## Scripts
For usage information, run any of these scripts with the option --help.


### crop_seg.py
This script is designed to crop images based on a pre-trained model and is capable of assigning classes through object detection.
To utilize this script, a model must be trained in advance using the detecto object detection package. Detailed instructions on model training, along with an illustrative notebook, can be found in the documentation repository for the detecto package: https://detecto.readthedocs.io/en/latest/.

  **Input:**
  The expected input comprises images containing multiple labels, as illustrated in the documentation. These images should resemble those used to train the model. 
  While the model can tolerate some deviation from the training set, increased deviation leads to a higher error rate. It's essential to note that these images should be in the JPG format.

  **Key Features:**
  1. Predict Class of a Label: Uses a pre-trained object detection model for label prediction (class and coordinates) with a configurable threshold, returning results in a Pandas DataFrame.

  2. Cropping Functionality: Crops images based on the model's coordinates predictions, saving them in separate directories for each class.

  3. File Management: Generates fitting filenames (with class) and organizes results in a structured manner.

  **Usage:**
  To utilize the script, execute it from the command line as follows:
    crop_seg.py [-h] [-c N] [-np N] -j </path/to/jpgs> -o </path/to/jpgs_outputs>

### rotation.py
This script is designed to automate the image rotation process using a pre-trained PyTorch model. 
The model predicts the angle by which each image needs to be rotated, with possible values of 0° (indicating that the image's orientation is correct), 90°, 180°, or 270°.

  **Input:**
  This script is most effective when used with images provided by Picturae, as it has been specifically trained on them. When applied to other images, it may result in more incorrect rotations than correct ones, making it less recommended for such cases.


# `rotation.py` 
is responsible for rotating the images.
The angle, the image must be rotated to, is predicted by a pretrained PyTorch model. 
The angle may have the following values: 0° (predicted that the image's orientation is already correct),
90°, 180°, 270°. 

  **Input:**
  This script works the best with images provided by picturae (since it was trained on them). With other images it might do more wrong rotations than correct and is therefore not really recommended in this case.  


  **Key Features:**
  1. PyTorch Configuration: The TorchConfig class provides a convenient configuration for PyTorch, including device selection (GPU or CPU), model path specification, and image transformation settings. 
    
  2. Rotation Detection Neural Network: The RotationDetector class implements a neural network for rotation detection. It extends a base neural network model and customizes the classifier layer to predict rotation angles.
    
  3. Image Rotation Functionality: The rotation function leverages the pre-trained model to predict the rotation angle for an input image and performs the rotation. The script supports four rotation angles: 0°, 90°, 180°, and 270°.

  **Usage:**
  To utilize the script, execute it from the command line as follows:
    python rotation.py [-h] -i <input_images> -o <rotated_images>

  
### image_classifier
This script is designed to simplify the process of image classification using pre-trained TensorFlow classifier models. 
This script is particularly useful for tasks that involve predicting classes for images and efficiently organizing them based on these predictions.

  **Key Features:**
  1. Command-Line Usage: Users can execute the script from the command line with options to specify the classifier model, input image directory, and output directory for saving results.
  The command `-h` or `--help` displays a usage message and a list of available command-line options, along with brief explanations for each option.

  2. Model Selection: The script supports three pre-defined classifier models, each tailored to a specific classification task. Users can choose the appropriate model for their image classification needs (e.g., distinguishing between 'nuri' and 'not_nuri' (1), 'handwritten' and 'printed' (2), or 'multi' and 'single' labels (3)).

  3. Automatic Class Selection: Based on the chosen model, the script automatically selects the class labels associated with that model. This simplifies the process of predicting image classes, as users don't need to manually specify class names.

  4. Predictions and Organization: After parsing command-line arguments and selecting the model and class names, the script proceeds to load the selected model, predict classes for the images in the provided directory, and organize the images into separate directories according to their predicted classes.

  5. Customizable Output Directory: Users have the option to specify an output directory for saving both the results (in CSV format) and the classified images. The default output directory is set to the current working directory.
      
  **Usage:**
  To utilize the script, execute it from the command line as follows:
    image_classifier.py [-h] -m <model_number> -j <path_to_jpgs> -o <path_to_outputs>


### tesseract_ocr.py
This script is designed for Optical Character Recognition (OCR) using the Tesseract OCR engine. 
It performs OCR on a directory containing cropped images in JPG format, applies preprocessing steps, and saves the results in JSON format: `{"ID": "<filename>", "text": "<ocr transcript>"}`. 

  **Input:**
  The input should be single label images. Also angles of the texts should be very small, otherwise Tessseract is not able to recognise them.

  **Key Features:**
  1. Image Preprocessing: Grayscale conversion, Gaussian blur, noise reduction, thresholding, dilation, and erosion.
    
  2. Deskewing: Automatic skew angle detection and correction for improved OCR accuracy.
    
  3. QR Code Detection: Identification and decoding of QR codes present in images.
    
  4. Tesseract OCR: Multilingual support, customizable configurations, and text processing for accurate results.
    
  5. Configuration and Language Settings: Customizable Tesseract configurations and support for multiple languages.
    
  6. Image Saving: Save preprocessed images to a specified directory with optional filename appendix.
      
  **Usage:**
  To utilize the script, execute it from the command line as follows:
    tesseract_ocr.py [-h] [-v] [-t <thresholding>] [-b <blocksize>] [-c <c_value>] -d <crop-dir> [-multi <multiprocessing>] -o <outdir> [-o <out-dir>]


### vision_api.py
Performs Optical Character Recognition (OCR) using the Google Vision API on segmented labels, initiating API calls and generating results in a JSON file: `{"ID": "<filename>", "text": "<ocr transcript>"}`.
Please note that this service incurs costs, as it relies on the Google Cloud API. To utilize this service, a Google Cloud account is required, along with a JSON file containing the necessary credentials.

  **Input:**
  The input should consist of individual images containing single labels, ensuring that the images are correctly oriented. Preprocessing is unnecessary, as Google Vision applies its own image preprocessing routine on the server.
  Additionally, a path to the Google Cloud credentials JSON (https://developers.google.com/workspace/guides/create-credentials) file must be provided as an argument. 

  **Key Features:**
  1. Google Cloud Vision Interaction: Interacts with the Google Cloud Vision API for Optical Character Recognition (OCR) tasks on images.

  2. Credential Management: Exports credentials by setting the credentials JSON as an environment variable.

  3. Image Reading and Initialization: Reads image files and initializes an instance of the VisionApi class.

  4. String Processing: Processes Google Vision OCR output, replacing newlines with spaces, and supports ASCII or UTF-8 encoding.

  5. API Call and Error Handling: Performs the actual API call, handles errors, and returns the processed transcription along with bounding box information.

  **Usage:**
  To utilize the script, execute it from the command line as follows:
    vision_api.py [-h] [-np] -d <crop-dir> -c <credentials>
