Tutorial
========

*A brief example on how to build a workflow using this package*

.. contents ::

Data
----
To utilize this package, it is essential to have natural-history labels containing text that you aim to extract. 
Mock data has been supplied in the "tests/testdata" directory for experimentation. 
We encourage you to test the package with your own dataset. 
However, if your data substantially differs from the examples outlined in the associated paper, you may consider training your own segmentation and classification models. 
Code for the training process is available in the "training" repository.

1. Preperation
--------------
We first want to create a new directory in a location of your choice and cd into it:
    
    ``mkdir -p output_elie/output && cd elie_tutorial``

We will now copy our jpg images in there (adjust command with the location of your pictures):

    ``cp -r ../data .``

1. Segmentation
---------------
If your images are already cropped, that's excellent. 
In case they are not, there's no issue; we can assist you with the cropping process. 
However, if the results obtained with our pre-trained models are not satisfactory, we recommend considering training your own model with your specific dataset.

To initiate the segmentation process on your images, please execute the following command in the terminal:

    ``crop_seg.py -j data -o output``

Once the script execution is complete, you should find a new directory named `data_cropped` within your `output` directory. 
Don't forget that you can obtain information about the usage of any script by running the script with the `-h` flag.

3. Classification
-----------------
Now that we have a collection of cropped pictures, the next step is classification. 
Our objective is to classify them into the categories of "handwritten" and "printed".
To initiate the classification process, run the following command:

    ``image_classifier.py -m 2 -o output -j output/data_cropped``

Upon inspecting the output directory, you will notice the emergence of two new subdirectories: `handwritten` and `printed`. 
In the subsequent steps, we intend to utilize the labels associated with printed text.

4. OCR 
------
Now that we have the labels with printed text ready, we can proceed to extract their text. 
Fortunately, we provide a Tesseract wrapper that also handles the preprocessing, alleviating the need for manual intervention. 
Let's give it a try:

    ``tesseract_ocr.py -d output/data_cropped -o output``

Once again, you will discover the output in the `output` directory under the name `ocr_preprocessing.py`. 
The structure is explained in the tesseract_ocr section of this documentation.

5. Postprocessing
-----------------
- TODO











