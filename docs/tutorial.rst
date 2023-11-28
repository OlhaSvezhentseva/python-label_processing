Tutorial
========

*A brief introduction on how to build a workflow using this package*

Data
----
To use this package you need some kind of natural-history labels
with text on it that you want to extract. Some mock data is provided in the ``tests/testdata``
directory. You are encouraged to try this with your own data.
However if your data differs alot from the data described in the accompanying paper
you might want to train your own segmentation/classification models. 
Code for training is also provide in another repository.

1. Preperation
--------------
We first want to create a new directory in a location of your choice and cd into it e.g.:
    
    ``mkdir -p output_elie/output && cd elie_tutorial``

We will now copy our jpg images in there (adjust command with the location of your pictures)

    ``cp -r ../data .``

2. Segmentation
---------------
If your pictures are already cropped- good for you. If they are not, no problem we can help you with that.
However if the results with our models are not satisfying we encourage you again to train your own model
with your data. 
To segment the pictures you need to type the following command in the terminal:

    ``crop_seg.py -j data -o output``

After the script is finished there should be a new directory called `data_cropped` in your `output` directory
Remember that you can get information about the usage for any of the script by running the script the `-h`
flag.

3. Classification
-----------------
Alright now we have a bunch of cropped pictures but what do we with them? The answer is classifying. We want to 
classify them into the categories handwritten and printed.
To run the classification we need to run the following command:

    ``image_classifier.py -m 2 -o output -j output/data_cropped``

If look in the output directory 2 new subdirectories have emerged `handwritten` and `printed`.
We want to use the labels with printed text in the following

4. OCR 
------
We have now the printed labels ready and we can extract their text. For this job we can use the 
tesseract wrapper provided, that luckily does also the preprocessing for you, so you dont have to think about it.
Lets try it out:

    ``image_classifier.py -m 2 -o output -j output/data_cropped``

Again you will find the output in the `output` directory. It is called `ocr_preprocessing.py`.
The structure is explained in the image_classifier section of this documentation.

5. Postprocessing
-----------------
- TODO






