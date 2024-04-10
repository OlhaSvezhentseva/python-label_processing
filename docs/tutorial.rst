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
Code for the training process is available in the "training_notebooks" repository.

1. Preparation
--------------
We first want to create a new directory in a location of your choice and cd into it:
    
    ``mkdir -p elie_tutorial/output && cd elie_tutorial``

We will now copy our jpg images in there (adjust command with the location of your pictures):

    ``cp -r ../data .``

2. Label Detection
-------------------
If your images are already cropped, that's excellent. 
In case they are not, there's no issue; we can assist you with the cropping process. 
However, if the results obtained with our pre-trained models are not satisfactory, we recommend considering training your own model with your specific dataset.

To initiate the label detection process on your images, please execute the following command in the terminal:

    ``detection.py -j data -o output``

Once the script execution is complete, you should find a new directory named `data_cropped` within your `output` directory. 
Don't forget that you can obtain information about the usage of any script by running the script with the `-h` flag.

3. Label Analysis
------------------
Now that we have a collection of cropped pictures, the next step is pixel analysis.
We want to know which labels are empty (no text) and which one are not empty.
To initiate the analysis, run the following command:

    ``analysis.py -j output/data_cropped -o output``

Once the script execution is complete, you should find two new directories named `empty` and `not_empty` within your `output` directory. 

4. Label Classification
------------------------
The next step is classification. 
Our objective is to classify the labels into the categories of "handwritten" and "printed".
To initiate the classification process, run the following command:

    ``classifiers.py -m 2 -o output -j output/not_empty``

Upon inspecting the output directory, you will notice the emergence of two new subdirectories: `handwritten` and `printed`. 
In the subsequent steps, we intend to utilize the labels associated with printed text.

1. Label Rotation
------------------
The labels are cropped, their content filtered and classified.
Now we want to make sure that they are all rotated to a 0° angle, so their text can be read correctly by the OCR.
To initiate the rotation process, run the following command:

    ``rotation.py -o output -i output/printed``
    
The rotated pictures are saved in the `output` directory.

6. OCR 
-------
Now that we have the labels with printed text ready, we can proceed to extract their text. 
Fortunately, we provide a Tesseract wrapper that also handles the preprocessing, alleviating the need for manual intervention. 
Let's give it a try:

    ``tesseract.py -d output/data_cropped -o output``

The results are in the `output` directory under the name `ocr_preprocessed.json`. 
The structure is explained in the tesseract_ocr section of this documentation.

1. Postprocessing
-----------------
This process facilitates the examination of generated transcripts by categorizing them into distinct classes: "nuris," empty transcripts, nonsense transcripts, and plausible transcripts.

Transcripts are designated as nonsensical if the average length of a single token is less than 2 letters. 
This determination stems from the recognition that Pytesseract might produce numerous single-letter tokens, particularly when erroneously identifying paper imperfections as text.
Plausible transcripts undergo correction by removing non-ASCII and non-alphanumeric symbols before being saved as `corrected_transcripts.json`.

    ``process.py -j ocr_preprocessed.json -o output``

A JSON file will be saved for each category in the output folder.

1. Clustering
-------------
For analyzing the extracted data, we suggest employing another Python package known as `python-mfnb`. 
This toolset is specifically crafted for the extraction, processing, and organization of information derived from collecting event labels within natural history collections. 
Equipped with various scripts tailored for specific functions, such as clustering labels, associating labels with collecting events, and preprocessing datasets, this package facilitates tasks such as pinpointing inconsistencies, constructing collecting event objects, and executing full-text searches on label collections. 
For further details on utilizing this package, please consult its documentation.

Illustrating the usage of this package, consider the following example. 
The input file required is the path to the `corrected_transcripts.json` generated during the postprocessing step. 
In this instance, we leverage the `--min-score=FLOAT` option within the `sort_labels.py` script from the `python-mfnb` package. 
This script operates by clustering labels based on text similarity and parsing localization, date, and collector's names from the raw text. 
The `--min-score` option allows us to designate a float value representing the minimum similarity score required for two labels to be grouped together.
The final argument represents the path and filename for the clustering output.

    ``sort_labels.py -s 0.6 corrected_transcripts.json > sorted_transcripts.json``

To visualize the clustering results interactively in a scatter plot, you can utilize the `cluster_eval.py` script provided in our package. 
To do so, you will require the OCR JSON file output, the clustering output generated by the `sort_labels.py` script, the designated path for the visualization result output directory, and, lastly, the cluster size, which represents the minimum number of labels required for a cluster to be included in the plot.

    ``cluster_eval.py -gt corrected_transcripts.json -c sorted_transcripts.json -o output -s 2``













