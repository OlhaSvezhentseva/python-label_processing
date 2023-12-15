==========
Evaluation
==========

.. contents ::

Description
-----------

This module aims at evaluating the various modules/models of the pipeline.
For a thorough assessment, it is essential to furnish the *Ground Truth data* for comparison with the predicted data.
Key evaluation steps encompass:

1. Calculate evaluation metrics, such as CER, WER, IoU, or accuracy, based on the processed data.

2. Generate visualizations (violin plots, scatter plots, boxplots, confusion matrices) to represent data distribution or relationships.

3. Save calculated metrics, visualizations, and other relevant results in specified output folders or files.


Scripts
-------
For usage information, run any of these scripts with the option --help.


ocr_accuracy.py
~~~~~~~~~~~~~~~
This script evaluates the accuracy of Optical Character Recognition (OCR) output by comparing it to the ground truth. The Levenshtein distance is calculated for each transcript, measuring discrepancies on both character and word levels. This results in two scores: Character Error Rate (CER) and Word Error Rate (WER), indicating the extent to which the model misinterpreted the text.

CER is normalized between 0 and 1, with 0 signifying identical predicted and reference text. WER, on the other hand, represents the error count divided by the total number of words in the ground truth. WER is not normalized and can exceed 1, particularly if the predicted text contains more words than the ground truth, such as when OCR introduces additional nonsensical words.

The output includes the `ocr_evaluation.csv` file in the specified directory, providing an overview of each transcript (reference and predicted) along with corresponding scores. Additionally, two violin plots representing the score distributions are saved in the folder. For more details on the evaluation metrics, refer to `this article`_.

	**Key Features:**

		1. **Error Rate Calculation::** Implementation of a function (calculate_scores) to compute Character Error Rate (CER) and Word Error Rate (WER) between predicted and ground truth transcriptions.

		2. **CSV Result Table Creation::** Writing evaluation results into a CSV table, including file ID, reference text, OCR output, WER, and CER.

		3. **Violin Plot Generation:** Creating violin plots for CER and WER scores, saving them as images.

		4. **OCR Evaluation:** Evaluating OCR predictions by comparing them to ground truth transcriptions and generating summary statistics.

	**Usage:**

    	To utilize the script, execute it from the command line as follows:

    	.. code:: bash

		ocr_accuracy.py [-h] -g <ground_truth> -p <predicted_ocr> -r <results>


cluster_visualisation.py
~~~~~~~~~~~~~~~~~~~~~~~~
Generates cluster plots using word embeddings and saves the visualizations as HTML links. Word embeddings can be constructed either from the ground truth data or the predicted transcripts. Leveraging a pretrained `gensim model`_, each word in the label is transformed into a vector. These vectors undergo normalization, ensuring each label is uniquely represented by a single vector.

The gensim model accepts vector dimensions (currently set at 100), and using t-SNE a tool for visualizing `high-dimensional data`_, each label is plotted in a 2-dimensional space. Each dot is colored to denote its assigned cluster, facilitating the observation of whether word embedding predictions align with clustering algorithm outcomes.

The resulting HTML plot allows users to hover over dots, revealing the transcript of the corresponding label. This feature aids in quickly assessing if neighboring labels (dots) share similar texts. Considering the potential for numerous clusters, a parameter can be specified to determine the minimal cluster size for plotting—indicating the number of labels required for a cluster to be included in the visualization. This functionality enables focused examination of larger clusters for in-depth analysis.

	**Key Features:**

		1. **Word Vectorization:** Uses gensim Word2Vec to build word vectors for labels, accommodating both ground truth and predicted transcripts.

		2. **Data Processing and Analysis:** Processes label vectors to generate mean vectors; calculates cluster sizes for determining clusters to plot.

		3. **Dimensionality Reduction and Visualization:** Applies t-SNE for 2D dimensionality reduction and utilizes Plotly Express to create an interactive scatter plot.

	**Usage:**

    	To utilize the script, execute it from the command line as follows:

    	.. code:: bash

		cluster_visualisation.py [-h] -gt <ground_truth_ocr_output> -c <cluster_output>  -o <path_to_output_directory> -s <cluster_size>


evaluation_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~~
This script is designed for evaluating the accuracy of of the TensorFlow classifier.

It performs accuracy assessment and generates confusion matrices for a set of predictions. The script reads an input CSV file containing both predicted (pred) and ground truth (gt) labels, calculates accuracy scores, and produces confusion matrices. 

It allows for customizable output directory specification and provides a concise help message for command-line usage.


	**Key Features:**

		1. **Unique Class Extraction:** The script extracts unique classes from the ground truth (gt) column in the input CSV file. This is essential for accurate labeling in the confusion matrices.

		2. **Accuracy Score Calculation:** The script invokes the metrics function from the `accuracy_classifier.py` module to calculate accuracy scores based on the provided predicted and ground truth labels. The results are saved in the output directory if specified.

		3. **Confusion Matrix Generation:** The script runs the cm function from the `accuracy_classifier.py` module to create confusion matrices. These matrices are generated as heatmaps and can also be saved in the output directory if desired.


	**Usage:**

    	To utilize the script, execute it from the command line as follows:

    	.. code:: bash

		evaluation_classifier.py [-h] -o </path/to/outputs> -d </path/to/gt_dataframe>


label_redundancy.py
~~~~~~~~~~~~~~~~~~~
This script utilizes the 'label_evaluation' module to assess redundancy in label transcriptions within a dataset. It calculates the percentage of redundancy and saves the result in a text file. The dataset, provided as a JSON file, is specified via command-line arguments. 
The output, indicating the redundancy percentage, is stored in the user-defined target folder. 

	**Key Features:**

		1. **Data Cleaning Function:** Preprocesses a dataset by converting text to lowercase, removing punctuation and whitespace, and excluding entries containing 'http'.
		
		2. **Redundancy Calculation Function:** Calculates transcription redundancy by identifying duplicate entries in a preprocessed dataset.

		3. **Percentage Redundancy Calculation Function:** Calculates the percentage of transcription redundancy in a preprocessed dataset with grouped duplicates.
	
	**Usage:**

    	To utilize the script, execute it from the command line as follows:

    	.. code:: bash

		label_redundancy.py [-h] -d <dataset-dir> -o <output>


rotation_evaluation.py
~~~~~~~~~~~~~~~~~~~~~~
This script is designed to perform an evaluation of rotation predictions. It takes as input a CSV file containing relevant data, specifically columns named 'before' and 'pred', and produces two primary outputs.

	**Key Features:**

		1. **Comparison Plot:** A comparison plot is generated using seaborn, displaying the distribution of predictions ('straight', 'not_straight') with color-coded bars indicating whether the prediction matches or does not match the expected rotation. The resulting plot is saved as "comparison_plot.png" in the specified output folder.
		
		2. **Value Counts Text File:** The script calculates the value counts of predictions for each category ('straight', 'not_straight') and writes the results to a text file named "value_counts.txt" in the specified output folder.
	
	**Usage:**

    	To utilize the script, execute it from the command line as follows:

    	.. code:: bash

		python rotation_evaluation.py path_input_data.csv path_output_results_folder


segmentation_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~
This script is designed to evaluate the accuracy of segmentation results by calculating Intersection over Union (IoU) scores. It takes as input two CSV files containing ground truth and predicted coordinates, respectively.

	**Key Features:**

		1. **IoU Scores Calculation:** The script reads the ground truth and predicted coordinates from CSV files, calculates IoU scores for each corresponding pair of entries, and creates a new CSV file named "iou_scores.csv" containing the results.
		
		2. **Boxplot Generation:** A boxplot is created to visually represent the distribution of IoU scores. The resulting boxplot image is saved as "iou_box.jpg" in the specified output folder.
	
		3. **Barchart Generation:** A barchart is created to illustrate the class prediction distribution based on the calculated IoU scores. The resulting barchart image is saved as "class_pred.jpg" in the specified output folder.
	
	**Usage:**

    	To utilize the script, execute it from the command line as follows:

    	.. code:: bash

		segmentation_accuracy.py [-h] -g <ground_truth_coord> -p <predicted_coord> -r <results>

.. _gensim model: https://radimrehurek.com/gensim/models/word2vec.html
.. _high-dimensional data: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
.. _this article: https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510
