Postprocessing
==============

.. contents ::

Description
-----------
If transcripts are generated automatically, they might include spelling errors or additional symbols introduced during OCR inaccuracies. Factors such as paper imperfections (scratches, holes, etc.), suboptimal image quality, or insufficient contrast between the text and background could contribute to these issues. 
While the image pre-processing stage commonly addresses these challenges, there are instances where post-processing remains necessary.
Key post-processing steps encompass:

1. Transcript Categorization

2. Vocabulary Generation and Spelling Correction


Scripts
-------
For usage information, run any of these scripts with the option --help.


process.py
~~~~~~~~~~
The script, `process.py`, categorizes transcripts into four groups: "nuris," empty transcripts, nonsense transcripts, and plausible transcripts. 
Plausible transcripts undergo correction by eliminating non-ASCII and non-alphanumeric symbols before being saved as `corrected_transcripts.json`.
At the end one json file per category is saved in the output folder.


	**Key Features:**

		1. **Token Length & Plausibility:** Process OCR output, identify Nuri and empty labels, and correct plausible labels. Remove non-ASCII, non-alphanumeric, and pipe characters.

		2. **Saving Transcripts:** Saves categorized transcripts as CSV and JSON files in the specified output directory. Resulting files include "nuris.csv," "empty_transcripts.csv," "plausible_transcripts.json," and "corrected_transcripts.json."


	**Usage:**

		To run the file make sure you are in the folder "postprocessing" and use the following command:

		.. code:: bash

			process.py [-h] -j <ocr-json> -o <out-dir>
	

spelling.py
~~~~~~~~~~~
The script `spelling.py` verifies and corrects spelling mistakes by computing the Edit distance between words occurring less than 2 times and the 20 most common words in the transcripts. 
If the Edit distance falls below or equals a specified threshold, the script replaces the word with a frequently appearing word. This process assumes that the frequently used word represents the correctly spelled version of the word in question.


	**Key Features:**

		1. **Extract:** Extracts unique words from OCR-transcribed text. Extract the words with the highest occurrence from a given vocabulary.

		2. **Data Processing:** Tokenizes text labels, filters tokens based on length and content criteria.

		3. **Fix Spelling:** Implements a function to correct spelling mistakes in transcripts based on provided vocabulary and specified parameters. Uses the JiWER library for calculating the character error rate (CER) between words to determine potential corrections. Saves corrected transcripts in a JSON file named `spell_checked_transcripts.json`.


	**Parameters:**

	--transcripts: is the file you want correct transcripts from. It makes sense to use `corrected_transcripts.json` that was created in the previous step (`filter.py`).

	--freq: is the number of the most frequent words that low-frequent words will be compared to.

	--dist: threshold for Edit distance. Distance less/equal than this value will be considered to be a small one, so that the low-frequence word can be changed.

	--voc: (optional, per default False): path to the vocabulary.
	When no vocabulary is explicitly provided, the script generates a vocabulary containing each unique word along with its respective count, saving the result as `vocabulary.csv`. If a vocabulary is already available, it can be passed as input to optimize processing time. 
	The corrected transcripts are then stored as `spell_checked_transcripts.json`.


	**Usage:**

    	To run the file make sure you are in the folder "postprocessing" and use the following command (example):

   		.. code:: bash

	  		fix_spelling.py --transcripts corrected_transcripts.json --freq 20 --dist 0.34