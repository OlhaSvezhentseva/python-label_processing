# Postprocessing

## Description

In case transcripts were generated automatically they may contain spelling mistakes or extra symbols that were
incorrectly added during OCR. Possible reasons for that include paper imperfections (scratches, holes, etc,), 
poor image quality, or lack of contrast between the text and the background.

These issues are often tackled in the image pre-processing stage, but sometimes postprocessing is still needed.



## Structure
File `process_ocr.py` is responsible for filtering transcripts according to 4 categories:

1. nuris
2. empty transcripts
3. nonsense transcripts
4. plausible transcripts

Plausible transcripts are corrected by removing non-ASCII and non-alpha-numerical symbols
and the transcripts are saved as `corrected_transcripts.json`.


To run the file make sure you are in the folder "postprocessing" and use the following command:

    `python process_ocr.py -j transcripts.json -o postprocessed_transcripts`
 
 Parameters:
 -j (json): is path to the json file  with transcripts (OCR output)
 
 -o (out_dir): is  the path to the output folder
 
At the end one json file per category is saved in the output folder.


File `fix_spelling.py` checks if there are any spelling mistakes and tries to fix them.
This is achieved by calculating Edit distance between words that appear fewer than 2 times with the 20 most frequent
words in the transcripts.
If the Edit distance is lower/equal than a particular threshold, the word is substituted with a frequent word under
the assumption that this is the same word spelled correctly.


To run the file make sure you are in the folder "postprocessing" and use the following command:

    `python fix_spelling.py --transcripts corrected_transcripts.json --freq 20 --dist 0.34`


Parameters:

--transcripts: is the file you want correct transcripts from. It makes sense to use
corrected_transcripts.json that was created in the previous step (filter.py).

--freq: is the number of the most frequent words that low-frequent words will be compared to.

--dist: threshold for Edit distance. Distance less/equal than this value will be considered to be a small one,
so that the low-frequence word can be changed.

--~~~~voc: (optional, per default False): path to the vocabulary
When not specified, the script creates a vocabulary with each unique word and its count 
and saves it as vocabulary.csv. If there is a vocabulary already, you may pass it to save time. 

The corrected transcripts are saved as "spell_checked_transcripts.json"