# Evaluation

## Description

This module aims at evaluation of separate modules of the pipeline.



## Structure
File `ocr_accuracy.py` performs evaluation of the OCR output by comparing it to the ground truth. 
For each transcript Levenshtein distance is calculated between its prediction and reference text.
It is calculated both on character and word level, resulting in 2 scores: CER (Character Error Rate) and 
WER (Word Error Rate).  

Both metrics indicate the amount of text that the applied model did not read correctly.
You can find more information about these metrics here:
 https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510

CER is normalized by the program and lies between 0 and 1, where 0 means, that predicted text is identical 
to the reference text.

WER is basically the number of errors divided by the total number of words (in ground truth).
So WER is not normalized and can be greater than 1, especially if the predicted text has more words than the 
ground truth, this is sometimes the case when additional nonsense words are addded during OCR.

To run the file use the following command:

    `python ocr_accuracy.py -g ground_truth.json -p predicted_transcripts.json -r results_folder`

Parameters:

-g (ground_truth): path to the ground truth dataset

-p (predicted_ocr): path to the predicted transcripts

-r (results, default = user's working directory): path to the target folder where the accuracy results are saved.

As a result `ocr_evaluation.csv` is saved in the desired directory, 
it contains an overwiew of each transcript (reference text and predicted one) as well as corresponding scores.
Besides, 2 violine plots representing the distribution of scores are saved in the folder.