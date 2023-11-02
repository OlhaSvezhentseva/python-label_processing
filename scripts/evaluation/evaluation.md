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





File `cluster_visualisation.py` plots clusters using word embeddings and 
saves the plot as an HTML-link.

Word embeddings can be built either using the ground truth data, or the predicted transcripts.
With the help of a pretrained gensim model (https://radimrehurek.com/gensim/models/word2vec.html) each word in the label gets represented by a vector.
Then, the vectors are normalized, so that each label is represented only by one vector.
It is possible to pass vector dimensions to a gensim model, 
currently 100-dimensional space is used. With the help of tsne 
(a tool to visualize high-dimensional data, https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
) each label is plotted on a 2-dimensional space. 

Besides, each dot has a colour representing the cluster it was assigned to. So, 
dots with the same colour refer to one cluster. 

In that way we can see if word embedding's predictions of label follow the same direction
as the results of clustering algorithms. 

HTML plot enables hovering over the dots and seeing the transcript of the label.
This makes possible to check at once if neighbouring labels (dots)
have similar texts. As there are often too many clusters to be represented clearly on the plot, 
it is possible to pass a parameter representing the minimal size of cluster that will be plotted,
that is the number of labels that a cluster must have in order to be plotted.
It allows us to look at the bigger clusters and study them. 


To run the file use the following command:

    `python visualisation_new.py -c clusters.json -s 10`

Parameters:

-c (cluster_json): path to the clustering output file

-gt (ground_truth, default = None): path to the ground truth file

-o (ouptut_dir, default = user's working directory): path to the target folder where plot will be saved

-s (cluster_size, default = 1): the number of labels a cluster must have in order to be plotted,
per default all clusters are plotted.

