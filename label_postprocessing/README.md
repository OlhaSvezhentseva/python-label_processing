
# Label Postprocessing 
## Description

The aim of postprocessing phase is to filter the ocr output and make corrections.


## Structure
File `filter.py` is responsible for filtering the ocr ouput according to 4 categories:
nuris, empty transcripts, plausible output, nonsense output.
Plausible output is corrected using regular expressions and is saved as `corrected_transcripts.json`.

 File `vocabulary.py` extracts unique words from the trasnscripts and counts their occurrences.
 
 File `fix_spelling.py` checks if there are any spelling mistakes and fixes them.
 This is achieved by calculating Edit distance between words that appear fewer than 2 times with the 
 20 most frequent words in the transcript. If the Edit distance is lower/equal than a particular threshold,
 the word is substituted with a frequent word under the assumption that this is the same word
 spelled correctly.
 

# Usage
1. First run `filter.py` to filter ocr outputs into different categories.

Example:


    `python filter.py  --ocr_output ocr_pytesseract_all.json`

2. Run `fix_spelling.py` to extract vocabulary (optionally) of the transcripts and correct spelling mistakes.
 
Example:
   
     `python fix_spelling.py --transcripts corrected_transcripts.json --freq 20 --dist 0.34`
     
transcripts: is the file you want correct transcripts from. It makes sense to use 
`corrected_transcripts.json` that was created in the previous step.

freq: is the number of the most frequent words that low-frequent words will be compared to.

dist: threshold for Edit distance. Distance less/equal than this value will be considered
to be a small one, so that the low-frequent word can be changed.

3. If you already have `vocabulary.csv` file and it should not be generated again,
you may specify it:


    `python fix_spelling.py --transcripts corrected_transcripts.json --freq 20 --dist 0.34 --voc vocabulary.csv`

## Contact
Olha Svezhentseva <Olha.Svezhentseva@mfn.berlin>

