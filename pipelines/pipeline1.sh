#!/bin/bash
#shell script as a first try to connect the components of the pipeline
set -e #Exit immediately if a command exits with a non-zero status.
set -x #Print commands and their arguments as they are executed.
set -u #Treat unset variables as an error when substituting.


if [ "$1" == "-h" ]; then
  echo "usage:"
  echo "$0 [-h] <output dir> <dirname cropped files>"
  exit 0
fi


#check if output dir exists and if not create it
mkdir -p "$1"
outlog="${1}/out.log"
errlog="${1}/err.log"


echo "step 1: rotating pictures..."
#create directory for rotated pictures
rotated_dir="${1}/rotated"
mkdir -p "$rotated_dir"
#acutual rotation
rotation.py -o "$rotated_dir" -i "$2" > "$outlog" 2> "$errlog"
#TODO check if pictures exists in this direcory 

echo "step 2: performing image classification..."
typed_dir="${1}/typed"
image_classifier.py -o "$1" -j "$rotated_dir" > "$outlog" 2> "$errlog"
#images are split into thre directories -> 'handwritten', 'to_crop', 'typed'
#only proceed with the typed for now

echo "step 3: would be to split the pictures based on background color..."
#TODO

results_ocr="${1}/ocr_preprocessed.json"
echo "step 4 performing ocr and saving resulting json in ${results_ocr}" 
tesseract_ocr.py -d "$typed_dir" -o "$1" > "$outlog" 2> "$errlog"

echo "step 5: postprocessing..."
process_ocr.py -j "$results_ocr" -o "$1" > "$outlog" 2> "$errlog"
postprocecessed_json=$(realpath "${1}/corrected_transcripts.json")

printf "pipeline finished postprocecessed json in %s", "$postprocecessed_json" 

