#!/bin/bash
#shell script as a first try to connect the components of the pipeline
set -e #Exit immediately if a command exits with a non-zero status.
#set -x #Print commands and their arguments as they are executed.
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


echo "step 1: performing image classification..."
#typed_dir="${1}/typed"
image_classifier.py -o "$1" -j "$2" > "$outlog" 2> "$errlog"
#images are split into thre directories -> 'handwritten', 'to_crop', 'typed'
#only proceed with the typed for now
#spd-say 'image classification done'
say "image classification done"

echo "step 2: rotating pictures..."
#create directory for rotated pictures
rotated_dir="${1}/rotated"
input="${1}/typed"
mkdir -p "$rotated_dir"
#acutual rotation
rotation.py -o "$rotated_dir" -i "$input" > "$outlog" 2> "$errlog"
#TODO check if pictures exists in this direcory 
#spd-say 'pictures are rotated'
say "pictures are rotated"


echo "step 3: would be to split the pictures based on background color..."
#TODO

results_ocr="${1}/ocr_preprocessed.json"
echo "step 4: performing ocr and saving resulting json in ${results_ocr}" 
tesseract_ocr.py -d "$rotated_dir" -o "$1" > "$outlog" 2> "$errlog"
#spd-say 'ocr finished'
say "ocr finished"
echo "step 5: postprocessing..."
process_ocr.py -j "$results_ocr" -o "$1" > "$outlog" 2> "$errlog"
postprocecessed_json=$(realpath "${1}/corrected_transcripts.json")
#spd-say 'post processing done'
corrected_transcripts="${1}/corrected_transcripts.json"
printf "postprocecessed json in %s", "$postprocecessed_json" 
say "post processing done"

echo "step 6: calculate redundancy in ${corrected_transcripts}"
label_redundancy.py -d "$corrected_transcripts" -o "$1" > "$outlog" 2> "$errlog"
redundancy=$(realpath "${1}/percentage_red.txt")
#spd-say 'post processing done'
say "calculation redundancy done"
printf "pipeline finished redundancy json in %s", "$redundancy" 
say "Great job! The pipeline is ready"
sleep 2; 
#spd-say -t female1 -w 'Great job! The pipeline is ready'
#spd-say -t -w 'Nice'
