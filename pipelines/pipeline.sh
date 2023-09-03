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

# Check if output dir exists and if not create it
output_dir="$1"
mkdir -p "$output_dir"
outlog="${output_dir}/out.log"
errlog="${output_dir}/err.log"

# Function to run a command and log its output
run_command() {
  local command="$1"
  local log_file="$2"
  $command >> "$log_file" 2>> "$log_file"
  if [ $? -ne 0 ]; then
    echo "Error running: $command" >> "$log_file"
  fi
}
echo "step 1: rotating pictures..."
# Create a directory for rotated pictures
rotated_dir="${output_dir}/rotated"
mkdir -p "$rotated_dir"
# Actual rotation
run_command "rotation.py -o '$rotated_dir' -i '$2'" "$outlog"
#TODO check if pictures exists in this direcory 

echo "step 2: performing image classification..."
typed_dir="${output_dir}/typed"
# Images are split into three directories -> 'handwritten', 'to_crop', 'typed'
# Only proceed with 'typed' for now
run_command "image_classifier.py -o '$output_dir' -j '$rotated_dir'" "$outlog"

echo "step 3: would be to split the pictures based on background color..."
#TODO

results_ocr="${output_dir}/ocr_preprocessed.json"
echo "step 4: performing OCR and saving resulting JSON in ${results_ocr}" 
run_command "tesseract_ocr.py -d '$typed_dir' -o '$output_dir'" "$outlog"

echo "step 5: postprocessing..."
run_command "process_ocr.py -j '$results_ocr' -o '$output_dir'" "$outlog"
postprocessed_json="$(realpath "${output_dir}/corrected_transcripts.json")"

echo "Pipeline finished. Postprocessed JSON: $postprocessed_json" 

