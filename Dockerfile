FROM python:3.12.3-bookworm
WORKDIR .
COPY . .
RUN apt-get update
RUN yes | apt-get install tesseract-ocr
RUN yes | apt install libtesseract-dev
RUN yes | apt-get install python3-opencv
RUN pip3 install .
#CMD ["pip", "list"]
CMD ["classifiers.py", "-m", "2", "-j", "unit_tests/testdata/cropped_pictures", "-o", "unit_tests/testdata/output"]
