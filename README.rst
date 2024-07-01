Collection Mining - Entomological Label Information Extraction
==============================================================

*A Python package for the Berlin Natural History Museum*

.. contents ::

Introduction
------------
This package is a comprehensive solution that seamlessly integrates a range of AI models and functional components, meticulously designed to facilitate the detection, classification, rotation, OCR, and clustering of entomology specimen labels. 
It serves as the foundational framework for the initial steps of information extraction, working in conjunction with the python-mfnb clustering package, which handles clustering in subsequent stages.


Features
--------
- **AI Models Integration:** Three specialized TensorFlow classifiers for various label styles.
- **Text Extraction:** Modules for label classification, image preprocessing, and OCR postprocessing.
- **Versatility:** Compatible with both Pytesseract and Google Vision OCR.
- **Efficiency:** Streamlines entomology label processing, saving time and resources.

For more detailed insights into the methodology, please refer to the forthcoming paper: Margot Belot et al. (in preparation), *A Semi-Automated Pipeline for High Throughput Information Extraction of Insect Specimen Labels*.


Installation
------------
1. Create a python environment with python 3.10 (e.g. with conda). On newer python versions some of the dependencies might not work.
   This can be done with conda like this:

   ``conda create --name mfnb python=3.10``

2. Clone the repository:

   ``git clone https://code.naturkundemuseum.berlin/collection-mining/label_processing.git``

3. Navigate to the repository directory:

   ``cd <path to python-label_processing-main>``
   
4. Install with pip, which will automatically fetch the requirements.

   ``pip install .``

5. If it is intended to use tesseract, it needs to be installed. This can be done via a package manager:
   
   on ubuntu/debian:

   ``sudo apt install tesseract-ocr``

   or on Mac OS:
   
   ``brew install tesseract``


Input preparation
-----------------
The modules are best to be performed on **JPEG images** of labels from entomology databases such as:
   
   - `AntWeb`_
   - `Bees&Bytes`_
   - LEP_PHIL - pictures of specimens from the Philippines (by Théo Leger)
   - `Atlas of Living Australia`_


In terms of data acquisition, the following **standards** are recommended to optimize the outputs:

- Standardized, uniform picture quality (macro photography, .jpg format, 300 DPI resolution).
- Clearly separated labels without overlapping.
- Horizontal text alignment.
- Exclusion of specimens in the picture.
- Consistent label placement and orientation across pictures.
- Preferably black background, but white is acceptable.

.. _AntWeb: https://www.antweb.org/
.. _Bees&Bytes: https://www.zooniverse.org/projects/mfnberlin/bees-and-bytes  
.. _Atlas of Living Australia: https://www.ala.org.au/


Setting up Google Cloud Vision API and getting credentials
----------------------------------------------------------
- In order to use the google API you need to create a Google account and set it up for Vision.
- How to setup your Google Cloud Vision is explained `on the website`_.
- You need to retrieve your credentials json (everything is explained in the provided link).
- The credentials json file should then be provided as an input in the `vision.py` script.


Installing Pytesseract for MacOS
--------------------------------
Informations about Pytesseract can be found `here`_ or `this website`_.
To install Pytesseract with Homebrew, first install `it`_ and follow the `steps`_.

.. _on the website: https://cloud.google.com/vision/docs/setup
.. _here: https://pypi.org/project/pytesseract/
.. _this website: https://tesseract-ocr.github.io/tessdoc/Installation.html
.. _it: https://brew.sh/
.. _steps: https://formulae.brew.sh/formula/tesseract


Installing zbar for MacOS and Linux
-----------------------------------
To use the more powerful qr-code reading function of zbar additional dependencies
have to be installed (only for Mac OS and Linux. On Windows they come with the 
Python DLLs) These can be installed via the command line with the following
commands:

Mac OS:

``brew install zbar``

Linux:

``sudo apt-get install libzbar0``


Contacts
--------

Margot Belot margot.belot@mfn.berlin

Olha Svezhentseva Olha.Svezhentseva@mfn.berlin

Leonardo Preuss preuss.leonardo@mfn.berlin

