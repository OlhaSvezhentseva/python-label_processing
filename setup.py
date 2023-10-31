from setuptools import setup
import glob

with open ("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='Entomological Label Information Extraction',
    version='1.1',
    description='Package for specimen label information extraction and processing',
    packages=["label_processing", "label_evaluation", "label_postprocessing"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "operatzing System :: OS Independent",
    ],
    scripts=glob.glob("scripts/*/*.py"),
    #include_package_data=True, #include the static data specified in the MANIFEST.in
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=['any'],
    install_requires = [
        "numpy",
        "pandas",
        "torch",
        "detecto",
        "opencv-python",
        "detecto",
        "pytesseract",
        "tesseract",
        "pillow",
        "google-cloud",
        "deskew",
        "seaborn",
        "matplotlib",
        "jiwer",
        "cer",
        "plotly",
        "kaleido",
        "torchvision",
        "regex",
        "nltk",
        "tensorflow",
        "scikit-learn",
        "plotly-express"
    ],
)
