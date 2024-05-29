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
        "Operating System :: OS Independent",
    ],
    scripts = [script for script in glob.glob("scripts/*/*.py") if "__init__" not in script],
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=['any'],
    install_requires = [
        "cer",
        "deskew",
        "detecto",
        "google-cloud-vision",
        "jiwer",
        "kaleido",
        "matplotlib",
        "nltk",
        "numpy",
        "opencv-python",
        "pandas",
        "pillow",
        "plotly-express",
        "plotly",
        "pytesseract",
        "regex",
        "renku-sphinx-theme"
        "scikit-learn",
        "seaborn",
        "sphinx-rtd-theme",
        "sphinx",
        "tensorflow==2.15.0",
        "tesseract",
        "torch",
        "torchvision"
    ],
)
