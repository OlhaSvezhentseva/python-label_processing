from setuptools import setup


with open ("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='label_processing',
    version='1.1',
    description='Package for specimen label information extraction and processing',
    packages=["label_processing", "label_evaluation", "label_postprocessing"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "operatzing System :: OS Independent",
    ],
    scripts=["scripts/crop_seg.py",
             "scripts/tesseract_ocr.py",
             "scripts/vision_api.py",
             "scripts/label_redundancy.py",
             "scripts/ocr_accuracy.py",
             "scripts/segmentation_accuracy.py",
             "scripts/cluster_id.py",
             "scripts/postprocessing_nuri.py",
             "scripts/background_color.py",
             "scripts/rotation.py",
             "scripts/process_ocr.py",
             "scripts/filter.py",
             "scripts/fix_spelling.py",
             "scripts/image_classifier.py", 
             "scripts/evaluation_classifier.py",
             "scripts/cluster_visualisation.py",
             "scripts/rotation_evaluation.py"
             "pipelines/pipeline.sh"],
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
        "pyzbar",
        "torchvision",
        "regex",
        "nltk",
        "tensorflow",
        "scikit-learn",
        "plotly-express"
    ],
)
