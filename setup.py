from setuptools import setup


with open ("README.rst", "r") as fh:
    long_description = fh.read()

setup(  
    name='label_processing',
    version='1.1',
    description='Package for label processing',
    py_modules=["redundancy", "segmentation_cropping", "text_recognition", "vision", "utils"],
    package_dir={'': 'label_processing'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "operatzing System :: OS Independent",
    ],
    scripts=["scripts/crop_seg.py",
             "scripts/tesseract_ocr.py",
             "scripts/vision_api.py",
             "scripts/label_redundancy.py"],
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
        "google"
    ],
)
