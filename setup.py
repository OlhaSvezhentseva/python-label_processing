from setuptools import setup


with open ("README.md", "r") as fh:
    long_description = fh.read()

setup(  
    name='label_processing',
    version='0.0.1',
    description='Package for label processing',
    py_modules=["apply_model"],
    package_dir={'': 'label_processing'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "operatzing System :: OS Independent",
    ],
    scripts=["scripts/create_crops.py",
             "scripts/ocr_tess.py"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=['any'],
    install_requires = [
        "numpy",
        "pandas",
        "torch",
        "detecto",
        "opencv-python",
        "detecto"
    ],
    
)
