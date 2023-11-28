# Import third-party libraries
import unittest
import os
from pathlib import Path

# Import the necessary module from the 'label_processing' module package
from label_processing.tensorflow_classifier import *


class TestTFClassifier(unittest.TestCase):
    """
    A test suite for the TensorFlow classifier module.
    """
    model = get_model("../../models/model_classifier")
    classes = ['handwritten', 'to_crop', 'typed']
    outdir = "../testdata/output"
    jpg_dir = "../testdata/cropped_pictures"
    df = class_prediction(model, classes,
                          jpg_dir, outdir)

    def test_class_prediction_normal(self):
        """
        Test the normal case of class prediction.

        This test checks if the class prediction function creates the expected output CSV file
        and if the number of rows in the DataFrame matches the number of images in the input directory.
        """
        self.assertTrue(os.path.exists("../testdata/output/cropped_pictures_prediction_classifer.csv"))
        self.assertEqual(len(self.df.index), len(os.listdir(self.jpg_dir)))

    def test_class_prediction_empty(self):
        """
        Test the case of class prediction with an empty directory.

        This test checks if the function raises a FileNotFoundError when given an empty input directory.
        """
        empty_dir = "../testdata/empty_dir"
        Path(empty_dir).mkdir(parents=True, exist_ok=True)
        self.assertRaises(FileNotFoundError, class_prediction, self.model, self.classes, empty_dir, self.outdir)

    def test_create_dirs(self):
        """
        Test the creation of directories based on class predictions.

        This test checks if the create_dirs function creates directories corresponding to the predicted classes.
        """
        test_dir = os.path.join(self.outdir, "temp_dir")
        os.mkdir(test_dir)
        create_dirs(self.df, test_dir)
        for model_class in self.classes:
            self.assertTrue(model_class in os.listdir(test_dir))
            os.rmdir(os.path.join(test_dir, model_class))
        os.rmdir(test_dir)

    def test_make_file_name(self):
        """
        Test the generation of a new file name.

        This test checks if the make_file_name function correctly generates a new file name.
        """
        filename = "CASENT0179609_L.jpg"
        filename_stem = Path(filename).stem
        new_name = make_file_name(filename_stem, self.classes[0], 1)
        self.assertEqual(new_name, f"CASENT0179609_L_label_{self.classes[0]}_1.jpg")

    def test_filter_pictures(self):
        """
        Test the filtering of pictures based on class predictions.

        This test checks if the filter_pictures function filters pictures correctly based on class predictions.
        """
        filter_pictures(self.jpg_dir, self.df, self.outdir)
        picture_count = 0
        for model_class in self.classes:
            picture_count += len(os.listdir(os.path.join(self.outdir, model_class)))

        self.assertEqual(len(os.listdir(self.jpg_dir)), picture_count)