# Import third-party libraries
import unittest
from pathlib import Path
import detecto

# Import the necessary module from the 'label_processing' module package
from label_processing.segmentation_cropping import *


class TestSegmentationCropping(unittest.TestCase):
    """
    A test case for the Segmentation and Cropping functionality.

    Attributes:
        path_to_model (str): The file path to the segmentation model.
        jpg_path (Path): The file path to a sample JPG image for testing.
        label_predictor (PredictLabel): An instance of the PredictLabel class for testing.
    """
    path_to_model = "../../models/model_segmentation_label.pth"
    jpg_path: Path =  Path("../testdata/uncropped/coll.mfn-berlin.de_u_43acf9__label.jpg")
    label_predictor = PredictLabel(path_to_model, ["label"], jpg_path)

    
    def test_predict_label_constructor(self):
        """
        Test the constructor of the PredictLabel class.

        Ensures that the PredictLabel instance is created with the correct attributes.
        """
        label_predictor = PredictLabel(self.path_to_model, ["label"], self.jpg_path)
        self.assertIsInstance(label_predictor.jpg_path, Path)
        self.assertIsInstance(label_predictor.model, detecto.core.Model)

    def test_predict_label_constructor_2(self):
        """
        Test an alternative constructor of the PredictLabel class.

        Ensures that the PredictLabel instance is created correctly when the image path
        is set separately.
        """
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        label_predictor.jpg_path = self.jpg_path
        self.assertIsInstance(label_predictor.jpg_path, Path)

    def test_class_prediction(self):
        """
        Test the class prediction method of the PredictLabel class.

        Verifies that the class prediction method returns a DataFrame with the predicted entries.
        """
        entries = self.label_predictor.class_prediction(self.jpg_path)
        self.assertIsInstance(entries, pd.DataFrame)

        # self.assertIsInstance(entries, list)
        # [self.assertIsInstance(entry, dict) for entry in entries]

    def test_class_prediction_parallel(self):
        """
        Test the parallel class prediction method.

        Verifies that the parallel class prediction method returns a DataFrame with the predicted entries,
        and the DataFrame has the correct number of columns.
        """
        df = prediction_parallel("../testdata/uncropped", self.label_predictor, 1)
        # print(df["score"])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 7)

    def test_number_of_labels_detected_single_image(self):
        """
        Test the number of labels detected in a single image.

        Verifies that the correct number of labels are detected in a single image.
        """
        df = self.label_predictor.class_prediction(self.jpg_path)
        self.assertEqual(len(df), 3)

    def test_number_of_labels_detected_image_folder(self):
        """
        Test the number of labels detected in an image folder.

        Verifies that the correct number of labels are detected in a folder of images.
        """
        df = prediction_parallel("../testdata/uncropped", self.label_predictor, 1)
        self.assertEqual(len(df), 16)

    def test_threshold(self):
        """
        Test the threshold cleaning functionality.

        Verifies that the cleaning function removes entries below the specified threshold.
        """
        df = prediction_parallel("../testdata/uncropped", self.label_predictor, 1)
        # print(df["score"])
        clean_df = clean_predictions(Path("../testdata/uncropped"), df,1.0)
        self.assertEqual(len(clean_df),0)

    def test_crops(self):
        """
        Test the creation of crops from predictions.

        Verifies that crop files are created for each prediction in the specified output directory.
        """
        df = prediction_parallel("../testdata/uncropped", self.label_predictor, 1)
        # no cleaning
        create_crops(Path("../testdata/uncropped"), df, out_dir=Path("check_crops"))
        crop_files = glob.glob(os.path.join(Path("check_crops/uncropped_cropped"),'*.jpg'))
        self.assertEqual(len(df),len(crop_files))
    