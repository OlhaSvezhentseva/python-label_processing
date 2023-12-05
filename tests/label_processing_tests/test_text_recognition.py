# Import third-party libraries
import unittest
import cv2
import numpy as np
from pathlib import Path

# Import the necessary module from the 'label_processing' module package
from label_processing.text_recognition import ImageProcessor, Tesseract, Threshmode


class TestImageProcessor(unittest.TestCase):
    """
    A test suite for the ImageProcessor class.
    """
    image_path: Path =  Path("../testdata/cropped_pictures/coll.mfn-berlin.de_u_115ff7__Preview_label_typed_1.jpg")
    image = cv2.imread(str(image_path))
    
    
    def test_construcor_image_from_image(self):
        """
        Test the constructor of ImageProcessor with an image provided directly.

        This test checks if the ImageProcessor instance is correctly constructed with an image and a path.
        """
        preprocessor = ImageProcessor(self.image, self.image_path)
        self.assertIsInstance(preprocessor.image, np.ndarray)
        self.assertEqual(preprocessor.path, self.image_path)
        
    
    def test_image_processor_from_path(self):
        """
        Test creating an ImageProcessor instance from an image path.

        This test checks if the ImageProcessor instance is correctly created from an image path.
        """
        preprocessor = ImageProcessor.read_image(self.image_path)
        self.assertIsInstance(preprocessor.image, np.ndarray)
        self.assertEqual(preprocessor.path, self.image_path)
    
    def test_image_processor_preprocessing(self):
        """
        Test the preprocessing method of ImageProcessor.

        This test checks if the preprocessing method of ImageProcessor works with the specified threshold mode.
        """
        preprocessor = ImageProcessor.read_image(self.image_path)
        preprocessor = preprocessor.preprocessing(Threshmode.OTSU)
        self.assertIsInstance(preprocessor.image, np.ndarray)
        
    def test_different_pictures(self):
        """
        test if the preprocessed picture is different from the original picture
        """
        preprocessor = ImageProcessor.read_image(self.image_path)
        preprocessor = preprocessor.preprocessing(Threshmode.OTSU)
        if self.image.shape == preprocessor.image.shape:
            self.assertFalse(np.allclose(self.image, preprocessor.image))

    
    def test_save_image(self):
        preprocessor = ImageProcessor.read_image(self.image_path)
        path = "../testdata/output"
        expected_path = Path(f"{path}/{self.image_path.name}")
        preprocessor.save_image(path)
        self.assertTrue(expected_path.exists())
    
    def test_qr_code_reader(self):
        preprocessor = ImageProcessor.read_image(self.image_path)
        value = preprocessor.read_qr_code()
        self.assertTrue(value == None or isinstance(value, str))
    

class TestTesseract(unittest.TestCase):
    image_path: Path =  Path("../testdata/cropped_pictures/coll.mfn-berlin.de_u_115ff7__Preview_label_typed_1.jpg")
    image = cv2.imread(str(image_path))
    
    def test_constructor_no_image(self):
        tesseract_wrapper = Tesseract()
        self.assertIsNone(tesseract_wrapper.image)
        tesseract_wrapper.image = self.image
        self.assertIsInstance(tesseract_wrapper.image, np.ndarray)
    
    def test_constructor_image(self):
        tesseract_wrapper = Tesseract(image = self.image)
        self.assertIsInstance(tesseract_wrapper.image, np.ndarray)
    
    def test_image_to_string(self):
        preprocessor = ImageProcessor(self.image, self.image_path)
        tesseract_wrapper = Tesseract(image = preprocessor)
        result = tesseract_wrapper.image_to_string()
        self.assertIsInstance(result["text"], str)
    
    
        
