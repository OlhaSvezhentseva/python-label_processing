import unittest
import cv2
import numpy as np

from label_processing.text_recognition import ImageProcessor, Tesseract, Threshmode
from pathlib import Path


class TestImageProcessor(unittest.TestCase):
    image_path: Path =  Path("../testdata/cropped_pictures/coll.mfn-berlin.de_u_115ff7__Preview_label_typed_1.jpg")
    image = cv2.imread(str(image_path))
    
    
    def test_construcor_image_from_image(self):
        preprocessor = ImageProcessor(self.image, self.image_path)
        self.assertIsInstance(preprocessor.image, np.ndarray)
        self.assertEqual(preprocessor.path, self.image_path)
        
    
    def test_image_processor_from_path(self):
        preprocessor = ImageProcessor.read_image(self.image_path)
        self.assertIsInstance(preprocessor.image, np.ndarray)
        self.assertEqual(preprocessor.path, self.image_path)
    
    def test_image_processor_preprocessing(self):
        preprocessor = ImageProcessor.read_image(self.image_path)
        preprocessor = preprocessor.preprocessing(Threshmode.OTSU)
        self.assertIsInstance(preprocessor.image, np.ndarray)
        
    #TODO Test if pictures are actually different 
    
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
    
    
        
