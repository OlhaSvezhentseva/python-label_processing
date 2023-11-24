import unittest
import cv2
import numpy as np
from pathlib import Path

from label_processing.vision import VisionApi

class TestVisionApi(unittest.TestCase):
    """
    A test suite for the VisionApi class.
    """
    image_path: Path =  Path("../testdata/cropped_pictures/coll.mfn-berlin.de_u_115ff7__Preview_label_typed_1.jpg")
    global_vision = VisionApi.read_image(image_path, "test")
    
    
    def test_read_image(self):
        """
        Test the read_image method of VisionApi.

        This test checks if the read_image method correctly reads an image and initializes a VisionApi instance.
        """
        local_vision = VisionApi.read_image(self.image_path, "test")
        self.assertIsInstance(bytes, local_vision.image)
    
    def test_process_string(self):
        """
        Test the process_string method of VisionApi.

        This test checks if the process_string method correctly processes a test string.
        """
        test_string = "hello\nWorld\n"
        self.assertEqual(self.global_vision.process_string(test_string), "hello world")
    
    def test_vision_ocr(self):
        """
        Test the vision_ocr method of VisionApi.

        This test is a placeholder and can be extended to test the vision_ocr method when implemented.
        """
        pass