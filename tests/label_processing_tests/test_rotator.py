# Import third-party libraries
import unittest
import torchvision
import cv2
import numpy as np

# Import the necessary module from the 'label_processing' module package
from old.modules.rotator import *

class RotatorTests(unittest.TestCase):
    """
    A test suite for the Rotator module.
    """
    image_path: Path =  Path("../testdata/cropped_pictures/coll.mfn-berlin.de_u_115ff7__Preview_label_typed_1.jpg")
    image = cv2.imread(str(image_path))
    
    
    def test_rotation_detetctor(self):
        """
        Test the rotation detector with an EfficientNet model.

        This test checks if the RotationDetector class correctly rotates an image using an EfficientNet model.
        """
        efficientnet = torchvision.models.efficientnet_b0()
        rot_det = RotationDetector(efficientnet)
        rotated_img = rotation(rot_det, self.image, TorchConfig)
        self.assertIsInstance(rotated_img, np.ndarray)
    