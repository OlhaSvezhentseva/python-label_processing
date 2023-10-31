import unittest
import torchvision
import cv2
import numpy as np

from label_processing.rotator import *

class RotatorTests(unittest.TestCase):
    image_path: Path =  Path("../testdata/cropped_pictures/coll.mfn-berlin.de_u_115ff7__Preview_label_typed_1.jpg")
    image = cv2.imread(str(image_path))
    
    
    def test_rotation_detetctor(self):
        efficientnet = torchvision.models.efficientnet_b0()
        rot_det = RotationDetector(efficientnet)
        rotated_img = rotation(rot_det, self.image, TorchConfig)
        self.assertIsInstance(rotated_img, np.ndarray)
        