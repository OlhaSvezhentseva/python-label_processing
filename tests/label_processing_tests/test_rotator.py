import unittest
import os

# Import the necessary module from the 'label_processing' module package
from label_processing.label_rotation_module import *

class TestRotateImage(unittest.TestCase):
    def test_rotate_image(self):
        img_path = "/Users/Margot/Desktop/test/d2d306bf-449f-4af8-8c12-b73dba69e37b_label_front_0001_label.jpg"
        output_dir = '/Users/Margot/Desktop/output'
        angle = 1 
        
        # Call the function to test
        rotate_image(img_path, angle, output_dir)
        
        # Construct the rotated image path
        rotated_img_path = os.path.join(output_dir, os.path.basename(img_path))
        
        # Check if the rotated image is created in the output directory
        self.assertTrue(os.path.exists(rotated_img_path), "Rotated image file not found")

if __name__ == '__main__':
    unittest.main()
