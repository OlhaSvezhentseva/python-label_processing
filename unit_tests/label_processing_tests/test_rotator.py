import unittest
import os

# Import the necessary module from the 'label_processing' module package
from label_processing.label_rotation_module import *

class TestRotateImage(unittest.TestCase):
    def test_rotate_image(self):
        filename = "BLF1542(10)-6_L_label_handwritten_1.jpg"
        img_path = f"../testdata/cropped_pictures/{filename}"
        output_dir = '../testdata//output'
        angle = 1 
        
        # Call the function to test
        rotate_image(img_path, angle, output_dir)
        
        # Construct the rotated image path
        rotated_img_path = os.path.join(output_dir, os.path.basename(img_path))
        
        # Check if the rotated image is created in the output directory
        self.assertTrue(os.path.exists(rotated_img_path), "Rotated image file not found")

if __name__ == '__main__':
    unittest.main()
