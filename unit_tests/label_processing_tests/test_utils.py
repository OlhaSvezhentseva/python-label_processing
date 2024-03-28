# Import third-party libraries
import unittest
from pathlib import Path

# Import the necessary module from the 'label_processing' module package
from label_processing.utils import * 

class TestUtils(unittest.TestCase):
    """
    A test suite for the utility functions in the label_processing module.
    """
    nuri_transcript = {"ID": "coll.mfn-berlin.de_u_43acfb_label_box_2.jpg", "text": "http   : //cl. \n mfn-berli.de\n /43acfb"}
    link = "http://coll.mfn-berlin.de/u/43acfb"
    
    
    def test_check_dir(self):
        """
        Test the check_dir function.

        This test checks if the check_dir function correctly handles directories with and without JPG files.
        """
        dir_with_jpgs = Path("../testdata/cropped_pictures")
        dir_without = Path("../testdata")
        self.assertIsNone(check_dir(dir_with_jpgs))
        with self.assertRaises(FileNotFoundError):
            check_dir(dir_without)
    
    def test_generate_filename(self):
        """
        Test the generate_filename function.

        This test checks if the generate_filename function correctly generates filenames for files and directories.
        """
        mock_path_file = "../testdata/cropped_pictures/CASENT0179609_L_label_typed_1.jpg"
        mock_path_dir = "../testdata/output"
        extension = ".jpg"
        appendix = "_test"
        filename_file  = generate_filename(mock_path_file, appendix, extension)
        filename_dir  = generate_filename(mock_path_dir, appendix, extension)
        self.assertEqual(filename_file, "CASENT0179609_L_label_typed_1_test.jpg")
        self.assertEqual(filename_dir, "output_test.jpg")

    def test_check_text(self):
        """
        Test the check_text function.

        This test checks if the check_text function correctly identifies certain patterns in text.
        """
        pattern_1 = "http:/hello/world"
        pattern_2 = "CASENTO 56396"
        self.assertTrue(check_text(pattern_1))
        self.assertFalse(check_text(pattern_2))
    
    def test_replace_nuri_actual(self):
        """
        Test the replace_nuri function with a transcript containing a Nuri link.

        This test checks if the replace_nuri function correctly replaces the Nuri link in the transcript.
        """
        replaced = replace_nuri(self.nuri_transcript)
        self.assertEqual(replaced["text"], self.link)
    
    def test_replace_nuri_no_nuri(self):
        """
        Test the replace_nuri function with a transcript containing no Nuri link.

        This test checks if the replace_nuri function leaves the text unchanged when no Nuri link is present.
        """
        no_nuri_transcript = {"ID": "coll.mfn-berlin.de_u_43acfb_label_box_2.jpg",
                           "text": "Somewhere in Kasachstan"}
        replaced = replace_nuri(no_nuri_transcript)
        self.assertEqual(replaced["text"], no_nuri_transcript["text"])
        
    