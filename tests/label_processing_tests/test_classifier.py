import unittest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from label_processing.tensorflow_classifier import *

class TestTFClassifier(unittest.TestCase):
    
    model = get_model("../../models/model_classifier")
    classes =  ['handwritten', 'to_crop', 'typed']
    outdir = "../testdata/output"
    jpg_dir = "../testdata/cropped_pictures"
    df = class_prediction(model,classes,
                              jpg_dir, outdir)
    
    def test_class_prediction_normal(self):
        self.assertTrue(os.path.exists("../testdata/output/cropped_pictures_prediction_classifer.csv"))
        self.assertEqual(len(self.df.index), len(os.listdir(self.jpg_dir)))
    
    def test_class_prediction_empty(self):
        empty_dir  = "../testdata/empty_dir"
        Path(empty_dir).mkdir(parents=True, exist_ok=True)
        self.assertRaises(FileNotFoundError, class_prediction, self.model, self.classes, empty_dir, self.outdir)
            
    def test_create_dirs(self):
        test_dir = os.path.join(self.outdir, "temp_dir") 
        os.mkdir(test_dir)
        create_dirs(self.df, test_dir)
        for model_class in self.classes:
            self.assertTrue(model_class in os.listdir(test_dir))
            os.rmdir(os.path.join(test_dir, model_class))
        os.rmdir(test_dir)
        
    def test_make_file_name(self):
        filename = "CASENT0179609_L.jpg"
        filename_stem =  Path(filename).stem
        new_name = make_file_name(filename_stem, self.classes[0], 1)
        self.assertEqual(new_name, f"CASENT0179609_L_label_{self.classes[0]}_1.jpg")
        
    def test_filter_pictures(self):
        filter_pictures(self.jpg_dir,self.df, self.outdir)
        picture_count = 0
        for model_class in self.classes:
            picture_count += len(os.listdir(os.path.join(self.outdir, model_class)))
        
        self.assertEqual(len(os.listdir(self.jpg_dir)), picture_count)
                                 
        
        