import unittest
import cv2
import numpy as np
from pathlib import Path
import detecto

from label_processing.segmentation_cropping import *


class TestSegmentationCropping(unittest.TestCase):
    path_to_model = "../../models/model_labels_box.pth" 
    jpg_path: Path =  Path("../testdata/uncropped/coll.mfn-berlin.de_u_43acf9__label.jpg")
    label_predictor = PredictLabel(path_to_model, ["box"], jpg_path)

    
    def test_predict_label_constructor(self):
        label_predictor = PredictLabel(self.path_to_model, ["box"], self.jpg_path)
        self.assertIsInstance(label_predictor.jpg_path, Path)
        self.assertIsInstance(label_predictor.model, detecto.core.Model)

    def test_predict_label_constructor_2(self):
        label_predictor = PredictLabel(self.path_to_model, ["box"])
        label_predictor.jpg_path = self.jpg_path
        self.assertIsInstance(label_predictor.jpg_path, Path)
    
    def test_class_prediction(self):
        entries = self.label_predictor.class_prediction(self.jpg_path)
        self.assertIsInstance(entries, list)
        [self.assertIsInstance(entry, dict) for entry in entries]
    
    def test_class_prediction_parallel(self):
        df = prediction_parallel("../testdata/uncropped", self.label_predictor, 1)
        self.assertIsinstace(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 7)
        
    #TODO Tests for remaining functions