
import unittest
from label_evaluation.evaluate_text import *


class TextSimilarityTestSuite(unittest.TestCase):
    """A class to test Calculation of Text similarity."""

    def test_capital_letters(self):
        actual = calculate_scores("ANTWEB CASENT 0922859", "antweb casent 0922859")
        # wer and cer
        expected = (0.0, 0.0)
        self.assertEqual(actual, expected)

    def test_same_word(self):
        actual = calculate_scores("coll", "coll")
        expected = (0, 0)
        self.assertEqual(actual, expected)

    def test_single_mistake(self):
        actual = calculate_scores("ANTWEB CASENT 0922859", "ANTWEB CASENT 0922855")
        expected = (0.33, 0.05)
        self.assertEqual(actual, expected)

    def test_space(self):
        actual = calculate_scores("ANTWEB", "ANTWEB ")
        expected = (0.0, 0.17)
        self.assertEqual(actual, expected)

    def test_punctuation (self):
        actual = calculate_scores("Aenictus formosensis Forel, 1913 det. Michael Staab 2014",
                                  "Aenictus formosensis Forel, 1913. det. Michael Staab 2014")
        expected = (0.12, 0.02)
        self.assertEqual(actual, expected)

    def test_empty_string (self):
        actual = calculate_scores("Aenictus formosensis Forel, 1913 det. Michael Staab 2014",
                                  "")
        expected = (1, 1)
        print("len")
        print(len(" "))
        self.assertEqual(actual, expected)