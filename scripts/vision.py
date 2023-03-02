#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from google.cloud import vision
import io
import glob
import json
import os
import argparse

#CREDENTIALS = '/home/leonardo/to_save/Projects/Museum_for_Natural_history/ocr_to_data/total-contact-297417-48ed6585325e.json'
RESULTS_JSON = "ocr_google_vision.json"
#DIR = '/home/leonardo/to_save/Projects/Museum_for_Natural_history/ocr_to_data/results_ocr/test'

def parsing_args():
    '''generate the command line arguments using argparse'''
    usage = 'googgle_vision.py [-h] [-np] -d <crop-dir>'
    parser =  argparse.ArgumentParser(description=__doc__,
            add_help = False,
            usage = usage
            )

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-c', '--credentials',
            metavar='',
            type=str,
            required = True,
            help=('Path tom the google credentials')
            )
    
    parser.add_argument(
            '-d', '--dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory which contains the cropped jpgs on which the'
                  'ocr is supposed to be applied')
            )

    
    args = parser.parse_args()

    return args

class VisionApi():
    """
    Class concerning the Google Vision API performed on a directory.
    """

    def __init__(self, path: str, image: bytes, credentials: str,
                 encoding: str) -> None:
        VisionApi.export(credentials) #check credententials
        self.image = image
        self.path = path
        self.encoding = encoding

        
    @staticmethod            
    def export(credentials) -> None:
        """
        Exports credentials json.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials

    @staticmethod
    def read_image(path, credentials, encoding = 'utf8') -> VisionApi:
        with io.open(path, 'rb') as image_file:
            image = image_file.read()
        return VisionApi(path, image, credentials, encoding)
    
    def process_string(self, result_raw: str, encode = 'utf-8') -> str:
        processed = result_raw.replace('\n', ' ')
        if self.encoding == "ascii":
            #turning it to ascii
            processed = processed.encode("ascii", "ignore")
            return processed.decode()
        else:
            return processed
        
        
        
    
    def vision_ocr(self) -> dict[str, str]:
        client = vision.ImageAnnotatorClient()
        vision_image = vision.Image(content=self.image)
        response = client.text_detection(image=vision_image)
        single_transcripts = response.text_annotations #get the ocr results
        #list of transcripts
        transcripts = [str(transcript.description) for transcript in single_transcripts]
        #create string of transcripts
        transcript = self.process_string(transcripts[0])
        #get filename
        filename = os.path.basename(self.path)
        if response.error.message:
            raise Exception(
                f'{response.error.message}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors')
        return {'ID' : filename, 'text': transcript}

def perform_vision_ocr(crop_dir: str, credentials: str,
                       encoding: str = 'utf8') -> None:
    
    results_json = []
    #TODO check if empty
    for file in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
        image = VisionApi.read_image(file, credentials)
        ocr_result: dict = image.vision_ocr()
        results_json.append(ocr_result)
    
    filepath = RESULTS_JSON
    #select wheteher it should be saved as utf-8 or ascii
    if encoding == 'utf8':
        with open(filepath, "w", encoding = 'utf8') as f:
            print("utf8")
            json.dump(results_json, f, ensure_ascii=False)
    else:
        with open(filepath, "w", encoding = 'ascii') as f:
            json.dump(results_json, f)
            
    

if __name__ == '__main__':
    args = parsing_args()
    perform_vision_ocr(args.dir, args.credentials)
