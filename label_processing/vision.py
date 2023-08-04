#Import Librairies
from __future__ import annotations
import io
import os
from google.cloud import vision
import label_processing.utils

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
    def export(credentials: str) -> None:
        """
        Exports the credentials json, by adding it as an environment variable
        in your shell.

        Args:
            credentials (str): path to the credentials json file
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials

    @staticmethod
    def read_image(path: str, credentials: str, encoding: str = 'utf8') -> VisionApi:
        """
        Reads an image with io and returns it as an instance of the VisionApi
        class.

        Args:
            path (str): path to image
            credentials (str): path to the credentials json file
            encoding (str, optional): choose in which encoding th result will 
            be saved (ascii or utf-8). defaults to 'utf8'

        Returns:
            VisionApi: Instance of the VisionApi class
        """
        with io.open(path, 'rb') as image_file:
            image = image_file.read()
        return VisionApi(path, image, credentials, encoding)
    
    def process_string(self, result_raw: str) -> str:
        """
        Processes the google vision ocr output and replaces newlines by spaces
        and if specified turns string from unicode into ascii encoding.

        Args:
            result_raw (str): the raw output string directly from google_vision

        Returns:
            str: processed string
        """
        processed = result_raw.replace('\n', ' ')
        if self.encoding == "ascii":
            #turning it to ascii
            processed = processed.encode("ascii", "ignore")
            return processed.decode()
        else:
            return processed
        
    def vision_ocr(self) -> dict[str, str]:
        """
        Performs the actual API call, does error handling and returns the 
        transcription already processed.

        Raises:
            Exception: raises exception if API does not respond

        Returns:
            dict[str, str]: dictionary with the filename and the transcript
        """
        client = vision.ImageAnnotatorClient()
        vision_image = vision.Image(content=self.image)
        response = client.text_detection(image=vision_image)
        single_transcripts = response.text_annotations #get the ocr results
        #list of transcripts
        transcripts = [str(transcript.description) for transcript in single_transcripts]
        #create string of transcripts
        if transcripts: #check if transcripts is not empty
            transcript = self.process_string(transcripts[0])
        else:
            transcript = " "
        #get filename
        filename = os.path.basename(self.path)
        if response.error.message:
            raise Exception(
                f'{response.error.message}\nFor more info on error messages, '
                'check:  https://cloud.google.com/apis/design/errors')
        entry = {'ID' : filename, 'text': transcript}
        if label_processing.utils.check_text(entry["text"]): 
            entry = label_processing.utils.replace_nuri(entry)
        return entry
        
                
