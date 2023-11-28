# Import third-party libraries
from __future__ import annotations
import io
import os
from google.cloud import vision
import warnings

# Import the necessary module from the 'label_processing' module package
import label_processing.utils

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

class VisionApi():
    """
    Class for interacting with the Google Cloud Vision API for OCR tasks on images.
    """

    def __init__(self, path: str, image: bytes, credentials: str, encoding: str) -> None:
        """
        Initialize the VisionApi instance.

        Args:
            path (str): Path to the image file.
            image (bytes): Image content in bytes.
            credentials (str): Path to the credentials JSON file.
            encoding (str): Encoding for the result ('ascii' or 'utf8').
        """
        VisionApi.export(credentials) #check credententials
        self.image = image
        self.path = path
        self.encoding = encoding

    @staticmethod            
    def export(credentials: str) -> None:
        """
        Export the credentials JSON by setting it as an environment variable.

        Args:
            credentials (str): Path to the credentials JSON file.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials

    @staticmethod
    def read_image(path: str, credentials: str, encoding: str = 'utf8') -> VisionApi:
        """
        Read an image file and return an instance of the VisionApi class.

        Args:
            path (str): Path to the image file.
            credentials (str): Path to the credentials JSON file.
            encoding (str, optional): Encoding for the result ('ascii' or 'utf8'). Defaults to 'utf8'.

        Returns:
            VisionApi: Instance of the VisionApi class.
        """
        with io.open(path, 'rb') as image_file:
            image = image_file.read()
        return VisionApi(path, image, credentials, encoding)
    
    def process_string(self, result_raw: str) -> str:
        """
        Process the Google Vision OCR output, replacing newlines with spaces and encoding as specified.

        Args:
            result_raw (str): Raw output string directly from Google Vision.

        Returns:
            str: Processed string.
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
        Perform the actual API call, handle errors, and return the processed transcription.

        Raises:
            Exception: Raises an exception if the API does not respond.

        Returns:
            Dict[str, str]: Dictionary with the filename and the transcript.
        """
        client = vision.ImageAnnotatorClient()
        vision_image = vision.Image(content=self.image)
        response = client.text_detection(image=vision_image)
        single_transcripts = response.text_annotations #get the ocr results
        #list of transcripts
        transcripts = [str(transcript.description) for transcript in single_transcripts]
        
        bounding_boxes = []
        for transcript in single_transcripts: 
            vertices = [
            {word: f"({vertex.x},{vertex.y})"} for vertex, word in 
            zip(transcript.bounding_poly.vertices, transcripts)
            ]
            bounding_boxes.append(vertices)
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
        entry = {'ID' : filename, 'text': transcript,
                 'bounding_boxes': bounding_boxes}
        if label_processing.utils.check_text(entry["text"]): 
            entry = label_processing.utils.replace_nuri(entry)
        return entry
        
                
