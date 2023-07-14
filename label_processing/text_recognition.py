#Import Librairies
from __future__ import annotations
import os
import copy
import cv2
import shutil
import math
import pytesseract as py
import numpy as np
from pyzbar.pyzbar import decode
from typing import  Union, Tuple, Optional, Literal, get_args
from deskew import determine_skew
from enum import Enum

from label_processing import utils #from this package


#Configuarations
CONFIG = r'--psm 6 --oem 3' #configuration for ocr
LANGUAGES = 'eng+deu+fra+ita+spa+por' #specifying languages used for ocr

def find_tesseract() -> None:
    """
    Searches for the tesseract executable and raises an error if it is not found.
    """
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        raise FileNotFoundError(("Could not find tesseract on your machine!"
                                 "Please read the README for instructions!"))
    else:
        py.pytesseract.tesseract_cmd = tesseract_path


#---------------------Image Preprocessing---------------------#


class Image():
    """
    A class for image preprocessing and other image actions.
    """
    def __init__(self, image: np.ndarray, path: str, blocksize: int = None,
                 c_value: int = None):
        """

        Args:
            image (_type_): _description_
            path (_type_): _description_
        """
        self.image = image
        self.path = path
        self.filename = os.path.basename(self.path)
        self.blocksize: Optional[int] = blocksize
        self.c_value: Optional[int] = c_value
        #preprocessing parameters
    
    @property 
    def blocksize(self) -> int:
        return self._blocksize
        
    @blocksize.setter
    def blocksize(self, value: int|None) -> None:
        if value is not None:
            if (value <= 1 or value % 2 == 0):
                raise ValueError("value for blocksize has to be atleast 3 and needs\
                    to be odd")
        self._blocksize = value
    
    @property
    def c_value(self) -> int:
        return self._c_value
    
    @c_value.setter
    def c_value(self, value: int) -> None:
        self._c_value = value
    
    @property
    def image(self) -> np.ndarray:
        return self._image
    
    @image.setter
    def image(self, image: np.ndarray) -> None:
        self._image = image
    
    @property
    def path(self) -> str:
        return self._path
    
    @path.setter
    def path(self, path: str) -> None:
        self._path = path
    
    def copy_this(self) -> Image:
        return copy.copy(self)
    
    @staticmethod
    def read_image(path: str) -> Image:
        """
        Returns instance of preprocessing of a picture.

        Args:
            path (str): path to a jpg file

        Returns:
            Preprocessing: instance of preprocessing
        """
        
        return Image(cv2.imread(path), path)
        
    
    def get_grayscale(self) -> Image:
        image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    
    def blur(self, ksize: tuple[int, int] = (5,5)) -> Image:
        image = cv2.GaussianBlur(self.image, ksize, 0)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    
    def remove_noise(self) -> Image:
        image = cv2.medianBlur(self.image,5)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance
    
    def thresholding(self, thresh_mode: Enum) -> Image:
        """
        Method for thresholding: can perform three different kinds of 
        thresholding: otsu's global thresholding, adaptive mean local
        thresholding and adaptive gaussian local thresholding

        Args:
            thresh_mode (Enum): Thresholding mode

        Returns:
            Image: Instance of this class
        """
        
        if thresh_mode == Threshmode.OTSU:
            image = cv2.threshold(self.image, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif thresh_mode == Threshmode.ADAPTIVE_GAUSSIAN:
            #set blocksize and c_value
            gaussian_blocksize = self.blocksize if self.blocksize  else 73
            gaussian_c = self.c_value if self.c_value else 16
            
            image = cv2.adaptiveThreshold(self.image ,255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                gaussian_blocksize, gaussian_c)
        elif thresh_mode == Threshmode.ADAPTIVE_MEAN:
            #set blocksize and c_value
            mean_blocksize = self.blocksize if self.blocksize  else 35
            mean_c = self.c_value if self.c_value else 17
            
            image = cv2.adaptiveThreshold(self.image ,255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                mean_blocksize,mean_c)             
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance
    

    def dilate(self) -> Image:
        kernel = np.ones((5,5),np.uint8)
        image =  cv2.dilate(self.image, kernel, iterations = 1)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance
    
    def erode(self) -> Image:
        kernel = np.ones((5,5),np.uint8)
        image = cv2.erode(self.image, kernel, iterations = 1)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance
    
    @staticmethod
    def _rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
        ) -> np.ndarray:
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = (abs(np.sin(angle_radian) * old_height) 
            + abs(np.cos(angle_radian) * old_width))
        height = (abs(np.sin(angle_radian) * old_width) 
            + abs(np.cos(angle_radian) * old_height))

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)),
                                               int(round(width))),
                              borderValue=background)

    def get_skew_angle(self) -> Optional[np.float64]: #returns either float or None 
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #print(f"Calculating skew angle for {self.filename}")
        angle = determine_skew(grayscale, max_angle = 30)
        return angle
        
    def deskew(self, angle: Optional[np.float64]) -> Image:
        #print(f"Rotating {self.filename}")
        image = self._rotate(self.image, angle, (255, 255, 255))
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def preprocessing(self, thresh_mode: Enum) -> Image:
        """
        Performs preprocessing -> grayscaling, binarization,
        blurring, noise removal, deskewing

        Args:
            thresh_mode: threshmode. Defaults to "otsu".
        Returns:
            Image: Image object
        """
        #skewangle has to be calculated before processing
        angle = self.get_skew_angle()
        image = self.get_grayscale()
        #blurring before thresholding
        image = image.blur()
        image = image.thresholding(thresh_mode=thresh_mode)
        #image = image.remove_noise()
        image = image.deskew(angle)
        return image

#---------------------Read QR-Code---------------------#
    

    def read_qr_code(self) -> Optional[str]:
        """
        Tries to identify if picture has a qr-code and then reads and returns it.

        Returns:
            Optional[str]: decoded qr-code text as a str or none if there is no
            qr-code found
        """
        detect = cv2.QRCodeDetector()
        value = detect.detectAndDecode(self.image)[0]
        return value if value else None
    
    def read_qr_code_2(self) -> Optional[str]:
        """
        tries to identify if picture has a qr-code and then reads and returns it

        Returns:
            Optional[str]: decoded qr-code text as a str or none if there is no
            qr-code found
        """
        #decode function imported from pyzbar
        value = decode(self.image)
        if value:
            value = value[0][0].decode("utf8")
        return value if value else None
    
    # def read_qr_code_3(self) -> Optional[str]:
    #     """
    #     tries to identify if picture has a qr-code and then reads and returns it

    #     Returns:
    #         Optional[str]: decoded qr-code text as a str or none if there is no
    #         qr-code found
    #     """
    #     #decode function imported from pyzbar
    #     with utils.HiddenPrints():
    #         qread = qreader.QReader()
    #         image = image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
    #         decoded_text = qread.detect_and_decode(image=image)
    #     return decoded_text[0] if decoded_text else None
    
    def save_image(self, dir_path: str, appendix: Optional[str] = None) -> None:
        if appendix:
            filename = utils.generate_filename(self.filename, appendix,
                                               extension = "jpg")
        else:
            filename = self.filename
        filename_processed = os.path.join(dir_path, filename)
        cv2.imwrite(filename_processed, self.image)
    

class Threshmode(Enum):
    """
    diferent possibilities for threholding

    Args:
        Enum (int):  
    """
    OTSU = 1
    ADAPTIVE_MEAN = 2
    ADAPTIVE_GAUSSIAN = 3
    
    @classmethod
    def eval(cls, threshmode: int) -> Enum:
        if threshmode == 1:
            return cls.OTSU
        if threshmode == 2:
            return cls.ADAPTIVE_GAUSSIAN
        if threshmode == 3:
            return cls.ADAPTIVE_MEAN
    
    


#---------------------OCR Tesseract---------------------#


class Tesseract():
    
    def __init__(self, languages = LANGUAGES, config = CONFIG,
                 image: Optional[Image] = None):
        self.config = config
        self.languages = languages
        self.image = image if image else None
        
    @property
    def image(self) -> Image:
        return self._image
    
    @image.setter
    def image(self, img: Image) -> None:
        self._image = img
    
    @staticmethod
    def _process_string(result_raw: str) -> str:
        """
        Processes the ocr_output by replacing \n with spaces and encoding it to
        ascii and decoding it again to utf-8.

        Args:
            result_raw (str): raw string from pytesseract output

        Returns:
            str: processed string
        """
        processed = result_raw.replace('\n', ' ')
        return processed
    
    def image_to_string(self) -> dict[str, str]:
        """
        Apply OCR and Image parameters on jpg images.
        
        Args:
            img (str): path to jpgs
            languages (str): OCR available languages
            config (str): any additional custom configuration flags
            that are not available via the pytesseract function.
            
        Returns:
            str: output as string from Tesseract OCR processing.
        """
        transcript = py.image_to_string(self.image.image, self.languages,
                                        self.config)
        transcript = self._process_string(transcript)
        return {"ID": self.image.filename, "text": transcript}



        
        

