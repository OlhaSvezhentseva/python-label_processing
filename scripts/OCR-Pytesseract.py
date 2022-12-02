from PIL import Image
import pytesseract
import os
import cv2
import re
from pytesseract import Output
import numpy as np
import glob 
import sys
import json


pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract" 


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def improved_image_to_string(img, languages, config):
    gray = get_grayscale(img)
    thresh = thresholding(gray)
    opening_var = opening(gray)
    canny_var = canny(gray)
    return pytesseract.image_to_string(thresh, languages, config)

# apply OCR
def OCR(path, output, config, languages):
    print("Start OCR")
    for filepath in glob.glob(os.path.join(f"{path}/*.jpg")):
        filename = os.path.basename(filepath)
        print(f"Performing OCR on {os.path.basename(filepath)}!")
        img = cv2.imread(filepath)
        file1 = open(output, "a+")
        file1.write(filename+"\n") 
        non_processed = pytesseract.image_to_string(img, languages, config)
        #processed = improved_image_to_string(img, languages, config)
        file1.write(non_processed+"\n")
        file1.close()
    print("Successful")

path = "/Users/Margot/Desktop/tests_OCR/pictures_typed copie" #path to cropped pictures
output = "/Users/Margot/Desktop/ocrOutput18.txt" #output as one single txt file
config=r'--psm 3 --oem 3' #psm4, 3 and 12 is good - 5 assumes that the text is vertical
languages = 'eng+deu+fra+ita+spa+por'

OCR(path, output, config, languages)

