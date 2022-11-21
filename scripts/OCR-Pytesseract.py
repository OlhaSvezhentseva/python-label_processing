#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image
import pytesseract as pt
import os
import cv2
import re
from pytesseract import Output
import numpy as np
import glob 
import sys
import json


# In[4]:


pt.pytesseract.tesseract_cmd = r"/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract"


# In[10]:


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

def improved_image_to_string(img, config):
    gray = get_grayscale(img)
    thresh = thresholding(gray)
    opening_var = opening(gray)
    canny_var = canny(gray)
    return pt.image_to_string(thresh, config = config)

Try to add options:
- to have an output with singles txt files (1 per picture)
- dataframe output with filenames and ocr output (2 columns)
- output as xml files(1 per picture)
- find a way to make all the crops straight (better results)!!!!! Priority
# In[13]:


# path for the folder for getting the images
path ="/Users/Margot/Desktop/typed"
  
# link to the file in which output needs to be kept
output ="/Users/Margot/Desktop/typed/ocrOutput.txt" #output as one single txt file
custom_config = r'--oem 3 --psm 6'

# iterating the images inside the folder
print("Start OCR")
for filepath in glob.glob(os.path.join(f"{path}/*.jpg")):
    filename = os.path.basename(filepath)
    print(f"Performing OCR on {os.path.basename(filepath)}!")
    img = cv2.imread(filepath)
# applying ocr using pytesseract for python
    non_processed = pt.image_to_string(img, config=custom_config)
    processed = improved_image_to_string(img, custom_config)
# saving the  text for appending it to the output.txt file
# a + parameter used for creating the file if not present
# and if present then append the text content
    file1 = open(output, "a+")
# providing the name of the image
    file1.write(filename+"\n") 
# providing the content in the image
    file1.write(processed+"\n")
    file1.close()
print("Successful")
# for printing the output file
#file2 = open(fullTempPath, 'r')
#print(file2.read()
#file2.close()       

