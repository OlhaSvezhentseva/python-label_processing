#!/usr/bin/env python3

"""
Module containing the Pytesseract OCR parameters to be performed on the cropped
jpg outputs.
"""

#Import Libraries
import argparse
import os
import glob
from time import time
import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Callable
#Import module from this package
from label_processing.text_recognition import (Tesseract, 
                                               ImageProcessor,
                                               Threshmode,
                                               find_tesseract,
                                               )
from label_processing import utils

FILENAME = "ocr_preprocessed.json"

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'tesseract_ocr.py [-h] [-v] [-t <thresholding>] [-b <blocksize>] \
            [-c <c_value>] -d <crop-dir> [-multi <multiprocessing>] -o <outdir> [-o <out-dir>]'
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
            '-v', '--verbose',
            metavar='',
            action=argparse.BooleanOptionalAction,
            default = False,
            help=('Select whether verbose or quiet mode')
            )
    
    parser.add_argument(
            '-t', '--thresholding',
            metavar='',
            choices = (1, 2, 3),
            type=int,
            default = 1,
            action='store',
            help=('Optional argument: select which thrsholding should be used primarily.\n'
                 '1 : Otsu\'s thresholding.\n'
                 '2 : adaptive mean thresholding.\n'
                 '3 : gaussian adaptive thrsholding.\n'
                 'Default is otsus')
            )
    
    parser.add_argument(
            '-b', '--blocksize',
            metavar='',
            action="store",
            type = int,
            default = None,
            help=('Optional argument: blocksize parameter for adaptive thresholding')
            )
    
    parser.add_argument(
            '-c', '--c_value',
            metavar='',
            action="store",
            type = int,
            default = None,
            help=('Optional argument: c_value parameter for adaptive thesholding')
            )
    
    parser.add_argument(
            '-o', '--outdir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the json should be saved')
            )
    
    parser.add_argument(
            '-d', '--dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory which contains the cropped jpgs on which the'
                  'ocr is supposed to be applied')
            )

    parser.add_argument(
        '-multi', '--multiprocessing',
        metavar='',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=('Select whether to use multiprocessing')
    )
    
    args = parser.parse_args()

    return args


def ocr__on_file(file_path, args,  thresh_mode, tesseract, new_dir):
    """
    Perform OCR on an image file, including preprocessing and reading QR codes if present.

    Args:
        file_path (str): The path to the input image file.
        args (argparse.Namespace): Command-line arguments.
        thresh_mode (Threshmode): The thresholding mode to use during preprocessing.
        tesseract (Tesseract): An instance of the Tesseract OCR processor.
        new_dir (str): The directory where the preprocessed image will be saved.

    Returns:
        Tuple[dict, bool, bool]: A tuple containing the transcript dictionary, a boolean indicating QR code detection, and a boolean indicating Nuri detection.
    """
    image = ImageProcessor.read_image(file_path)
    qr = False
    nuri = False
    if args.blocksize is not None:
        image.blocksize(args.blocksize)
    if args.c_value is not None:
        image.c_value(args.c_value)
    # trying to read the qr_code
    decoded_qr = image.read_qr_code()
    if decoded_qr is not None:
        # verbose_print(f"Qr-Code detected in {image.filename}\n")
        transcript: dict[str, str] = {"ID": image.filename,
                                      "text": decoded_qr}
        qr = True
    else:
        # Preprocessing
        # verbose_print(f"Performing preprocessing on {image.filename}")
        image = image.preprocessing(thresh_mode)  # preprocessed image
        image.save_image(new_dir)  # saving image in new directory
        # OCR
        tesseract.image = image
        # verbose_print(f"Performing OCR on {image.filename}\n")
        transcript: dict[str, str] = tesseract.image_to_string()
        # get nuri
        if utils.check_text(transcript["text"]):
            nuri = True
            transcript = utils.replace_nuri(transcript)
    return (transcript, qr, nuri)


def ocr_on_dir(crop_dir: str,
               new_dir: str,
               verbose_print: Callable,
               args: argparse.ArgumentParser
               ) -> list[dict[str, str]]:
    """
    Performs ocr on a given directory 

    Args:
        crop_dir (str): path to directory with cropped pictures
        new_dir (str): path to new directory
        verbose_print (Callable): print funktion die abhängig vom user input 
        printet oder nicht
        args (argparse.ArgumentParser): arg parse argumente

    Returns:
        list[dict[str, str]]: _description_
    """
    tesseract = Tesseract()
    ocr_results: list = []
    count_qr: int = 0
    total_nuri: int = 0
    thresh_mode: Enum = Threshmode.eval(args.thresholding)
    # for file_path in glob.glob(os.path.join(f"{crop_dir}/*.jpg")):
    files = glob.glob(os.path.join(f"{crop_dir}/*.jpg"))
    if not args.multiprocessing:
        for file in files:
            transcript, qr, nuri = ocr__on_file(file, args,  thresh_mode, tesseract, new_dir)
            ocr_results.append(transcript)
            if qr == True: count_qr += 1
            if nuri == True: total_nuri += 1
    else:
    # Use all the cores
        with mp.Pool() as pool:
            result = pool.starmap(ocr__on_file, [(file, args,  thresh_mode, tesseract, new_dir) for file in files])
            for transcript, qr, nuri in result:
                ocr_results.append(transcript)
                if qr == True: count_qr += 1
                if nuri == True: total_nuri += 1

    verbose_print(f"QR-codes read: {count_qr}")
    verbose_print(f"get_nuri: {total_nuri}")
    return ocr_results

if __name__ == "__main__":
    args = parsing_args()
    #New function verbose print
    verbose_print: Callable = print if args.verbose else lambda *a, **k: None    
    #Find path to tesseract
    find_tesseract()
    verbose_print("Tesseract succesfully detected.\n")
    crop_dir = args.dir
    utils.check_dir(crop_dir)
    new_dir = utils.generate_filename(crop_dir, "preprocessed")
    #Parent directory of the cropped pictures
    outdir = args.outdir 
    new_dir_path = os.path.join(outdir, new_dir)
    Path(new_dir_path).mkdir(parents=True, exist_ok=True)
    
    verbose_print(f"\nPerforming OCR on {os.path.abspath(crop_dir)} .\n")
    result_data = ocr_on_dir(crop_dir,
                             new_dir_path,
                             verbose_print,
                             args)
    verbose_print((f"\nPreprocessed images have been saved in"
                   f"os.path.abspath{os.path.abspath(new_dir_path)} ."))
    
    verbose_print(f"Saving results in {os.path.abspath(outdir)} .")
    utils.save_json(result_data, FILENAME, outdir)

