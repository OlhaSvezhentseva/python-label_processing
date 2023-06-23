'''
SOFTWARE LICENSE AGREEMENT
By using this software, you, the Licensee, accept the following licensing agreement with READ-COOP SCE, the proprietor and Licensor of this software.
1. License Grant
Subject to the terms and conditions of this Agreement, Licensor hereby grants to Licensee a non-exclusive, non-transferable, and non-sublicensable license to use this software ("Software") in object code form only (where applicable), solely for Licensee's internal business purposes.
2. Ownership and Intellectual Property Rights
Licensor retains all rights, title, and interest in and to the Software, including, without limitation, all copyrights, patents, trade secrets, and other intellectual property rights. Licensee acknowledges that the Software is protected by applicable intellectual property laws and agrees not to take any action that would infringe upon Licensor's rights.
3. Restrictions
Licensee shall not, and shall not permit any third party to: (a) reverse engineer, decompile, disassemble or otherwise attempt to discover the source code of the Software; (b) modify, adapt, or create derivative works based on the Software; (c) rent, lease, sell, sublicense, or otherwise transfer the Software; (d) use the Software for any purpose other than as expressly permitted under this Agreement.
4. Term and Termination
This Agreement shall begin on the first use of the Software and continue until terminated by either party in writing. Upon termination, Licensee shall cease all use of the Software and destroy all copies in its possession or control.
5. Limited Warranty
Licensor warrants that the Software, when used correctly and in accordance with this Agreement, will substantially perform the functions described it is intended for, for a period of 90 days from the first use of the Software. Licensor's sole obligation, and Licensee's exclusive remedy, for any breach of this warranty shall be, at Licensor's option, to either repair or replace the nonconforming Software.
6. Limitation of Liability
In no event shall Licensor be liable for any indirect, incidental, special, or consequential damages arising out of or related to this Agreement, even if advised of the possibility of such damages. Licensor's total liability under this Agreement shall not exceed the amount paid by Licensee for the Software.
7. Governing Law
This Agreement shall be governed by and construed in accordance with the laws of Austria, without regard to its conflict of laws principles.
'''

'''
Classifier to detect orientation of image (0°, 90°, 180°, 270°) and to correct orientation.
'''

#Import third party libraries

import torch
import torch.nn as nn
import torchvision
from torchvision import io, transforms
from torchvision.io import ImageReadMode
import os
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')


if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'

# checkpoint = torch.load(load_path, map_location=map_location)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROTATIONS = [0, 90, 180, 270]

device = torch.device("cpu")
script_dir = os.path.dirname(__file__)
model_path = "../models/mfn_rot_classifier.pth"

def parsing_args() -> argparse.ArgumentParser:
    '''generate the command line arguments using argparse'''
    usage = 'rotation.py [-h] -o <output_image_dir> -i <input_image_dir>'
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
            '-o', '--output_image_dir',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory where the rotated images will be stored.\n'
                  'Default is the user current working directory.')
            )
    
    parser.add_argument(
            '-i', '--input_image_dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the jpgs are stored.')
            )
    
    args = parser.parse_args()

    return args

args = parsing_args()
input_image_dir = args.input_image_dir
output_image_dir = args.output_image_dir


class RotationDetector(nn.Module):
    def __init__(self, basenet):
        super().__init__()
        self.basenet = basenet

        self.basenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=4, bias=True)
        )
        self.SM = nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.basenet(x)
        y = self.SM(y)
        return y

transform = torch.nn.Sequential(
    transforms.ConvertImageDtype(dtype=torch.float32),
    transforms.Resize(size=(224,224), antialias=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
)


def main():
    efficientnet = torchvision.models.efficientnet_b0()
    model = RotationDetector(efficientnet)
    # model.load_state_dict(torch.load(f"{model_path}"))
    model.load_state_dict(torch.load(f"{model_path}", map_location=map_location))
    model.to(device)
    model.eval()
    img_files = [f for f in os.listdir(input_image_dir) if os.path.isfile(input_image_dir + os.sep + f) and f.endswith('.jpg')]

    for img_file in img_files:

        # image = io.read_image(input_image_dir + os.sep + img_file, mode=ImageReadMode.RGB )
        # print(image)
        image = Image.open(input_image_dir + os.sep + img_file)

        # Define a transform to convert the image to tensor
        tr = transforms.ToTensor()
        image = tr(image)
        # print(image)

        # # image = Image.open(input_image_dir + os.sep + img_file, mode="r")
        new_image = transform(image)
        new_image = new_image.to(device)

        with torch.no_grad():
            output = model.forward(new_image.unsqueeze(0))

        pred = output.argmax(dim=1)[0]
        rotation_deg = ROTATIONS[pred]

        print(img_file, end=", rotation: ")
        print(rotation_deg, "°")
        image = transforms.functional.rotate(image, -rotation_deg, expand=True)
        save_image(image, output_image_dir + os.sep + img_file)

if __name__ == "__main__":
    main()
