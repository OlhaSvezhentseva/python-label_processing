# Import Librairies
import cv2     
from collections import Counter                 


class BackgroundColorDetector:
    def __init__(self, image_path, thresh=None):
        """
        Initialize the BackgroundColorDetector.

        Args:
            image_path (str): The path to the image file.
            thresh (int, optional): The threshold for deciding if an image is dark or light. Defaults to 160 if not provided.
        """
        self.img = cv2.imread(image_path)
        self.manual_count = {}
        self.w, self.h, self.channels = self.img.shape
        self.total_pixels = self.w * self.h
        if thresh is None:
            self.thresh = 160
        else:
            self.thresh = thresh

    def count(self) -> None:
        """
        Iterates through all pixels in the image and retrieves their RGB values.
        """
        for y in range(0, self.h):
            for x in range(0, self.w):
                rgb = (self.img[x, y, 2], self.img[x, y, 1], self.img[x, y, 0])
                if rgb in self.manual_count:
                    self.manual_count[rgb] += 1
                else:
                    self.manual_count[rgb] = 1

    def average_colour(self) -> tuple[int, int, int]:
        """
        Calculate the average RGB color in the image.

        Returns:
            tuple[int, int, int]: The average RGB color as a tuple of integers.
        """
        red = 0
        green = 0
        blue = 0
        sample = 10
        for top in range(0, sample):
            red += self.number_counter[top][0][0]
            green += self.number_counter[top][0][1]
            blue += self.number_counter[top][0][2]

        average_red = red / sample
        average_green = green / sample
        average_blue = blue / sample
        return round(average_red), round(average_green), round(average_blue)

    def twenty_most_common(self):
        """
        Get the 20 most common RGB values in the image.
        """
        self.count()
        self.number_counter = Counter(self.manual_count).most_common(20)

    def detect(self):
        """
        Detect the background color of the image.

        Returns:
            tuple[int, int, int]: The detected background color as an RGB tuple.
        """
        self.twenty_most_common()
        self.percentage_of_first = (float(self.number_counter[0][1]) / self.total_pixels)
        if self.percentage_of_first > 0.5:
            return self.number_counter[0][0]
        else:
            return self.average_colour()

    def get_graytone(self):
        """
        Get the greyscale value of the detected background color.

        Returns:
            float: The greyscale value.
        """
        red, green, blue = self.detect()
        graytone = (0.299 * red) + (0.587 * green) + (0.114 * blue)
        return graytone

    def decide(self) -> bool:
        """
        Decide if the image has a light or dark background based on the threshold.

        Returns:
            bool: True if the image has a light background, False if it has a dark background.
        """
        return True if self.get_graytone() > self.thresh else False
