import cv2     
from collections import Counter                 

class BackgroundColorDetector():
    def __init__(self, image_path, thresh = None):
        self.img = cv2.imread(image_path)
        self.manual_count = {}
        self.w, self.h, self.channels = self.img.shape
        self.total_pixels = self.w*self.h
        if thresh is None:
            self.thresh = 160
        else:
            self.thresh = thresh

    #NOTE This can be very slow because it iterates through all pixels of a picture
    def count(self) -> None:
        """
        iterates through all pixels in a picture and retrieves their RGB VALUE
        """
        for y in range(0, self.h):
            for x in range(0, self.w):
                rgb = (self.img[x, y, 2], self.img[x, y, 1], self.img[x, y, 0])
                if rgb in self.manual_count:
                    self.manual_count[rgb] += 1
                else:
                    self.manual_count[rgb] = 1

    def average_colour(self) -> tuple[int, int, int]:
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
        self.count()
        #get most common 20 rgb values
        self.number_counter = Counter(self.manual_count).most_common(20)
        #for rgb, value in self.number_counter:
            #print(rgb, value, ((float(value)/self.total_pixels)*100))

    def detect(self):
        self.twenty_most_common()
        self.percentage_of_first = (
            float(self.number_counter[0][1])/self.total_pixels)
        #print(self.percentage_of_first)
        if self.percentage_of_first > 0.5:
            return self.number_counter[0][0]
        else:
            return self.average_colour()
    
    def get_graytone(self):
        red, green, blue = self.detect()
        graytone: float = (0.299* red) + (0.587* green) + (0.114* blue)
        return graytone
    
    def decide(self) -> bool:
        #convert to greyscale using the linear luminence formula
        return True if self.get_graytone() > self.thresh else False
         