
"""Toolkit for data acquisition and transformation"""

__author__ = "Victor Mawusi Ayi"


from cv2 import resize
from glob import glob
from os import path
import random

import matplotlib.image as mpimg
import numpy as np


class dataset:
    
    def __init__(
        self,
        superdir,
        image_size=(500, 500),
        trainratio=0.7,
        classes=("night", "day")
    ):
        self.classes = classes
        self.data = []
        self.image_size = image_size
        self.labels = []
        self.superdir = superdir
        
        self.__load__()
        
    def __getdata__(indices):
        
        data = [self.data[x] for x in indices]
        labels = [self.labels[x] for x in indices]
    
        return data, labels
                    
    def __load__(self):
        
        get_files = lambda x, y: glob(
            path.join(x, y, "*")
        )
        
        data_size = 0
        
        for label in range(len(self.classes)):
            for file in get_files(
                self.superdir,
                self.classes[label]
            ):
                self.data.append(
                    resize(
                        mpimg.imread(file),
                        self.image_size
                    )
                )
                self.labels.append(label)
    
    def datasplit(self,  trainratio=0.7):
        self.datasize = len(self.data)
        self.traincount = int(self.datasize * trainratio)
        self.indices = list(range(self.datasize))
        random.shuffle(self.indices)
    
    def traindata(self):
        
        if self.data==[]:
            raise ValueError(
                "Cannot retrieve training data from "
                "an empty dataset."
            )

        else:
            indices = self.indices[:self.traincount]            
            return __getdata__(indices)
        
    def testdata(self):
        
        if self.data==[]:
            raise ValueError(
                "Cannot retrieve test data from "
                "an empty dataset."
            )

        else:
            indices = self.indices[self.traincount:]
            return __getdata__(indices) 
