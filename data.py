
"""Toolkit for data acquisition and transformation"""

__author__ = "Victor Mawusi Ayi"


from cv2 import resize
from glob import glob
from os import path
import random

import matplotlib.image as mpimg
from pandas import DataFrame

from features import (
    contrast,
    lightness,
    luminance,
    supracontrast,
    supralightness,
    supraluminance
)


class Dataset:
    
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
        self.setsplit()
        
    def __getdata__(self, indices):
        
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
                
        self.datasize = len(self.data)
        self.indices = list(range(self.datasize))
        random.shuffle(self.indices)
    
    def setsplit(self, trainratio=0.7):
        
        self.traincount = int(self.datasize * trainratio)
        
    def shuffle(self):
    
        random.shuffle(self.indices)
    
    def traindata(self):
        
        if self.data==[]:
            raise ValueError(
                "Cannot retrieve training data from "
                "an empty dataset."
            )
        else:
            indices = self.indices[:self.traincount]            
            return self.__getdata__(indices)
        
    def testdata(self):
        
        if self.data==[]:
            raise ValueError(
                "Cannot retrieve test data from "
                "an empty dataset."
            )
        else:
            indices = self.indices[self.traincount:]
            return self.__getdata__(indices)
    
    def featureset(
        self,
        settype="all",
        as_df=True,
        divisors=(3, 3, 4)
    ):
        
        s_c, s_li, s_lu = divisors 
        
        fvector = lambda x: (
            contrast(x),
            lightness(x),
            luminance(x),
            supracontrast(x, s_c),
            supralightness(x, s_li),
            supraluminance(x, s_lu)
        )
        
        if settype=="all":
            data, labels = self.__getdata__(self.indices)
        elif settype=="train":
            data, labels = self.traindata()
        elif settype=="test":
            data, labels = self.testdata()
        
        fset = [fvector(imgdata) for imgdata in data]
        
        if as_df:
            fset = DataFrame(fset)
            fset.columns = (
                "contrast",
                "lightness",
                "luminance",
                "supracontrast",
                "supralightness",
                "supraluminance"
            )

        return fset, labels
