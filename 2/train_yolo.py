import os
import cv2
import numpy as np

path = 'TraiCayScratch/SauRieng/'
lst_dir = os.listdir(path)

for filename in lst_dir:
    print(filename)
    fullname = path = 'TraiCayScratch/SauRieng/' + filename
    imgin = cv2.imread(fullname, cv2.IMREAD_COLOR)
    cv2.imshow('ImageIn', imgin)
    cv2.waitKey(0)