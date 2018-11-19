# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:00:54 2018

@author: Administrator
"""
import os, random, shutil
import xml.etree.ElementTree as ET 

from os import getcwd
import numpy as np
from PIL import Image
import cv2
import imgaug as ia 
from imgaug import augmenters as iaa

#imgpath='/media/lbh/AI/SSD-test/cancer/yes/'
imgpath='/media/lbh/AI/SSD-test/cancer/test/'
dimgpath='/media/lbh/AI/SSD-test/cancer/cancer-yes/'
xmlpath='/media/lbh/AI/SSD-test/cancer/cancer-xml/'
dxmlpath='/media/lbh/AI/SSD-test/cancer/cancer-xml/'

def ReadFile(fileDir):
    global g_nSum
    g_nSum = 0
    pathDir = os.listdir(fileDir)
    for name in pathDir:
        namexml = name[:-4]+'.xml'

        in_file = open(xmlpath+namexml)
        tree = ET.parse(in_file)
        root = tree.getroot()
        objects=root.findall('object')
        bndboxs=[]
        i=1
	g_nSum = g_nSum + 1
        for i in range(len(objects)) :
            xobject=objects[i]
            bndbox=xobject.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)

            xmin = 137
            xmax = 366
            ymin = 51
            ymax = 176

            im=cv2.imread(imgpath+name)
            cv2.rectangle(im,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            cv2.imwrite(dimgpath+name,im)

if __name__ == "__main__":
    global g_nSum
    g_nSum = 0

    print('start')
    ReadFile(imgpath)
    print('finish! Total number of files: %s' % str(g_nSum))
