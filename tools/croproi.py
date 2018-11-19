# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:37:04 2018

@author: Administrator
"""
import xml.etree.ElementTree as ET 
import  os
from os import getcwd
import numpy as np
import cv2

imgpath='D:\\lijian\\Data\\data\\atrophy'
dimgpath='D:\\lijian\\Data\\data\\atrophycrop\\'
xmlpath='D:\\lijian\\Data\\data\\atrophy'

def read_xml_annotation(root,image_id):
    in_file = open(os.path.join(root,image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    objects=root.findall('object')
    bndboxs=[]
    i=1
    for i in range(len(objects)) :
        xobject=objects[i]
        bndbox=xobject.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        bndboxs.append([xmin,ymin,xmax,ymax])
    return bndboxs

def cropimg():
    filelist=os.listdir(imgpath)
    for files in filelist:
        if files.endswith('.jpg'):
            print('current file: %s' % files)
            ##重命名文件
            filenamex=files.split('.')
            ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            imagefile=os.path.join(imgpath,files)
            img=cv2.imread(imagefile)
            bndboxlist = read_xml_annotation(xmlpath, str(filenamex[0])+'.xml')
            #print(bndboxlist)
            if bndboxlist!=None:
                for m in range(len(bndboxlist)):
                    bndbox=bndboxlist[m]
                    width=bndbox[2]-bndbox[0]
                    height=bndbox[3]-bndbox[1]
                    cropImg=img[bndbox[1]:bndbox[3],bndbox[0]:bndbox[2]]
                    if width>=height:
                        cropimg=cropImg[0:height,0:height]
                    #print(bndbox[0],bndbox[1],bndbox[2],bndbox[3])
                    else:
                        cropimg=cropImg[0:width,0:width]
                    #cropImg=img[100:200,50:60]
                    ipath=str("{}{}{}.jpg").format(dimgpath,filenamex[0],m)
                    cv2.imwrite(ipath,cropimg)
            else:
                print(files)
                    
                    
if __name__ == "__main__":
    cropimg()
                    

    

