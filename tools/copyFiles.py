# -*- coding: utf-8 -*-
import os, random, shutil

imgpath='/media/lbh/AI/SSD-test/cancer/yes/'
dimgpath='/media/lbh/AI/SSD-test/cancer/yes-Box/'

xmlpath='/media/lbh/AI/Data/VOC2007-2017/VOC2007-Narrow/types-xml/cancer/'
dxmlpath='/media/lbh/AI/SSD-test/cancer/cancer-xml/'

def copyFile(fileDir):
    pathDir = os.listdir(fileDir)
    for name in pathDir:
		nameJpg = name[:-4] + '.xml'
		shutil.copyfile(xmlpath+nameJpg, dxmlpath+nameJpg)

def copyFileRandom(fileDir):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, 14029)
    #print sample
    for name in sample:
		shutil.copyfile(fileDir+name, dxmlpath+name)
		nameJpg = name[:-4] + '.jpg'
		shutil.copyfile(imgpath+nameJpg, dimgpath+nameJpg)
if __name__ == '__main__':
    print('start!')
    copyFile(imgpath)
    print('finish!')
