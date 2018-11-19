# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET 
import  os

axmlpathRefer='/media/lbh/软件/Data/VOC2007-2017/VOC2007-Narrow/ChuanMei/atrophy-xml/'
cxmlpathRefer='/media/lbh/软件/Data/VOC2007-2017/VOC2007-Narrow/ChuanMei/cancer-xml/'

VOCpath='/media/lbh/软件/Data/VOC2007-2017/VOC2007-Narrow/ChuanMei-Expand/VOC2007/'


def delete_xml():

    count = 0
    cxmlfilelist = os.listdir(cxmlpathRefer)
    for cxmlfilesName in cxmlfilelist:
	cxmlpath=VOCpath + 'Annotations/' + cxmlfilesName
	if os.path.exists(cxmlpath):

	    cxmlfilesNamePre = cxmlfilesName[:-4]
            print(cxmlfilesNamePre)
	    cjpgpath=VOCpath + 'JPEGImages/' + cxmlfilesNamePre + '.jpg'
	    if os.path.isfile(cxmlpath):
		os.remove(cxmlpath)
		os.remove(cjpgpath)
		count = count + 1
    print(count)
if __name__ == "__main__":
    print('start')
    read_xml()
    print('finish')
