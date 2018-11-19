# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:41:57 2018

@author: Administrator
"""

import xml.etree.ElementTree as et
import os 
srcf='./py-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations'
imgsrcf='D:\\TEST\\img\\'
xmlsrcf='D:\\TEST\\xml\\'
#srcf='D:\\'
#in_file = open('D:\\1.xml')
#tree = et.parse(in_file)
#root = tree.getroot()
#filename=root.find('filename')
#filename.text="2_1.jpg"
#tree.write('D:\\1.xml')
def renamexml():
    imgfilelist=os.listdir(imgsrcf)
    index=1
    for files in imgfilelist:
        if files.endswith('.jpg'):
            ##重命名文件
            filenamex=files.split('.')
            imgnew_name='v'+str(index)+'.jpg'
            xmlnew_name='v'+str(index)+'.xml'
            os.rename(os.path.join(imgsrcf,files),os.path.join(imgsrcf,imgnew_name))
            
            oldpath=os.path.join(xmlsrcf,str(filenamex[0])+'.xml')
            os.rename(oldpath,os.path.join(xmlsrcf,xmlnew_name))
            path=os.path.join(xmlsrcf,xmlnew_name)
            #print(path)
            in_file = open(path)
            tree = et.parse(in_file)
            root = tree.getroot()
            filename=root.find('filename')
            filename.text=imgnew_name
            tree.write(path)
            index+=1
            
            ##重命名xml内部名称
            

        
#查询box是否超出范围        
def checkbox():
    ncount=1
    for file in os.listdir(srcf):
        path=os.path.join(srcf,file)
        xml=open(path)
        tree=et.parse(xml)
        root=tree.getroot()
        sizenode=root.find('size')
        width=int(sizenode.find('width').text)
        height=int(sizenode.find('height').text)
        
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
            bndboxs.append((xmin,ymin,xmax,ymax))
            '''
            if xmin<=0:
                bndbox.find('xmin').text='1'
            if ymin<=0:
                bndbox.find('ymin').text='1'
            if xmax>=width:
                bndbox.find('xmax').text=str(width-1)
            if ymax>=height:
                bndbox.find('ymax').text=str(height-1)
  	    if xmin>xmax:
		bndbox.find('xmin').text=str(xmax)
		bndbox.find('xmax').text=str(xmin)
	    if ymin>ymax:
		bndbox.find('ymin').text=str(ymax)
		bndbox.find('ymax').text=str(ymin)

            '''
            if xmin==0:
                print(path)
            if ymin==0:
                print(path)
            if xmax<xmin:
                print(path)
            if ymax<ymin:
                print(path)
            if ymax>=height:
                print(path)
            if xmax>=width:
                print(path)
            if xmin<=0:
                print(path)
            if ymin<=0:
                print(path)

        #print(ncount)
        ncount+=1
        #tree.write(path)
            

if __name__ == "__main__":
    checkbox()
    #renamexml()
    print('finish!!')
    
