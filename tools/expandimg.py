# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:00:54 2018

@author: Administrator
"""
import xml.etree.ElementTree as ET 
import  os
from os import getcwd
import numpy as np
from PIL import Image
import cv2
import imgaug as ia 
from imgaug import augmenters as iaa

imgpath='G:\\Data\\VOC2007-2017\\VOC2007-PreExpand\\Equal\\types\\gist'
dimgpath='G:\\Data\\VOC2007-2017\\VOC2007-PreExpand\\Equal\\types\\gist'
xmlpath='G:\\Data\\VOC2007-2017\\VOC2007-PreExpand\\Equal\\xmls\\gist'
dxmlpath='G:\\Data\\VOC2007-2017\\VOC2007-PreExpand\\Equal\\xmls\\gist'


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
        bndboxs.append((xmin,ymin,xmax,ymax))
    return bndboxs

    
def change_xml_annotation(root,droot, image_id, new_name,newboxlist):
    in_file = open(os.path.join(root, str(image_id)+'.xml')) #这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot() 
    sizenode=xmlroot.find('size')
    width=int(sizenode.find('width').text)
    height=int(sizenode.find('height').text)
    
    filename=xmlroot.find('filename')
    strname=new_name+'.jpg'
    filename.text=strname
    ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    for i in range(len(newboxlist)):
        new_target=newboxlist[i]
        new_xmin = new_target[0]
        if new_xmin==0:
            new_xmin=1
        new_ymin = new_target[1]
        if new_ymin==0:
            new_ymin=1
        new_xmax=new_target[2]
        if new_xmax>=width:
           new_xmax=width-1
        new_ymax=new_target[3]
        if new_ymax>=height:
            new_ymax=height-1
        objects = xmlroot.findall('object')
        xobject=objects[i]
        bndbox = xobject.find('bndbox')
        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)
    ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tree.write(os.path.join(droot,new_name+'.xml'))   

def func0(nameprex,seq):
    global g_nSum
    g_nSum = 0
    filelist=os.listdir(imgpath)
    for files in filelist:
        if files.endswith('.jpg'):
            g_nSum = g_nSum+1
    print('Total number of files: %s' % str(g_nSum))
    index=1   
    filelist=os.listdir(imgpath)
    for files in filelist:
        if files.endswith('.jpg'):
#            print('current file: %s' % files)
            g_nSum = g_nSum+1
            index = index+1
#            if g_nSum>6243:
#                break
            ##重命名文件
            filenamex=files.split('.')
            ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#            new_imgname=nameprex+str(files)
            new_imgname=str(g_nSum)+'.jpg'
        
            dstimgpath=os.path.join(dimgpath,new_imgname)
            imagefile=os.path.join(imgpath,files)
            img=cv2.imread(imagefile)
        
            img=np.array(img)
        
            bndboxlist = read_xml_annotation(xmlpath, str(filenamex[0])+'.xml')
        
            seq_det = seq.to_deterministic() # 保持坐标和图像同步改变，而不是随机
            image_aug = seq_det.augment_images([img])[0]
        
            cv2.imwrite(dstimgpath,image_aug)
            newboxlist=[]
            if bndboxlist!=None:
                for m in range(len(bndboxlist)):
                    bndbox=bndboxlist[m]
                    bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=bndbox[0], y1=bndbox[1], x2=bndbox[2], y2=bndbox[3])], shape=img.shape)
                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    new_bndbox = []
                    new_bndbox.append(int(bbs_aug.bounding_boxes[0].x1))
                    new_bndbox.append(int(bbs_aug.bounding_boxes[0].y1))
                    new_bndbox.append(int(bbs_aug.bounding_boxes[0].x2))
                    new_bndbox.append(int(bbs_aug.bounding_boxes[0].y2))
                    newboxlist.append(new_bndbox)
                    # 修改xml tree 并保存
                    newname=str(g_nSum)
                change_xml_annotation(xmlpath, dxmlpath,filenamex[0],newname, newboxlist)
    

def renamexml():
    imgfilelist=os.listdir(dimgpath)
    index=1
    for files in imgfilelist:
        if files.endswith('.jpg'):
            ##重命名文件
            filenamex=files.split('.')
            temnIndex=index+24972
            imgnew_name=str(temnIndex)+'.jpg'
            xmlnew_name=str(temnIndex)+'.xml'
            os.rename(os.path.join(dimgpath,files),os.path.join(dimgpath,imgnew_name))
            
            oldpath=os.path.join(dxmlpath,str(filenamex[0])+'.xml')
            os.rename(oldpath,os.path.join(dxmlpath,xmlnew_name))
            path=os.path.join(dxmlpath,xmlnew_name)
            #print(path)
            in_file = open(path)
            tree = ET.parse(in_file)
            root = tree.getroot()
            filename=root.find('filename')
            filename.text=imgnew_name
            tree.write(path)
            index+=1
            
            ##重命名xml内部名称
            
if __name__ == "__main__":
    global g_nSum
    g_nSum = 0
    print('expanding ..1...rotate 90')
	#旋转90
    seq=iaa.SomeOf(1,[iaa.Affine(rotate=90)])
    func0('m',seq)
  	#旋转180
    print('expanding ..2...rotate 180')
    seq=iaa.SomeOf(1,[iaa.Affine(rotate=180)])
    func0('m',seq)  
    #up垂直镜像
    print('expanding ..3...mirror vertically')
    seq = iaa.Sequential(iaa.Flipud(1))
    func0('h',seq)
    #lr水平镜像   
    print('expanding ..4...Horizontal')
    seq = iaa.Sequential(iaa.Fliplr(1))
    func0('g',seq)
    #增亮
    print('expanding ..5...Brightening')
    seq = iaa.Sequential(iaa.Multiply((1.2, 1.5)))
    func0('e',seq)
    #加高斯噪声
    print('expanding ..6...Gaussian noise-1')
    seq = iaa.AdditiveGaussianNoise(1) 
    func0('c',seq)
    #加高斯噪声
    print('expanding ..7...Gaussian noise-4')
    seq = iaa.AdditiveGaussianNoise(4) 
    func0('d',seq)
    print('expanding finish! Total number of files: %s' % str(g_nSum))