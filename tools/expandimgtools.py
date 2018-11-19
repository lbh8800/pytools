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
import sys
import pickle,gzip
from matplotlib import pyplot
import shutil
'''
imgpath='/home/hxjy/lj/data/ads5_v2/src/JPEGImages'
dimgpath0='/home/hxjy/lj/data/ads5_v2/src/atrophy'
dimgpath1='/home/hxjy/lj/data/ads5_v2/src/cancer'
dimgpath2='/home/hxjy/lj/data/ads5_v2/src/polyp'
dimgpath3='/home/hxjy/lj/data/ads5_v2/src/gist'
dimgpath4='/home/hxjy/lj/data/ads5_v2/src/ulcer'
xmlpath='/home/hxjy/lj/data/ads5_v2/src/Annotations'
 '''
def sortimg2folder():
    filelist = os.listdir(xmlpath)
    index = 1
    for files in filelist:
        if files.endswith('.xml'):
            ##重命名文件
            filenamex = files.split('.')
            #print (files)
            imgfile=filenamex[0]+'.jpg'
            in_file = open(os.path.join(xmlpath, files))
            tree = ET.parse(in_file)
            root = tree.getroot()
            objects = root.find('object')
            classname=objects.find('name').text
            print (classname)
            if classname=='atrophy':
                shutil.move(os.path.join(imgpath, imgfile),os.path.join(dimgpath0, imgfile))
            elif classname=='cancer':
                shutil.move(os.path.join(imgpath, imgfile), os.path.join(dimgpath1, imgfile))
            elif classname=='polyp':
                shutil.move(os.path.join(imgpath, imgfile), os.path.join(dimgpath2, imgfile))
            elif classname=='gist':
                shutil.move(os.path.join(imgpath, imgfile), os.path.join(dimgpath3, imgfile))
            elif classname=='ulcer':
                shutil.move(os.path.join(imgpath, imgfile), os.path.join(dimgpath4, imgfile))
            ##重命名xml内部名称

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
        new_xmax = new_target[2]
        new_ymax = new_target[3]
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

def func0(nameprex,seq,imgpath,dimgpath):

    filelist=os.listdir(imgpath)
    for files in filelist:
        if files.endswith('.jpg'):
            print('current file: %s' % files)
            ##重命名文件
            filenamex=files.split('.')
            ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            new_imgname=nameprex+str(files)
            dstimgpath=os.path.join(dimgpath,new_imgname)
            imagefile=os.path.join(imgpath,files)
            img=cv2.imread(imagefile)
        
            img=np.array(img)

            image_aug = seq.augment_image(img)

            cv2.imwrite(dstimgpath,image_aug)



def read_xml_annotation2(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    objects = root.findall('object')
    bndboxs = []
    i = 1
    for i in range(len(objects)):
        xobject = objects[i]
        bndbox = xobject.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        bndboxs.append([xmin, ymin, xmax, ymax])
    return bndboxs


def cropimg():
    filelist = os.listdir(imgpath1)
    for files in filelist:
        if files.endswith('.jpg'):
            print('current file: %s' % files)
            ##重命名文件
            filenamex = files.split('.')
            ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            imagefile = os.path.join(imgpath1, files)
            img = cv2.imread(imagefile)
            bndboxlist = read_xml_annotation2(xmlpath1, str(filenamex[0]) + '.xml')
            # print(bndboxlist)
            if bndboxlist != None:
                for m in range(len(bndboxlist)):
                    bndbox = bndboxlist[m]
                    width = bndbox[2] - bndbox[0]
                    height = bndbox[3] - bndbox[1]
                    cropImg = img[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
                    if width >= height:
                        cropimg = cropImg[0:height, 0:height]
                    # print(bndbox[0],bndbox[1],bndbox[2],bndbox[3])
                    else:
                        cropimg = cropImg[0:width, 0:width]
                    # cropImg=img[100:200,50:60]
                    ipath = str("{}{}{}.jpg").format(dimgpath1, filenamex[0], m)
                    cv2.imwrite(ipath, cropimg)
            else:
                print(files)


def cropimg2():
    imgin='/home/hxjy/lj/data/v2/src/ulcer'
    imgout='/home/hxjy/lj/data/v2/src/ulcercrop/'
    xmlpath1='/home/hxjy/lj/data/v2/src/Annotations'
    filelist = os.listdir(imgin)
    for files in filelist:
        if files.endswith('.jpg'):
            print('current file: %s' % files)
            ##重命名文件
            filenamex = files.split('.')
            ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            imagefile = os.path.join(imgin, files)
            img = cv2.imread(imagefile)
            bndboxlist = read_xml_annotation2(xmlpath1, str(filenamex[0]) + '.xml')
            # print(bndboxlist)
            if bndboxlist != None:
                for m in range(len(bndboxlist)):
                    #func0('e',seq)
                    bndbox = bndboxlist[m]
                    width = bndbox[2] - bndbox[0]
                    height = bndbox[3] - bndbox[1]
                    if width >= 100 and height >= 100:
                        cropImg = img[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
                        if width >= height:
                            cropimg = cropImg[0:height, 0:height]
                            # print(bndbox[0],bndbox[1],bndbox[2],bndbox[3])
                        else:
                            cropimg = cropImg[0:width, 0:width]
                            # cropImg=img[100:200,50:60]
                        ipath = str("{}{}{}.jpg").format(imgout, filenamex[0], m)
                        cv2.imwrite(ipath, cropimg)
            else:
                print(files)


def gz2jpg():

    print('Loading data from mnist.pkl.gz ...')
    with gzip.open('mnist.pkl.gz','rb') as f:
        train_set, valid_set, test_set=pickle.load(f)

    imgs_dir='mnist'
    os.system('mkdir -p {}'.format(imgs_dir))
    datasets = {'train': train_set, 'val': valid_set, 'test': test_set}

    for dataname,dataset in datasets.items():
        print('Converting {} fataset ...'.format(dataname))
        data_dir= os.sep.join([imgs_dir, dataname])
        os.system('mkdir -p {}'.format(data_dir))

        for i, (img, label) in enumerate(zip(*dataset)):
            filename= '{:0>6d}_{}.jpg'.format(i, label)
            filepath=os.sep.join([data_dir,filename])
            img=img.reshape((28,28))
            pyplot.imsave(filepath,img,cmap='gray')
            if (i+1) %10000 == 0:
                print('{} images converted!'.format(i+1))



def gen_caffe_imglist():
    input_path='/home/hxjy/lj/data/v1/val'
    output_path='/home/hxjy/lj/data/v1/val/val.txt'
    filenames = os.listdir(input_path)
    with open(output_path,'w') as f:
        for filename in filenames:
            filepath=os.sep.join([input_path,filename])
            label = filename[:filename.rfind('.')].split('_')[0]

            if label=='0':
                label='0'
            if label=='1':
                label='1'

            if label=='2':
                label='2'
            if label=='3':
                label='3'
            line='{} {}\n'.format(filename,label)
            f.write(line)


def resizeimg():
    input_path='/home/hxjy/lj/data/v2/T11/0'
    output_path='/home/hxjy/lj/data/v2/T11/0'
    filenames = os.listdir(input_path)
    for filename in filenames:
        if filename.endswith('.jpg'):
            filepath = os.path.join(input_path, filename)
            img=cv2.imread(filepath)
            img=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
            filepath1=os.path.join(output_path,filename)
            print(filepath1)
            cv2.imwrite(filepath1,img)


def renameimg():
    srcimgpath = '/home/hxjy/lj/data/v2/T11/3'
    imgfilelist = os.listdir(srcimgpath)
    index = 1
    for files in imgfilelist:
        if files.endswith('.jpg'):
            ##重命名文件
            filenamex = files.split('.')
            imgnew_name = '3_' + str(index) + '.jpg'
            os.rename(os.path.join(srcimgpath, files), os.path.join(srcimgpath, imgnew_name))
            index += 1

            ##重命名xml内部名称

def expandimg():
    imgpath = '/home/hxjy/lj/data/v2/T11/3'
    path = '/home/hxjy/lj/data/v2/T11/3'

    # 高斯模糊`：
    # seq = iaa.Sequential(iaa.GaussianBlur(sigma=(1, 3.0)))
    # func0('a3',seq)
    # 增加对比度

    seq = iaa.ContrastNormalization((1.4, 1.8))
    func0('f', seq, imgpath=imgpath, dimgpath=path)
    # 高斯模糊
    # seq = iaa.Sequential(iaa.GaussianBlur(sigma=(1.5, 4.0)))
    # func0('b3',seq,imgpath=imgpath,dimgpath=path)
    # 加高斯噪声
    # seq = iaa.AdditiveGaussianNoise(1)
    # func0('c3',seq)
    # 加高斯噪声
    seq = iaa.AdditiveGaussianNoise(4)
    func0('d3', seq, imgpath=imgpath, dimgpath=path)

    # 增亮注意名称错了用了up1.jpg
    seq = iaa.Sequential(iaa.Multiply((0.8, 0.8)))
    func0('e3', seq, imgpath=imgpath, dimgpath=path)
    # seq = iaa.Sequential(iaa.Multiply((1.1, 1.1)))
    # func0('e22',seq)

    # seq = iaa.Sequential(iaa.CropAndPad(percent=(-0.2, 0)))
    # func0('p2', seq)
    seq = iaa.Sequential(iaa.CropAndPad(percent=(-0.2, 0)))
    func0('p2P', seq, imgpath=imgpath, dimgpath=path)

    # lr水平镜像
    # seq = iaa.Sequential(iaa.Fliplr(1))
    # func0('g2',seq)
    # up垂直镜像
    seq = iaa.Sequential(iaa.Flipud(1))
    func0('h2', seq, imgpath=imgpath, dimgpath=path)

    # seq=iaa.SomeOf(1,[iaa.Affine(rotate=40),iaa.Affine(rotate=30),iaa.Affine(rotate=60),
    #                  iaa.Affine(rotate=120)])
    # func0('m',seq)
    seq = iaa.Sequential(iaa.CropAndPad(percent=(-0.3, 0)))
    func0('p4', seq, imgpath=imgpath, dimgpath=path)
def searchimg():
    xmlpath=''
    srcimgpath=''
    dstimgpath=''
    filelist = os.listdir(xmlpath)
    index = 1
    for files in filelist:
        if files.endswith('.xml'):
            ##重命名文件
            filenamex = files.split('.')
            #print (files)
            imgfile=filenamex[0]+'.jpg'
            in_file = open(os.path.join(xmlpath, files))
            tree = ET.parse(in_file)
            root = tree.getroot()
            objects = root.find('object')
            classname=objects.find('name').text
            if classname == 'atrophy':
                shutil.move(os.path.join(srcimgpath, imgfile), os.path.join(dstimgpath, imgfile))
            elif classname == 'cancer':
                shutil.move(os.path.join(srcimgpath, imgfile), os.path.join(dstimgpath, imgfile))

if __name__ == "__main__":
    print('expanding finish!')
    print('renameing.....')
    #renameimg()
    #sortimg()
    #cropimg2()
    #resizeimg()
    #gen_caffe_imglist()
    searchimg()
    print('finish!')