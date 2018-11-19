# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET 
import  os


xmlpath='./py-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations'
output_path1='./py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt'
output_path2='./py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt'
output_path3='./py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
output_path4='./py-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
def read_xml():
    imgcout = 7991040
    traincout=int((imgcout*0.49)*0.2)
    valcout=int(imgcout*0.21*0.2)
    testcout=int(imgcout*0.3*0.2)
    train_1 =1
    train_2 = 1
    train_3 = 1
    train_4 = 1
    train_5 = 1
    val_1 = 1
    val_2 = 1
    val_3 = 1
    val_4 = 1
    val_5 = 1
    test_1 = 1
    test_2 = 1
    test_3 = 1
    test_4 = 1
    test_5 = 1
    index = 0
    filelist = os.listdir(xmlpath)
    f1=open(output_path1, 'w')
    f2=open(output_path2, 'w')
    f3 = open(output_path3, 'w')
    f4 = open(output_path4, 'w')
    for files in filelist:
        print(index)
        index+=1
        if files.endswith('.xml'):

            line=''
            in_file = open(os.path.join(xmlpath,files))
            tree = ET.parse(in_file)
            root = tree.getroot()
            objects=root.find('object')
            label=objects.find('name').text
            filename=files.split('.')
            #print(label)line
            if label=='cancer':
                line = '{}\n'.format(filename[0])
                if train_1<=traincout:
                    f1.write(line)
                    train_1+=1
                elif val_1<=valcout:
                    f2.write(line)
                    val_1 += 1
                else:
                    f3.write(line)
                    test_1 += 1
            if label=='atrophy':
                line = '{}\n'.format(filename[0])
                if train_2 <= traincout:
                    f1.write(line)
                    train_2 += 1
                elif val_2 <= valcout:
                    f2.write(line)
                    val_2 += 1
                else:
                    f3.write(line)
                    test_2 += 1
            if label=='ulcer':
                line = '{}\n'.format(filename[0])
                if train_3 <= traincout:
                    f1.write(line)
                    train_3 += 1
                elif val_3 <= valcout:
                    f2.write(line)
                    val_3 += 1
                else:
                    f3.write(line)
                    test_3 += 1
            if label=='polyp':
                line = '{}\n'.format(filename[0])
                if train_4 <= traincout:
                    f1.write(line)
                    train_4 += 1
                elif val_4 <= valcout:
                    f2.write(line)
                    val_4 += 1
                else:
                    f3.write(line)
                    test_4 += 1
            if label=='gist':
                line = '{}\n'.format(filename[0])
                if train_5 <= traincout:
                    f1.write(line)
                    train_5 += 1
                elif val_5 <= valcout:
                    f2.write(line)
                    val_5 += 1
                else:
                    f3.write(line)
                    test_5 += 1
                #print(line)
def write2txt():
    f=open(output_path4,'w')
    for line in open(output_path1):
	f.writelines(line)
    for line1 in open(output_path2):
	f.writelines(line1)
    f.close
def countlines():
    count1=0
    count2=0
    count3=0
    for line in open(output_path1):
	count1+=1
    for line in open(output_path2):
	count2+=1
    for line in open(output_path4):
	count3+=1
    print("train count: {}  val count {} cout:{} trainval:{}").format(count1,count2,count1+count2,count3)
if __name__ == "__main__":
    #read_xml()
    #write2txt()
    countlines()
    print('finish')
