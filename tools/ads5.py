#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from time import ctime,sleep
import threading
import copy
import socket


CLASSES = ('__background__',
           'cancer','atrophy','gist','ulcer','polyp')


NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final-V7-2.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

CONF_THRESH = 0.5
NMS_THRESH = 0.3



def mat_inter(box1,box2):
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False

def solve_coincide(box1,box2):
    if mat_inter(box1,box2)==True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col=min(x02,x12)-max(x01,x11)
        row=min(y02,y12)-max(y01,y11)
        intersection=col*row
        area1=(x02-x01)*(y02-y01)
        area2=(x12-x11)*(y12-y11)
        coincide=100.00*intersection/(area1+area2-intersection)

        return coincide
    else:
        return 0
def recvall(sock,count):
    buf=b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:return None
        buf+=newbuf
        count-=len(newbuf)
    return buf

def im_print():
    global g_bboxTem
    global g_scoreTem
    global g_classTem
    global g_box
    global g_box_Last
    sleep(2)
    global im
    global scores
    global boxes
    
    print 'imprint' 
    #print boxes
    temp_im = im
    #for i in range(5000):
    while 1:
	scores1 = np.copy(scores)
	boxes1 = np.copy(boxes)
	g_scoreTem = -1
	#print 'im_print'
	#print boxes
	#print scores
	for cls_ind, cls in enumerate(CLASSES[1:]):
		if cls == 'cancer':
                    cls_ind += 1 # because we skipped background
	    	    cls_boxes = boxes1[:, 4*cls_ind:4*(cls_ind + 1)]
		    cls_scores = scores1[:, cls_ind]
		    dets = np.hstack((cls_boxes,
		                  cls_scores[:, np.newaxis])).astype(np.float32)
		    keep = nms(dets, NMS_THRESH)
		    dets = dets[keep, :]
		    vis_detections1(im, cls, dets, thresh=CONF_THRESH,temp=0)
		if cls == 'atrophy':
                    cls_ind += 1 # because we skipped background
	    	    cls_boxes = boxes1[:, 4*cls_ind:4*(cls_ind + 1)]
		    cls_scores = scores1[:, cls_ind]
		    dets = np.hstack((cls_boxes,
		                  cls_scores[:, np.newaxis])).astype(np.float32)
		    keep = nms(dets, NMS_THRESH)
		    dets = dets[keep, :]
		    vis_detections1(im, cls, dets, thresh=CONF_THRESH,temp=0)
		if cls == 'ulcer':
                    cls_ind += 1 # because we skipped background
	    	    cls_boxes = boxes1[:, 4*cls_ind:4*(cls_ind + 1)]
		    cls_scores = scores1[:, cls_ind]
		    dets = np.hstack((cls_boxes,
		                  cls_scores[:, np.newaxis])).astype(np.float32)
		    keep = nms(dets, NMS_THRESH)
		    dets = dets[keep, :]
		    vis_detections1(im, cls, dets, thresh=CONF_THRESH,temp=0)
		if cls == 'polyp':
                    cls_ind += 1 # because we skipped background
	    	    cls_boxes = boxes1[:, 4*cls_ind:4*(cls_ind + 1)]
		    cls_scores = scores1[:, cls_ind]
		    dets = np.hstack((cls_boxes,
		                  cls_scores[:, np.newaxis])).astype(np.float32)
		    keep = nms(dets, NMS_THRESH)
		    dets = dets[keep, :]
		    vis_detections1(im, cls, dets, thresh=CONF_THRESH,temp=0)
		if cls == 'gist':
                    cls_ind += 1 # because we skipped background
	    	    cls_boxes = boxes1[:, 4*cls_ind:4*(cls_ind + 1)]
		    cls_scores = scores1[:, cls_ind]
		    dets = np.hstack((cls_boxes,
		                  cls_scores[:, np.newaxis])).astype(np.float32)
		    keep = nms(dets, NMS_THRESH)
		    dets = dets[keep, :]
		    vis_detections1(im, cls, dets, thresh=CONF_THRESH,temp=0)
	if g_scoreTem > -1:
	    boxTem = (g_bboxTem[0], g_bboxTem[1],g_bboxTem[2], g_bboxTem[3])
	    if solve_coincide(g_box,boxTem) < 50:
                g_box = boxTem
                g_box_Last = g_bboxTem
	    cv2.rectangle(im,(g_box_Last[0], g_box_Last[1]),(g_box_Last[2], g_box_Last[3]),(0,255,0),1)
	    cv2.putText(im, '{:s} {:.3f}%'.format(g_classTem, g_scoreTem*100),(int(g_box_Last[0]), int(g_box_Last[1]) - 2),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1)
	try:    	
	    cv2.imshow("Endoscope", im)
    	    cv2.waitKey (35)
        except:
	    continue

def listenToClient(client,address):
    print 'recv data'
    global dir
    global im
    while True:
        try:
            #clientSock,(remoteHost,remotePort) = sock.accept()
            #print("[%s:%s] connect" % (remoteHost, remotePort))
            length = recvall(client,16)
            #print 'length',length
            stringData=recvall(client,long(length))
	    if stringData:
                data_out=np.fromstring(stringData,dtype='uint8')
            #print 'data_out:',data_out
            #if not data_out == null:
                im = cv2.imdecode(data_out,1)
	    else:
		raise error('Client disconnected')
        #cv2.imshow('SERVER',decimg1)
        #cv2.imshow("Old", im)
        #if cv2.waitKey(33) == 27:
        #    break
        #sendDataLen = clientSock.send("this is send data from server")
        #print "recvData:", recvData
        #print "sendDataLen", sendDataLen
	except:
	    client.close()
	    return False

def get_im():
    global dir
    global im
    #capture = cv2.VideoCapture(dir)
    #if capture.isOpened():
        #while True:
            #ret, prev = capture.read()
            #if ret==True:
		#im = prev
		#sleep(0.034)
		#cv2.imshow("Old", prev)
	    	#cv2.waitKey (33)
		#print 'get_im'
            #else:
                #break

    sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.bind(('',8887))
    sock.listen(10)
    while True:
	client, address = sock.accept()
	client.settimeout(60)
	print '111'
	threading.Thread(target = listenToClient, args = (client,address)).start()
	#threading.Thread( args = (client,address)).start()
	print '222'
    #clientSock,(remoteHost,remotePort) = sock.accept()
    #print("[%s:%s] connect" % (remoteHost, remotePort))

    #while 1:
        
    sock.close()



def vis_detections(fig,ax,im, class_name, dets, thresh=0.5,temp=0):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    #plt.draw()
    #plt.show()
    #sleep(3)

def vis_detections1(im, class_name, dets, thresh=0.5,temp=0):
    global g_bboxTem
    global g_scoreTem
    global g_classTem
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
	im = im[:, :, (2, 1, 0)]
    if -1 == g_scoreTem:
        g_bboxTem = dets[0, :4]
        g_scoreTem = dets[0, -1]
        g_classTem = class_name
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > g_scoreTem:
	    g_bboxTem = bbox
            g_scoreTem = score
            g_classTem = class_name
              


def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    
    scores, boxes = im_detect(net, im)
    timer.toc()
    temp=1;
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3

    #im = im[:, :, (2, 1, 0)]
    #plt.ion()
    #fig, ax = plt.subplots(figsize=(12, 12))
    #fig = 1
    #ax.imshow(im, aspect='equal')

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections1(im, cls, dets, thresh=CONF_THRESH,temp=0)
    cv2.imshow("Endoscope", im)
    cv2.waitKey (33)
    #plt.draw()
    #plt.show()
    #plt.clf()
    #sleep(2)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    global g_bboxTem
    global g_scoreTem
    global g_classTem
    global g_box
    global g_box_Last
    g_box=(0,0,2,2)
    g_box_Last=[0,0,2,2]
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    global im
    global scores
    global boxes
    args = parse_args()
    cv2.namedWindow("Endoscope",cv2.WND_PROP_FULLSCREEN)
    #cv2.namedWindow("Endoscope",cv2.cv.CV_WINDOW_NORMAL)
    cv2.setWindowProperty("Endoscope",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    #cv2.namedWindow("Old")
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    #dir=cfg.DATA_DIR+"/demo"
    #print dir
    #for root,dirs,files in os.walk(dir):
    #    for im_name in files:
            #print os.path.join(root,file)
    
    #im_names = ['1_15_60_flip.jpg', '1_16_34.jpg', '1_20_63_flip.jpg',
    #            '6_2_102_flip.jpg', '6_2_102.jpg']
    #for im_name in im_names:
    #    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #        print 'Demo for data/demo/{}'.format(im_name)
    #        demo(net, im_name)

    #plt.show()

    global dir
    dir= cfg.DATA_DIR+"/demo/"+"ulcer.avi"
    print dir
    
    t1 = threading.Thread(target=get_im)
    t1.start()
    t2 = threading.Thread(target=im_print)
    t2.start()
    
    capture = cv2.VideoCapture(dir)

    temp_int=0
    num = 0
    

    while True:
    	if temp_int==0:
			#demo(net,im)
			"""Detect object classes in an image using pre-computed object proposals."""

		    	# Load the demo image
		    	#im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
		    	#im = cv2.imread(im_file)

		    	# Detect all object classes and regress object bounds
			
			timer = Timer()
		    	timer.tic()
		 
		    	scores, boxes = im_detect(net, im)
			#print 'main'
			#print boxes
			#print scores
		    	timer.toc()
		    	#print ('Detection took {:.3f}s for '
			#   '{:d} object proposals').format(timer.total_time, boxes.shape[0])
			temp_int+=1
	else:
			sleep(0.034)
			if temp_int == 2:
				temp_int = 0
			else:
				temp_int+=1



    


    #for t in threads:
    #    t.setDaemon(True)
    #    t.start()
    t1.join()
    t2.join()
    
    
