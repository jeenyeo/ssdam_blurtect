#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
import string
import ffmpy
import random
import caffe
import cv2
import numpy as np
#from python_wrapper import *
import os
from PIL import Image
import dlib
import shutil
import openface
import face_detection


def video_info(infilename):
    cap = cv2.VideoCapture(infilename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return fps, width, height


class VideoManager():
    
    def __init__(self, fps, to_image_quality=10, to_video_quality=25):
        self.fps = fps
        self.to_image_quality = to_image_quality
        self.to_video_quality = to_video_quality
        
    def _extract_images_from_video(self, input_video_path, output_folder_path): 
        try:     
            video_name = input_video_path.split(os.sep)[-1] 
            video_name_without_extension = video_name.split('.')[0] 
            video_ext = video_name.split('.')[1]   

            # create the output folder for the video in "output_folder_path"
            if not os.path.exists(output_folder_path):
                os.mkdir(output_folder_path)             

            # -qscale:v NUM --> NUM = 2->31, 31 is the worst quality of JPEG images
            # $ ffmpeg -i '/input/path/video.mp4' -vf fps=12 -qscale:v 2 '/output/path/img_%06d.png'
            ff = ffmpy.FFmpeg(
                inputs = { input_video_path: "-y" }, 
                outputs = { output_folder_path + '/img_' + video_name_without_extension 
                    + '_%06d.jpg': '-vf fps=%s -qscale:v %s' % (str(self.fps), str(self.to_image_quality))})    #_%06d = _000000 ...
            ff.run()    
        except Exception as e:
            print("An error has occured! %s" % str(e))
            return False

        return output_folder_path
    

def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() 
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)
    
    
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    

    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x
    
    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet
        
        #print("1: x,y", x,y)
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print("2: x,y", x,y)
    else:
        score = map[x,y]
    '''
   

    score = map[x,y]
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    return boundingbox_out.T


def align_face(im, boxes, imNum, path):
    #embedding 
    predictor_model = "./openface/models/dlib/shape_predictor_68_face_landmarks.dat"

    #aligner
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    ppNum = 0

    for i in range(x1.shape[0]):
        face_rect = dlib.rectangle(left=int(x1[i]), top=int(y1[i]), right=int(x2[i]), bottom=int(y2[i]))
        aligned_face = face_aligner.align(96, im, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        file_name = str(imNum) + '_' + 'alingedimage_' + str(ppNum) + '.jpg'
        cv2.imwrite(os.path.join(path, file_name), aligned_face)
        ppNum += 1

                
from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print(fmt % (time()-_tstart_stack.pop()))


def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    
    img2 = img.copy()

    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m

    
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    # first stage
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype = np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
    
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         

    #####
    # 1 #
    #####
    print("[1]:",total_boxes.shape[0])


    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        print("[2]:",total_boxes.shape[0])
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T


        total_boxes = rerec(total_boxes) # convert box to square
        print("[4]:",total_boxes.shape[0])
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        print("[4.5]:",total_boxes.shape[0])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)


    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))

            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
 
            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))

        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        score = out['prob1'][:,1]
        pass_t = np.where(score>threshold[1])[0]
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        print("[5]:",total_boxes.shape[0])
        
        mv = out['conv5-2'][pass_t, :].T
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                print("[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                print("[8]:",total_boxes.shape[0])
            
        #####
        # 2 #
        #####
        print("2:",total_boxes.shape)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            print("[9]:",total_boxes.shape[0])
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')
                
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    print("[11]:",total_boxes.shape[0])
                    points = points[pick, :]

    #####
    # 3 #
    #####
    print("3:",total_boxes.shape)

    return total_boxes, points

    
def initFaceDetector():
    minsize = 20
    caffe_model_path = "./mtcnn-master/model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
    return (minsize, PNet, RNet, ONet, threshold, factor)


def haveFace(img, facedetector):
    minsize = facedetector[0]
    PNet = facedetector[1]
    RNet = facedetector[2]
    ONet = facedetector[3]
    threshold = facedetector[4]
    factor = facedetector[5]
    
    if max(img.shape[0], img.shape[1]) < minsize:
        return False, []

    img_matlab = img.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp
    
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    containFace = (True, False)[boundingboxes.shape[0]==0]
    return containFace, boundingboxes


def imglist_text(path_dir, title):
    file_list = os.listdir(path_dir)
    file_list.sort()
    f = open(title, 'w')
    for i in file_list:
        f.write(i + '\n')
    f.close()

    path = os.path.split(path_dir)
    save_align_path = "./aligned-images/" + path[len(path) - 1] + "_aligned"

    if os.path.exists(save_align_path):
        shutil.rmtree(save_align_path)
        os.mkdir(save_align_path)
    else:
        os.mkdir(save_align_path)

    return save_align_path


def result_images(imglistfile, imagepath, save_align_path, imgNum):
    
    minsize = 20

    caffe_model_path = "./model"

    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)


    #error = []
    f = open(imglistfile, 'r')
    for imgpath in f.readlines():
        imgNum += 1
        imgpath = imagepath + imgpath.split('\n')[0]
        print("######\n", imgpath)
        img = cv2.imread(imgpath)
 
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp


        boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)

        align_face(img, boundingboxes, imgNum, save_align_path)

        
    f.close()
    

def main():

    if os.path.exists('./training-images/model'):
        shutil.rmtree('./training-images/model')
        os.mkdir('./training-images/model')
    else:
        os.mkdir('./training-images/model')

    VideoManager1 = VideoManager(10)
    VideoManager1._extract_images_from_video('sample_video.mp4', './training-images/model') # FIX ME : short video for face recognition (Don't blur me~~)
    
    imgNum = 0

    minsize = 20

    caffe_model_path = "./model"

    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
        
    #Unknowns_align_path = imglist_text('./training-images/Unknowns', 'imglist_Unknowns.txt')
    model_align_path = imglist_text('./training-images/model', 'imglist_model.txt')

    result_images('imglist_model.txt', './training-images/model/', model_align_path, imgNum)
    #result_images('imglist_Unknowns.txt','./training-images/Unknowns/', Unknowns_align_path, imgNum)

    # Loading model
    predictor_model = "./openface/models/dlib/shape_predictor_68_face_landmarks.dat"
    face_aligner = openface.AlignDlib(predictor_model)

    # Training part
    model_path = "./openface/models/openface/nn4.small2.v1.t7"

    with face_detection.FaceDetector(torch_net_model=model_path) as fd:
        fd.append_dir("Unknowns", "./aligned-images/Unknowns_aligned")
        fd.append_dir("model", "./aligned-images/model_aligned")
        
        fd.train_model()
    
        fd.save('./face_detector')

    # Prediction part
        
    with face_detection.FaceDetector.load('./face_detector') as fd:

        cv2.namedWindow('Window')
        cv2.moveWindow('Window', 20, 30)

        cap = cv2.VideoCapture()
        cap.open('test_video.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps, width, height = video_info('test_video.mp4') # FIX ME : original video for blur
        out = cv2.VideoWriter('output_video_.mp4', fourcc, int(fps), (int(width), int(height)))

        while cap.isOpened():
            
            ret, frame = cap.read()
            if frame is None :
                break

            frame_matlab = frame.copy()
            tmp = frame_matlab[:,:,2].copy()
            frame_matlab[:,:,2] = frame_matlab[:,:,0]
            frame_matlab[:,:,0] = tmp

            boundingboxes, points = detect_face(frame_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
            
            x1 = boundingboxes[:,0]
            y1 = boundingboxes[:,1]
            x2 = boundingboxes[:,2]
            y2 = boundingboxes[:,3]
            
            for i in range(x1.shape[0]):
                face_box = frame[(int(y1[i])):(int(y2[i])), (int(x1[i])):(int(x2[i]))]
                face_rect = dlib.rectangle(left=int(x1[i]), top=int(y1[i]), right=int(x2[i]), bottom=int(y2[i]))
                    
                aligned_face = face_aligner.align(96, frame, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                label = fd.predict(aligned_face)
                if (label == 'Unknowns'):
                    blur = cv2.medianBlur(face_box, 33)
                    frame[(int(y1[i])):(int(y2[i])), (int(x1[i])):(int(x2[i]))] = blur        

            cv2.imshow('Window', frame)
            out.write(frame)
            
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
