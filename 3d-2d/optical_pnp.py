#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
from os import listdir
import random
import numpy as np
import cv2 as cv


# In[2]:


#loading dataset
dataset_location = 'C:/Users/sushl/Downloads/Visual-Odometry-master/Visual-Odometry-master/KITTI_sample/images'
#dataset_location = 'C:/Users/sushl/Downloads/Archives/data_odometry_gray/dataset/sequences/01/image_0'
#dataset_location = 'C:/Users/sushl/Desktop/visual odometry/vnit_dataset'
L = os.listdir(dataset_location)
L.sort()


# In[3]:


#ground Truth
ground_truth = np.loadtxt('C:/Users/sushl/Downloads/Visual-Odometry-master/Visual-Odometry-master/KITTI_sample/poses.txt',delimiter = ' ')
#ground_truth = np.loadtxt('C:/Users/sushl/Desktop/New Text Document.txt',delimiter = ' ')
#ground_truth.shape
gx = ground_truth[:,3]
gz = ground_truth[:,11]


# In[4]:


#calibration matrix

k =np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
             [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
             [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
"""
k = np.array([[518.56666108, 0., 329.45801792],
    [0., 518.80466479, 237.05589955],
    [  0., 0., 1.]])
"""
sift = cv.SIFT_create()
bf = cv.BFMatcher()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()


# In[5]:


def keypoints(img):
    img = cv.equalizeHist(img)
    fast = cv.FastFeatureDetector_create()
    fast.setNonmaxSuppression(1)
    fast.setThreshold(70)
    kp = fast.detect(img,None)
    pt = np.float32([ kp[m].pt for m in range(len(kp))]).reshape(-1,1,2)
    return kp,pt


# In[6]:


def optical_flow_matches(img,trig_frame,pt1,window_size):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (window_size,window_size),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    ptimg, st, err = cv.calcOpticalFlowPyrLK(trig_frame, img, pt1, None, **lk_params)

    pt1i, st, err = cv.calcOpticalFlowPyrLK(img, trig_frame, ptimg, None, **lk_params)
    return ptimg,pt1i


# In[7]:


def point3D(k,R,t,pts1,pts2):
    rt = np.zeros((3,4))
    rt[:3,:3] = np.identity(3)
    projMatr1 = k@rt
    rt2 = np.zeros((3,4))
    rt2[:3,:3] = R
    rt2[:,3] = t.reshape((3))
    projMatr2 = k@rt2
    points4D = cv.triangulatePoints(projMatr1,projMatr2,pts1 ,pts2)
    points3D = points4D / points4D[3,:]
    return points3D.T[:,:3]


# In[8]:


"""
#for first two images
img1 = cv.imread(dataset_location +'/'+ L[0],cv.IMREAD_GRAYSCALE)
img2 = cv.imread(dataset_location +'/'+ L[1],cv.IMREAD_GRAYSCALE)
pts,kp1= keypoints(img1)
kp2,kp1 = optical_flow_matches(img2,img1,kp1,10)

E, mask = cv.findEssentialMat(kp1,kp2,k,cv.RANSAC, prob = 0.999,threshold = 0.4,mask=None)
#inlier points
pts1 = kp1[mask.ravel()==1]
pts2 = kp2[mask.ravel()==1]
matches = {}
#print(tuple(kp1[0].flatten()))
for i in range(len(pts2)):
    matches[tuple(pts1[i].flatten())] = tuple(pts2[i].flatten())
print(matches)
#recovering pose info
retval, R, t, mask = cv.recoverPose(E, pts1, pts2, k)
#print(t)
pointcloud = point3D(k,R,t,pts1,pts2)
retval, rvec, t, inliers = cv.solvePnPRansac(pointcloud,pts2, k, (0,0,0,0),useExtrinsicGuess = True ,iterationsCount = 100,reprojectionError = 8.0,confidence = 0.90,flags = 1)
R,Jec = cv.Rodrigues(rvec)
print(t,len(inliers))
translations = []
rotations = []
for i in range(3,15):
    img3 = cv.imread(dataset_location +'/'+ L[i],cv.IMREAD_GRAYSCALE)
    kp3,kp2i = optical_flow_matches(img3,img2,pts2,10)
    #pointcloud = point3D(k,R,t,kp1,kp2)
    matches32 = {}
    for i in range(len(pts2)):
        matches32[tuple(pts2[i].flatten())] = tuple(kp3[i].flatten())
    print(matches32)
    #print(len(kp3),len(kp2))
    retval, rvec, tvec, inliers = cv.solvePnPRansac(pointcloud,kp3, k, (0,0,0,0),useExtrinsicGuess = True ,iterationsCount = 100,reprojectionError = 8.0,confidence = 0.90,flags = 1)
    R3,Jec = cv.Rodrigues(rvec)
    print(tvec,len(inliers))
    translations.append(tvec)
    rotations.append(R3)

"""
# In[ ]:





# In[ ]:





# In[9]:


#for first two images
img1 = cv.imread(dataset_location +'/'+ L[0],cv.IMREAD_GRAYSCALE)
img2 = cv.imread(dataset_location +'/'+ L[1],cv.IMREAD_GRAYSCALE)
pts,kp1= keypoints(img1)
kp2,kp1 = optical_flow_matches(img2,img1,kp1,10)

E, mask = cv.findEssentialMat(kp1,kp2,k,cv.RANSAC, prob = 0.999,threshold = 0.4,mask=None)

#inlier points
pts1 = kp1[mask.ravel()==1]
pts2 = kp2[mask.ravel()==1]

#recovering pose info
retval, R, t, mask = cv.recoverPose(E, pts1, pts2, k)
pointcloud = point3D(k,R,t,pts1,pts2)
retval, rvec, t, inliers = cv.solvePnPRansac(pointcloud,pts2, k, (0,0,0,0),useExtrinsicGuess = True ,iterationsCount = 100,reprojectionError = 5.0,confidence = 0.75,flags = 1)
R,Jec = cv.Rodrigues(rvec)
#set of translations and rotations
translations = []
rotations = []
a = 0
t0 = np.zeros((3,1))
r0 = np.identity(3)

#storing previous imgs
framep1 = img1[:]
framep2 = img2[:]
Rp = R[:]
Tp = t[:]
t0 = np.zeros((3,1))
r0 = np.identity(3)
match = 600
count = 0

#for remaining images
while(a < len(L)-2):#len(L)-2
    img3 = cv.imread(dataset_location +'/'+ L[a],cv.IMREAD_GRAYSCALE)
    kp3,kp2i = optical_flow_matches(img3,img2,pts2,10)

    #match = threshold for pnp i.e it inliers of previous frame are greater than this threshold we will use pnp directly
    if match >200 :
        retval, rvec, tvec, inliers = cv.solvePnPRansac(pointcloud,kp3, k, (0,0,0,0),useExtrinsicGuess = True ,iterationsCount = 100,reprojectionError = 8.0,confidence = 0.80,flags = 1)
        R3,Jec = cv.Rodrigues(rvec)
        print(tvec,len(inliers))
        #translations.append(tvec)
        #rotations.append(R3)
        match = len(inliers)
        translations.append(t0 + r0@tvec)
        #print(len(translations))
        rotations.append(r0@R3)
        count = 0

    #else will trigulated point cloud from previous two imgs
    else:
        img1 = cv.imread(dataset_location +'/'+ L[a-2],cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(dataset_location +'/'+ L[a-1],cv.IMREAD_GRAYSCALE)
        pts,kp1= keypoints(img1)
        kp2,kp1 = optical_flow_matches(img2,img1,kp1,15)

        E, mask = cv.findEssentialMat(kp1,kp2,k,cv.RANSAC, prob = 0.999,threshold = 0.4,mask=None)

        #inlier points
        pts1 = kp1[mask.ravel()==1]
        pts2 = kp2[mask.ravel()==1]

        r0 = rotations[a-2]
        t0 = translations[a-2]
        retval, R, t, mask = cv.recoverPose(E, pts1, pts2, k)
        pointcloud = point3D(k,R,t,pts1,pts2)
        a = a-1
        match = 600
        print('me')
        count = count + 1
        if count >3:
            break


    a = a+1


# In[10]:


x0 = []
y0 = []
go = []
g1 = []
for i in range(len(translations)-1):
    y0.append(-1*translations[i][2])
    x0.append(translations[i][0])

fig,axes = plt.subplots()

axes.scatter(x0, y0,label = 'scale using dist ratio')
axes.plot(gx, gz,color = 'red',label = 'ground_truth')
axes.legend()

plt.show()


# In[ ]:
