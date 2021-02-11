# Visual Odometry
## Introduction:
Visual odometry(VO) is the process of determining the position and orientation of a robot by analyzing the associated camera images. The project is designed to estimate the motion of calibrated camera mounted over a mobile platform. Motion is estimated by computing the feature points of the image and determining the relative translation and rotation of the images.
## basic pipeline:
![](https://i.imgur.com/BElBW2r.png)
1. Image sequences 
2. Feature Detection 
3. Feature Matching (or Tracking) 
4. Motion Estimation 
    * 2-D to 2-D 
    * 3-D to 3-D 
    * 3-D to 2-D

## Different Approches used in the project: 
* 2D-2D Motion Estimation using Feature matching method. 
* 2D-2D Motion Estimation using Optical Flow Method.
* 3D-2D Motion estimation using Optical Flow method. 
* 3D-3D Motion Estimation using Feature matching method.

### 2D-2D Motion Estimation(Feature matching): 
* #### Approach:
    1) First image(I1) and Second image(I2) was Captured using calibrated monocular camera setup and Features were computed in both images using sift Feature Detector.
    2) Corresponding features were matched using FlannBasedMatcher/ brute force and accuracy was maintained using ratio test. 
    3) Using matched features essential matrix for image pair I1, I2 was computed.
    4) Decompose an essential matrix into a rotation matrix and a translation vector upto scale ambiguity.
    5) Relative 3D Point cloud was computed by triangulating the image pair.
    6) Repeat the process and compute point cloud for the next corresponding image pair.
    7) Relative scale was computed by taking the mean of distances between the consecutive points clouds obtained from keypoint matching of subsequent images and rescale translation accordingly.
    8) Concatenate transformation and repeat the process.
    9) [2D-2D Feature Matching Approch code](https://github.com/sushlokshah/visual-odometry/blob/main/monocular%20VO/with_scale.ipynb)
* #### reference paper :[2D-2D Feature Matching](https://github.com/sushlokshah/visual-odometry/blob/main/monocular%20VO/with_scale.ipynb)
* #### problems: 
    * slow computation due to feature matching and more computation complexity due to sift feature extraction method and slow brute force matching 

* #### remedy:
    * use feature tracking methods instead of feature matching and using fast feature extranction method which we used in the next Approch 2D-2D Motion Estimation(Feature tracking)
    * [2D-2D Feature tracking Approch code](https://github.com/sushlokshah/visual-odometry/blob/main/monocular%20VO/with_scale.ipynb)

* #### Result:


### 2D-2D Motion Estimation(Feature tracking):
* #### Approach:

    1) First image(I1) and Second image(I2) was Captured and Features were computed in the First image using the fast feature detection method.
    2) Features of I1 were tracked in I2 using Lucas Kanade optical flow method.
    3) Calculate tracked features calculate essential, rotation matrix, translation matrix, and relative scale between images as explained above.
    4) Track features in the next frames and concatenates transformation.
    5) Update the reference frame when a sufficient number of features were not tracked and repeat the process.
    6) [2D-2D Feature tracking Approch code](https://github.com/sushlokshah/visual-odometry/blob/main/monocular%20VO/with_scale.ipynb)

* #### Result:

* #### improvements: 
    * improvements in **time and space complexity** of the code with considerable improvement in **frame processing rate**.


### 3D-2D Motion Estimation: 
* #### Approach:
    1. First image(I1) and Second image(I2) was Captured and Features were computed in the First image using the fast feature detection method.
    2. then using 2d to 2d approch rotation and translation is computed between first two images. then these points are triangulated to get 3d point clouds.
    3. untill the reprojection error between reprojected 3d point and current frame is greater than some threshold motion is estimated using EPNP.
    4. else the reference frame is changed from I1 to Ir where r < n(current frame age).
    5. this process is repeated.
    6. [3D-2D implementation code](https://github.com/sushlokshah/visual-odometry/blob/main/monocular%20VO/with_scale.ipynb)

* #### Result:


### 3D-2D Motion Estimation with multiframe tracking approch:
* #### [reference paper](https://)
* problem with above approches implemented was that the reprojection error was continuously goes on increasing with no of frames and the there was no relation between the reprojection error of all previous frames with the current frame which resulted in the accumulation of error in translation and rotation estimation.

