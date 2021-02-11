# visual-odometry
## Introduction:
Visual odometry(VO) is the process of determining the position and orientation of a robot by analyzing the associated camera images. The project is designed to estimate the motion of calibrated camera mounted over a mobile platform. Motion is estimated by computing the feature points of the image and determining the relative translation and rotation of the images.
## Visual odometry pipeline: 
1 Image sequences  
2 Feature Detection  
3 Feature Matching (or Tracking)  
4 Motion Estimation (2-D-to-2-D 3-D-to-3-D 3-D-to-2-D)  

* **Odometry**  - Use of data from motion sensors( eg. wheel rotation) / images to estimate change in position over time i.e trajectory .
* **Visual odometry (VO)**- When this is done with the aid of Visual data i.e Images / Video to estimate the egomotion of camera

### basic pipeline of VO
![](https://i.imgur.com/BElBW2r.png)

#### scope
* We can fuse the camera setup with sensors like IMU and LIDAR to increase the accuracy for implementation in SLAM.
* 3D reconstruction also works on the same basic principle of VO.
* VO plays a major role in space or planet rovers to navigate and observe the environment.

### APPROACH
there are three different approaches based on the information available and utilised for the state estimation
* 2d to 2d motion estimation : using only 2d information available from images.(monocular approach)
* 3d to 2d motion estimation : using some known 3d point in the environment to estimate camera loaction with respect to some global reference.
* 3d to 3d motion estimation : using 3d point clouds triangulated from the images computing the tranformation.(stereo approach)
