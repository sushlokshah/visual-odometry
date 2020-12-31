# monocular VO
### 2d to 2d approach
#### Approach
* the main objective of this project is to find relative translation and rotation using only the 2d image without knowing the 3d point information.
* so for that main point is the extraction of features from the images and I used sift approch to find the features from the images.
* from the feature correspondance obtained by using brute force method computer essential matrix (which is the algebraic representation of epipolar geometry derived by exploiting the coplanarity constraint to image point and the 3d point).

![](https://i.imgur.com/Rs4enVi.png)

* this essential matrix hold the imformation about the relative rotation and tranlation between .

![](https://i.imgur.com/AlZwpl6.jpg)

* Decompose essential matrix to get relative rotation and translation vector
* but due lose the depth information and thus don’t get the actual translation from the 2d images.
* so we have to find out scale factor which is multiplied to the translation vector to get the actual translation.
* computer 3d point cloud to get the scale factor.
* repeat this process for all sequence of images

![](https://i.imgur.com/qiGAgSk.png)

### results
implemented the basic vanilla pipeline without bundle adjustment on KITTI dataset with a Monocular camera setup and plotted the trajectory with the ground truth.

![](https://i.imgur.com/0cXtEgU.png)

![](https://i.imgur.com/X7tF6KR.png)

