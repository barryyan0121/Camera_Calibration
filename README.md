# 畸变校正测试报告
## 畸变产生的原因

相机的成像过程实质上是坐标系的转换。首先空间中的点由 “世界坐标系” 转换到 “像机坐标系”，然后再将其投影到成像平面(图像物理坐标系)，最后再将成像平面上的数据转换到 图像像素坐标系。但是由于透镜制造精度以及组装工艺的偏差会引入畸变，导致原始图像的失真。

![Image of pic](https://github.com/barryyan0121/Camera_Calibration/blob/master/pic/pictures/20150414084703092.jpg)


## 畸变的类型

镜头的畸变分为径向畸变和切向畸变两类。

### 径向畸变

径向畸变是沿着透镜半径方向分布的畸变，产生原因是光线在原理透镜中心的地方比靠近中心的地方更加弯曲，这种畸变在普通廉价的镜头中表现更加明显，径向畸变主要包括桶形畸变和枕形畸变两种。
成像仪光轴中心的畸变为0，沿着镜头半径方向向边缘移动，畸变越来越严重。畸变的数学模型可以用主点(principle point)周围的泰勒级数展开式的前几项进行描述，通常使用前两项，即k1和k2，对于畸变很大的镜头，如鱼眼镜头，可以增加使用第三项k3来进行描述。
![Image of pic](https://github.com/barryyan0121/Camera_Calibration/blob/master/pic/pictures/20150414084722779.jpg)


### 切向畸变

切向畸变是由于透镜本身与相机传感器平面（成像平面）或图像平面不平行而产生的，这种情况多是由于透镜被粘贴到镜头模组上的安装偏差导致。其数学模型为：

```
x = x.*(1+k1*r2 + k2*r2.^2) + 2*p1.*x.*y + p2*(r2 + 2*x.^2);
y = y.*(1+k1*r2 + k2*r2.^2) + 2*p2.*x.*y + p1*(r2 + 2*y.^2);
```

## 畸变校正的方法

我们已知的是畸变后的图像，要得到没有畸变的图像就要通过畸变模型推导其映射关系。 真实图像 imgR 与 畸变图像 imgD 之间的关系为: imgR(U, V) = imgD(Ud, Vd)。通过这个关系，找出所有的 imgR(U, V) 。(U, V)映射到(Ud, Vd)中的 (Ud, Vd) 往往不是整数(U和V是整数，因为它是我们要组成图像的像素坐标位置，以这正常图像的坐标位置去求在畸变图像中的坐标位置，取出对应的像素值，这也是正常图像的像素值)。<br>推导公式为：<br>
![Image of pic](https://github.com/barryyan0121/Camera_Calibration/blob/master/pic/pictures/20150414084827516.jpg)<br>
其逆运算为：<br>
![Image of pic](https://github.com/barryyan0121/Camera_Calibration/blob/master/pic/pictures/20150414084840745.jpg)

## 代码实现

### 1. 准备标定图片
标定图片需要使用标定板在不同位置、不同角度、不同姿态下拍摄，最少需要3张，以10~20张为宜。标定板需要是黑白相间的矩形构成的棋盘图，制作精度要求较高，如下图所示：
![Image of pic](https://github.com/barryyan0121/Camera_Calibration/blob/master/pic/IR_camera_calib_img/00.png)

### 2.对每一张标定图片，提取角点信息
需要使用findChessboardCorners函数提取角点，这里的角点专指的是标定板上的内角点，这些角点与标定板的边缘不接触。

第一个参数Image，传入拍摄的棋盘图Mat图像，必须是8位的灰度或者彩色图像；

第二个参数patternSize，每个棋盘图上内角点的行列数，一般情况下，行列数不要相同，便于后续标定程序识别标定板的方向；

第三个参数corners，用于存储检测到的内角点图像坐标位置，一般用元素是Point2f的向量来表示：vector<Point2f> image_points_buf;

第四个参数flage：用于定义棋盘图上内角点查找的不同处理方式，有默认值。

### 3. 对每一张标定图片，进一步提取亚像素角点信息
为了提高标定精度，需要在初步提取的角点信息上进一步提取亚像素信息，降低相机标定偏差，常用的方法是cornerSubPix，另一个方法是使用find4QuadCornerSubpix函数，这个方法是专门用来获取棋盘图上内角点的精确位置的，或许在相机标定的这个特殊场合下它的检测精度会比cornerSubPix更高？

cornerSubPix函数原型：

第一个参数image，输入的Mat矩阵，最好是8位灰度图像，检测效率更高；

第二个参数corners，初始的角点坐标向量，同时作为亚像素坐标位置的输出，所以需要是浮点型数据，一般用元素是Pointf2f/Point2d的向量来表示：vector<Point2f/Point2d> iamgePointsBuf；

第三个参数winSize，大小为搜索窗口的一半；

第四个参数zeroZone，死区的一半尺寸，死区为不对搜索区的中央位置做求和运算的区域。它是用来避免自相关矩阵出现某些可能的奇异性。当值为（-1，-1）时表示没有死区；

第五个参数criteria，定义求角点的迭代过程的终止条件，可以为迭代次数和角点精度两者的组合

### 4. 在棋盘标定图上绘制找到的内角点(非必须，仅为了显示)
drawChessboardCorners函数用于绘制被成功标定的角点，函数原型

第一个参数image，8位灰度或者彩色图像；
第二个参数patternSize，每张标定棋盘上内角点的行列数；

第三个参数corners，初始的角点坐标向量，同时作为亚像素坐标位置的输出，所以需要是浮点型数据，一般用元素是Pointf2f/Point2d的向量来表示：vector<Point2f/Point2d> iamgePointsBuf；

第四个参数patternWasFound，标志位，用来指示定义的棋盘内角点是否被完整的探测到，true表示别完整的探测到，函数会用直线依次连接所有的内角点，作为一个整体，false表示有未被探测到的内角点，这时候函数会以(红色)圆圈标记处检测到的内角点；

### 5. 相机标定
获取到棋盘标定图的内角点图像坐标之后，就可以使用calibrateCamera函数进行标定，计算相机内参和外参系数，

calibrateCamera函数原型：

第一个参数objectPoints，为世界坐标系中的三维点。在使用时，应该输入一个三维坐标点的向量的向量，即vector<vector<Point3f>> object_points。需要依据棋盘上单个黑白矩阵的大小，计算出（初始化）每一个内角点的世界坐标。

第二个参数imagePoints，为每一个内角点对应的图像坐标点。和objectPoints一样，应该输入vector<vector<Point2f>> image_points_seq形式的变量；

第三个参数imageSize，为图像的像素尺寸大小，在计算相机的内参和畸变矩阵时需要使用到该参数；

第四个参数cameraMatrix为相机的内参矩阵。输入一个Mat cameraMatrix即可，如Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0));

第五个参数distCoeffs为畸变矩阵。输入一个Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0))即可；

第六个参数rvecs为旋转向量；应该输入一个Mat类型的vector，即vector<Mat>rvecs;

第七个参数tvecs为位移向量，和rvecs一样，应该为vector<Mat> tvecs；

### 6. 对标定结果进行评价
对标定结果进行评价的方法是通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到空间三维点在图像上新的投影点的坐标，计算投影坐标和亚像素角点坐标之间的偏差，偏差越小，标定结果越好。

对空间三维坐标点进行反向投影的函数是projectPoints，函数原型是：

### 7. 查看标定效果——利用标定结果对棋盘图进行矫正
利用求得的相机的内参和外参数据，可以对图像进行畸变的矫正，这里有两种方法可以达到矫正的目的，分别说明一下。

方法一：使用initUndistortRectifyMap和remap两个函数配合实现。

initUndistortRectifyMap用来计算畸变映射，remap把求得的映射应用到图像上。

方法二：使用undistort函数实现

undistort函数原型：

第一个参数src，输入参数，代表畸变的原始图像；

第二个参数dst，矫正后的输出图像，跟输入图像具有相同的类型和大小；

第三个参数cameraMatrix为之前求得的相机的内参矩阵；

第四个参数distCoeffs为之前求得的相机畸变矩阵；

第五个参数newCameraMatrix，默认跟cameraMatrix保持一致；

方法一相比方法二执行效率更高一些，推荐使用。

完整代码如下：

```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('/home/barry/Desktop/Camera_Calibration/pic/IR_camera_calib_img/*.png')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,5), corners, ret)

        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

import pickle


# Test undistortion on an image
img = cv2.imread('/home/barry/Desktop/Camera_Calibration/pic/img/calib.png')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('/home/barry/Desktop/Camera_Calibration/pic/save_dedistortion/calibrated_img.png',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
```

## Acknowledgement
This project refers to the following blogs:<br>
https://blog.csdn.net/piaoxuezhong/java/article/details/75268535<br>
https://blog.csdn.net/humanking7/article/details/45037239
