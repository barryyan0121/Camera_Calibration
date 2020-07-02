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
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

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
        # Refining corners position with sub-pixels based algorithm
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,5), corners, ret)

        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Test undistortion on an image
img = cv2.imread('/home/barry/Desktop/Camera_Calibration/pic/img/calib.png')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

print (("ret:"),ret)
print (("internal matrix:\n"),mtx)
# in the form of (k_1,k_2,p_1,p_2,k_3)
print (("distortion cofficients:\n"),dist)
print (("rotation vectors:\n"),rvecs)
print (("translation vectors:\n"),tvecs)

# calculate the error of reproject
total_error = 0
for i in range(len(objpoints)):
    img_points_repro, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], img_points_repro, cv2.NORM_L2)/len(img_points_repro)
    total_error += error
print(("Average Error of Reproject: "), total_error/len(objpoints))

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