import glob
import pickle
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

CALIBRATED = True

if not CALIBRATED:
    # Find all calibration image files
    image_files = glob.glob('camera_cal/calibration*.jpg')

    # Arrays to store object points and image points of all calibration images
    object_points = []  # 3D points to in real world space
    image_points = []  # 2D points in image plane

    # Prepare object points
    nx = 9
    ny = 6
    obj_pts = np.zeros((nx * ny, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Figure to show calibration images with detected corners (if successful)
    fig, axes = plt.subplots(4, 5, figsize=(12, 6))
    fig.tight_layout()
    # Iterate through all calibration images
    for i, image_file in enumerate(image_files):
        # Read in the calibration image
        image = mpimg.imread(image_file)

        # Convert the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If corners are found, append object points and image points
        if ret:
            image_points.append(corners)
            object_points.append(obj_pts)

            # Draw the detected corners
            image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            axes[i // 5, i % 5].imshow(image)

        # Set the axis off
        axes[i // 5, i % 5].axis('off')

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None)

    # Print the camera matrix and the distortion coefficients
    print('Camera matrix:')
    print(mtx)
    print('Distortion coefficients:')
    print(dist)

    # Save the camera matrix and the distortion coefficients
    calib = {'mtx': mtx, 'dist': dist}
    with open('calib.p', 'wb') as f:
        pickle.dump(calib, f)

else:
    # Load the camera matrix and the distortion coefficients
    with open('calib.p', 'rb') as f:
        calib = pickle.load(f)
    mtx = calib['mtx']
    dist = calib['dist']

# Undistort a test image
image = mpimg.imread('camera_cal/calibration1.jpg')
undistorted = cv2.undistort(image, mtx, dist, None, mtx)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.tight_layout()
ax1.imshow(image)
ax1.set_title('Original image')
ax2.imshow(undistorted)
ax2.set_title('Undistorted image')

# Show images
plt.show()
