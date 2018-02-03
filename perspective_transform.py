import pickle
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load a test image
image = mpimg.imread('test_images/straight_lines1.jpg')

# Load the camera matrix and distortion coefficients generated with
# 'calibration.py'
with open('calib.p', 'rb') as f:
    calib = pickle.load(f)
mtx = calib['mtx']
dist = calib['dist']

#
# Undistort the image
#
undistorted = cv2.undistort(image, mtx, dist, None, mtx)
undistorted_with_roi = np.copy(undistorted)

# Draw a polygon defining the region of interest
src = np.array([[205, 720], [595, 450], [685, 450], [1105, 720]])
# src = np.array([[205, 720], [555, 480], [725, 480], [1105, 720]])
cv2.polylines(undistorted_with_roi, [src], isClosed=True,
              color=(255, 0, 0), thickness=5)


# Get Perspective matrix
dst = np.array([[340, 720], [340, 0], [940, 0], [940, 720]])
M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
Minv = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32))
perspect = {'M': M, 'Minv': Minv}
with open('perspect.p', 'wb') as f:
    pickle.dump(perspect, f)

# Warp the image to bird-eye view
warped = cv2.warpPerspective(undistorted, M, undistorted.shape[1::-1], flags=cv2.INTER_LINEAR)
cv2.polylines(warped, [dst], isClosed=True,
              color=(255, 0, 0), thickness=5)

# Save the image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
fig.tight_layout()
ax1.imshow(undistorted_with_roi)
ax1.set_title('Source image')
ax2.imshow(warped)
ax2.set_title('Warped image')
plt.savefig('output_images/perspective_transform.jpg', bbox_inches='tight')
