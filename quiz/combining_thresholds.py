import numpy as np
import matplotlib.pyplot as plt
import cv2


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    assert orient == 'x' or orient == 'y'
    assert sobel_kernel % 2 == 1  # kernel size needs to be odd
    assert thresh[0] < thresh[1]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])

    return grad_binary.astype(int)


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    assert sobel_kernel % 2 == 1  # kernel size needs to be odd
    assert thresh[0] < thresh[1]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mag_binary = (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])

    return mag_binary.astype(int)


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    assert sobel_kernel % 2 == 1  # kernel size needs to be odd
    assert thresh[0] < thresh[1]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_dir = np.arctan2(np.abs(sobely), np.abs(sobelx))
    dir_binary = (grad_dir >= thresh[0]) & (grad_dir <= thresh[1])

    return dir_binary.astype(int)

# Load the image
image = cv2.imread('signs_vehicles_xygrad.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Choose a Sobel kernel size
ksize = 5  # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(
    image, orient='x', sobel_kernel=ksize, thresh=(30, 100))
grady = abs_sobel_thresh(
    image, orient='y', sobel_kernel=ksize, thresh=(30, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(30, 100))
dir_binary = dir_thresh(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

# combined = ((gradx == 1) & (grady == 1)) | (
#     (mag_binary == 1) & (dir_binary == 1))
combined = ((gradx == 1) & (grady == 1)) & (
    (mag_binary == 1) & (dir_binary == 1))

# Plot the result
f, axes = plt.subplots(3, 2, figsize=(6, 6))
f.tight_layout()
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original image')
axes[0, 1].imshow(gradx, cmap='gray')
axes[0, 1].set_title('Grad x')
axes[1, 0].imshow(grady, cmap='gray')
axes[1, 0].set_title('Grad y')
axes[1, 1].imshow(mag_binary, cmap='gray')
axes[1, 1].set_title('Mag')
axes[2, 0].imshow(dir_binary, cmap='gray')
axes[2, 0].set_title('Grad dir')
axes[2, 1].imshow(combined, cmap='gray')
axes[2, 1].set_title('Combined')
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
