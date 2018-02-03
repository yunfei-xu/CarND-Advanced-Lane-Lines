import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import argparse
import pickle
import logging
import os
from moviepy.editor import VideoFileClip
from sklearn import linear_model

FLAGS = None


def save_image(filename):
    plt.savefig('output_images/' + filename, bbox_inches='tight')


def color_thresh(img, thresh=(0, 255)):
    """
    Color thresholding.
    """
    assert thresh[0] < thresh[1]

    binary = (img >= thresh[0]) & (img <= thresh[1])
    return binary.astype(np.uint8)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Sobel thresholding.
    """
    assert orient == 'x' or orient == 'y'
    assert sobel_kernel % 2 == 1  # kernel size needs to be odd
    assert thresh[0] < thresh[1]

    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])

    return grad_binary.astype(np.uint8)


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Sobel magnitude thresholding.
    """
    assert sobel_kernel % 2 == 1  # kernel size needs to be odd
    assert thresh[0] < thresh[1]

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mag_binary = (scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])

    return mag_binary.astype(np.uint8)


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Sobel direction thresholding.
    """
    assert sobel_kernel % 2 == 1  # kernel size needs to be odd
    assert thresh[0] < thresh[1]

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad_dir = np.arctan2(np.abs(sobely), np.abs(sobelx))
    dir_binary = (grad_dir >= thresh[0]) & (grad_dir <= thresh[1])

    return dir_binary.astype(np.uint8)


class Line():
    """
    A class to receive the characteristics of each line detection.
    """

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        # x values for ploty (for plotting fitted line)
        self.fitx = None
        # number of frames exceeding which will cause a reset
        self.n = 5

    def fit(self):
        """
        Fit the points with second order polynomials.
        """
        if self.ally.shape[0] <= 20:
            logging.debug('Using previous fit coefficients')
            self.current_fit = self.recent_xfitted[-1]
        else:
            self.current_fit = np.polyfit(self.ally, self.allx, 2)

        self.recent_xfitted.append(self.current_fit)
        if len(self.recent_xfitted) > self.n:
            self.recent_xfitted.pop(0)

        # self.best_fit = np.mean(np.vstack(self.recent_xfitted), axis=0)
        self.best_fit = self.current_fit

        # Constraint the slope at the bottom is zero
        # coeff = np.polyfit(self.ally ** 2 - 1440 * self.ally, self.allx, 1)
        # self.current_fit = np.array([coeff[0], -1440 * coeff[0], coeff[1]])

        # self.recent_xfitted.append(self.current_fit)
        # if len(self.recent_xfitted) > self.n:
        #     self.recent_xfitted.pop(0)
        # self.best_fit = np.mean(np.vstack(self.recent_xfitted), axis=0)

    def predict(self, y):
        """
        Predict the x value given y value.
        """
        # return self.current_fit[0] * (y ** 2) + self.current_fit[1] * y + self.current_fit[2]
        return self.best_fit[0] * (y ** 2) + self.best_fit[1] * y + self.best_fit[2]

    def get_radius(self, ploty):
        """
        Compute radius at current position.
        """
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * ym_per_pix, self.fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])

    def get_dist_to_veh_center(self, center):
        """
        Compute the distance in meters of vehicle center from the line.
        Negative if on the left of the vehicle center.
        Positive if on the right of the vehicle center.
        """
        dist_in_pix = self.fitx[-1] - center
        xm_per_pix = 3.7 / 700
        self.line_base_pos = dist_in_pix * xm_per_pix


class LaneDetector:
    """
    The class for running lane detection algorithm.
    """

    def __init__(self):
        # Define left/right lanes
        self.left = Line()
        self.right = Line()

        # Force the line search to start from scratch, using histogram
        self.start_from_scratch = True

        # Load calibration parameters
        with open('calib.p', 'rb') as f:
            calib = pickle.load(f)
        self.mtx = calib['mtx']
        self.dist = calib['dist']

        # Load the perspective matrix
        with open('perspect.p', 'rb') as f:
            perspect = pickle.load(f)
        self.M = perspect['M']
        self.Minv = perspect['Minv']

        # Image size
        self.image_width = 1280
        self.image_height = 720

        # For plotting
        self.ploty = np.linspace(0, self.image_height - 1, self.image_height)

        # Frame id for statistics
        self.frame_id = 0

    def pipeline(self, image):
        """
        Lane detection pipeline.
        """
        if FLAGS.savefig:
            plt.figure(figsize=(12, 12))
            plt.imshow(image)
            plt.title('Original image')
            save_image('original_image.jpg')

        timer = {}
        start_time = time.time()

        # Undistortion
        undistorted = self.correct_distortion(image)
        end_time = time.time()
        timer['undistortion'] = end_time - start_time
        start_time = end_time

        # Thresholding
        binary = self.thresholding(undistorted)
        end_time = time.time()
        timer['thresholding'] = end_time - start_time
        start_time = end_time

        # Perspective tranform
        warped = self.perspective_transform(binary)
        end_time = time.time()
        timer['warp'] = end_time - start_time
        start_time = end_time

        # Find lines in the warped image
        self.find_lines(warped)
        end_time = time.time()
        timer['find_lines'] = end_time - start_time

        # Show computation time information
        logging.debug('Frame: {:04d} - undistortion: {:3.1f}ms - thresholding: {:3.1f}ms - warp: {:3.1f}ms - find_lines: {:3.1f}ms'.format(
            self.frame_id, timer['undistortion'] * 1000, timer['thresholding'] * 1000, timer['warp'] * 1000, timer['find_lines'] * 1000))

        # Compute radius at current position
        self.left.get_radius(self.ploty)
        self.right.get_radius(self.ploty)
        radius = (self.left.radius_of_curvature + self.right.radius_of_curvature) / 2
        logging.debug('Left radius: {:6.1f} - right radius: {:6.1f}'.format(self.left.radius_of_curvature,
                                                                            self.right.radius_of_curvature))

        # Vehicle center offset
        self.left.get_dist_to_veh_center(self.image_width / 2)
        self.right.get_dist_to_veh_center(self.image_width / 2)
        offset = (self.left.line_base_pos + self.right.line_base_pos) / 2

        # Overlay the detected lines on the undistorted image
        result = self.overlay_image(warped, undistorted)

        # Overlay radius on top of the image
        text = 'Lane radius: {:6.1f}m'.format(radius)
        cv2.putText(result, text, (10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Overlay offset on top of the image
        text = 'Vehicle is {:3.2f}m {} of lane center'.format(abs(offset), 'left' if (offset > 0) else 'right')
        cv2.putText(result, text, (10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Overlay frame id
        cv2.putText(result, 'frame {:04d}'.format(self.frame_id), (10, 700), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        self.frame_id += 1

        return result

    def correct_distortion(self, image):
        """
        Undistort the image.
        """
        undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        if FLAGS.savefig:
            plt.figure(figsize=(12, 12))
            plt.imshow(undistorted)
            plt.title('Undistorted image')
            save_image('undistorted_image.jpg')

        return undistorted

    def roi(self, image):
        mask = np.zeros_like(image)
        vertices = np.array([[0, 720],
                             [400, 450],
                             [880, 450],
                             [1280, 720]], dtype=np.int32)
        if len(image.shape) > 2:
            # This is a color image
            cv2.fillPoly(mask, [vertices], (255, ) * image.shape[2])
        else:
            # This is a gray scale image
            cv2.fillPoly(mask, [vertices], 255)
        return cv2.bitwise_and(image, mask)

    def thresholding(self, image):
        """
        Thresholding the image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if FLAGS.savefig:
            plt.figure(figsize=(12, 12))
            plt.imshow(gray, cmap='gray')
            save_image('gray.jpg')

        if FLAGS.savefig:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.tight_layout()
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original image')
            axes[0, 1].imshow(image[:, :, 0], cmap='gray')
            axes[0, 1].set_title('R channel')
            axes[1, 0].imshow(image[:, :, 1], cmap='gray')
            axes[1, 0].set_title('G channel')
            axes[1, 1].imshow(image[:, :, 2], cmap='gray')
            axes[1, 1].set_title('B channel')
            save_image('rgb.jpg')

        # Convert to HSV color space and separate the S channel
        if FLAGS.savefig:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.tight_layout()
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original image')
            axes[0, 1].imshow(hsv[:, :, 0], cmap='gray')
            axes[0, 1].set_title('H channel')
            axes[1, 0].imshow(hsv[:, :, 1], cmap='gray')
            axes[1, 0].set_title('S channel')
            axes[1, 1].imshow(hsv[:, :, 2], cmap='gray')
            axes[1, 1].set_title('V channel')
            save_image('hsv.jpg')

        if FLAGS.savefig:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.tight_layout()
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original image')
            axes[0, 1].imshow(lab[:, :, 0], cmap='gray')
            axes[0, 1].set_title('L channel')
            axes[1, 0].imshow(lab[:, :, 1], cmap='gray')
            axes[1, 0].set_title('A channel')
            axes[1, 1].imshow(lab[:, :, 2], cmap='gray')
            axes[1, 1].set_title('B channel')
            save_image('lab.jpg')

        # Get R channel image
        r_channel = image[:, :, 0]
        r_binary = color_thresh(r_channel, thresh=(150, 255))
        b_channel = image[:, :, 2]

        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        if FLAGS.savefig:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.tight_layout()
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original image')
            axes[0, 1].imshow(hls[:, :, 0], cmap='gray')
            axes[0, 1].set_title('H channel')
            axes[1, 0].imshow(hls[:, :, 1], cmap='gray')
            axes[1, 0].set_title('L channel')
            axes[1, 1].imshow(hls[:, :, 2], cmap='gray')
            axes[1, 1].set_title('S channel')
            save_image('hls.jpg')

        # Get the S channel image
        s_channel = hls[:, :, 2]
        s_binary = color_thresh(s_channel, thresh=(100, 255))

        # Get the H channel image
        # Shadow tend to have high S value and also H value
        # By thresholding H value can avoid detect shadow as lane markers
        h_channel = hls[:, :, 0]
        h_binary = color_thresh(h_channel, thresh=(0, 100))

        gradx_binary = abs_sobel_thresh(
            r_channel, orient='x', sobel_kernel=3, thresh=(30, 100))
        grady_binary = abs_sobel_thresh(r_channel, orient='y', sobel_kernel=3, thresh=(30, 100))

        mag_binary = mag_thresh(r_channel, sobel_kernel=3, thresh=(30, 100))
        dir_binary = dir_thresh(r_channel, sobel_kernel=3, thresh=(0.7, 1.3))
        # grad_binary = np.uint8(gradx_binary | (mag_binary & dir_binary))
        grad_binary = np.uint8(gradx_binary & grady_binary)

        combined_binary = (h_binary & s_binary) | (grad_binary & r_binary)

        if FLAGS.savefig:
            plt.figure(figsize=(12, 12))
            color_binary = np.dstack(((h_binary & s_binary),
                                      (grad_binary & r_binary),
                                      np.zeros_like(s_binary))) * 255
            plt.imshow(color_binary)
            plt.title('Stacked thresholds')
            save_image('stack_thresholds.jpg')

        if FLAGS.savefig:
            fig, axes = plt.subplots(3, 2, figsize=(12, 12))
            fig.tight_layout()
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original image (undistorted)')
            axes[0, 1].imshow(gradx_binary, cmap='gray')
            axes[0, 1].set_title('Grad x')
            axes[1, 0].imshow(grady_binary, cmap='gray')
            axes[1, 0].set_title('Grad y')
            axes[1, 1].imshow(mag_binary, cmap='gray')
            axes[1, 1].set_title('Mag')
            axes[2, 0].imshow(dir_binary, cmap='gray')
            axes[2, 0].set_title('Grad dir')
            axes[2, 1].imshow(grad_binary, cmap='gray')
            axes[2, 1].set_title('Combined grad')
            save_image('gradients.jpg')

        # return self.roi(combined_binary)
        return combined_binary

    def perspective_transform(self, image):
        """
        Warp the binary image to bird-eye view.
        """
        warped = cv2.warpPerspective(image, self.M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        if FLAGS.savefig:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
            fig.tight_layout()
            ax1.imshow(image, cmap='gray')
            ax1.set_title('Source image')
            ax2.imshow(warped, cmap='gray')
            ax2.set_title('Warped image')
            save_image('warped_image.jpg')

        return warped

    def inverse_perspective_transform(self, image):
        """
        Warp the bird-eye view back to original view.
        """
        return cv2.warpPerspective(image, self.Minv, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    def find_lines(self, warped):
        """
        Find lines on the warped image.
        """
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Set the width of the windows +/- margin
        margin = 70

        if FLAGS.savefig:
            out_img = np.dstack((warped, warped, warped)) * 255

        if self.start_from_scratch:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
            if FLAGS.savefig:
                plt.figure(figsize=(12, 3))
                plt.plot(histogram)
                plt.title('Histogram')
                plt.xlabel('Pixel')
                plt.ylabel('Counts')
                save_image('histogram.jpg')

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            mid_point = int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:mid_point])
            rightx_base = np.argmax(histogram[mid_point:]) + mid_point
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = warped.shape[0] - (window + 1) * window_height
                win_y_high = warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                if FLAGS.savefig:
                    # Draw the windows on the visualization image
                    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                                  (0, 255, 0), 2)
                    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                                  (0, 255, 0), 2)

                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        else:
            # We can do a highly targeted search
            left_fitx = self.left.predict(nonzeroy)
            left_lane_inds = ((nonzerox > (left_fitx - margin)) &
                              (nonzerox < (left_fitx + margin))).nonzero()[0]
            right_fitx = self.right.predict(nonzeroy)
            right_lane_inds = ((nonzerox > (right_fitx - margin)) &
                               (nonzerox < (right_fitx + margin))).nonzero()[0]

        # Extract left and right line pixel positions
        self.left.allx = nonzerox[left_lane_inds]
        self.left.ally = nonzeroy[left_lane_inds]
        self.right.allx = nonzerox[right_lane_inds]
        self.right.ally = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each line
        self.left.fit()
        self.right.fit()

        # Predict x values given y values
        self.left.fitx = self.left.predict(self.ploty)
        self.right.fitx = self.right.predict(self.ploty)

        if FLAGS.savefig:
            out_img[self.left.ally, self.left.allx] = [255, 0, 0]
            out_img[self.right.ally, self.right.allx] = [0, 0, 255]
            plt.figure(figsize=(12, 12))
            plt.imshow(out_img)
            plt.plot(self.left.fitx, self.ploty, color='yellow')
            plt.plot(self.right.fitx, self.ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            save_image('fit_lines.jpg')

    def overlay_image(self, warped, undistorted):
        """
        Overlay the lines found onto the undistorted image.
        """
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # color_warp[self.left.ally, self.left.allx] = [255, 0, 0]
        # color_warp[self.right.ally, self.right.allx] = [0, 0, 255]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left.fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right.fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.inverse_perspective_transform(color_warp)

        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        # result[newwarp[:, :, 0] == 255] = [255, 0, 0]
        # result[newwarp[:, :, 2] == 255] = [0, 0, 255]

        if FLAGS.savefig:
            plt.figure(figsize=(12, 12))
            plt.imshow(result)
            save_image('final_result.jpg')

        return result


def main():

    if FLAGS.filename.split('.')[-1] == 'jpg' or FLAGS.filename.split('.')[-1] == 'png':
        detector = LaneDetector()
        #
        # mpimg command 'image = mpimg.imread(filename)' reads png file as float32, NOT uint8
        # Use cv2.imread instead, but we need to convert BGR to RGB format
        #
        image = cv2.imread(FLAGS.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector.pipeline(image)
    elif FLAGS.filename.split('.')[-1] == 'mp4':
        detector = LaneDetector()
        clip = VideoFileClip(FLAGS.filename)
        result_clip = clip.fl_image(detector.pipeline)
        result_clip.write_videofile('result_' + FLAGS.filename.split('/')[-1])
    else:
        print('The file format is not supported')
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filename', type=str, help='the image (*.jpg/*.png)/video (*.mp4) file to use')
    parser.add_argument('--savefig', type=bool, default=False, help='if save intermediate figures')
    parser.add_argument('--logfile', type=str, default='log.txt', help='log file')
    FLAGS = parser.parse_args()

    if os.path.isfile(FLAGS.logfile):
        os.remove(FLAGS.logfile)
    logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)

    main()
