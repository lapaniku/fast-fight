from skimage.measure import compare_ssim
from skimage import measure
import argparse
import imutils
import cv2
import numpy as np

from skimage.feature import blob_dog
from skimage.color import rgb2gray

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

from imutils import contours

from scipy.spatial import distance as dist
import functools
import operator
from collections import OrderedDict


class Blob:
    
    def __init__(self, size):
        self._size = size

    @property
    def size(self):
        return self._size

    @property
    def centroid(self):
        return self._centroid
    
    @centroid.setter
    def centroid(self, c):
        self._centroid = c

    @property
    def compactness(self):
        return self._compactness
    
    @compactness.setter
    def compactness(self, c):
        self._compactness = c
    
    @centroid.setter
    def centroid(self, c):
        self._centroid = c
    
    
    def distance(self, other_blob):
        return dist.euclidean(self._centroid, other_blob.centroid)
    
    def features1(self):
        return [self._size, self._centroid[0], self._centroid[1]] 
    
    def features2(self):
        return [self._compactness, self._centroid[0], self._centroid[1], ] 

    def features3(self):
        return [self._size, self._compactness, self._centroid[0], self._centroid[1], ] 

def extract_features(file_path, W, H, K, h = 0.8, min_blob_size = 20, class_label=None):
    vc = cv2.VideoCapture(file_path)
    ret, first_frame = vc.read()
    # TODO: check ret
    first_frame = cv2.resize(first_frame, (W, H), interpolation = cv2.INTER_AREA)
    
    # Convert to gray scale 
    try:
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        pass

    video_blobs = []
    # process file
    caption = 'fight' if class_label == 1 else 'no-fight'
    color = (10, 69, 250) if class_label == 1 else (0, 255, 0)
    while(vc.isOpened()):
        try:
            # Read a frame from video
            ret, frame = vc.read()
            frame = cv2.resize(frame, (W, H), interpolation = cv2.INTER_AREA)
            if class_label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, caption, (int(W/2)-60,int(H/2)), font, 1, color, 2, cv2.LINE_AA)
                cv2.imshow('action', frame)
                cv2.waitKey(1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            (_, diff) = compare_ssim(prev_gray, gray, full=True)
            #diff = (diff * 255).astype("uint8")
            diff = ((diff < h) * 255).astype("uint8")
            #print("SSIM: {}".format(score))

            # print('min: {}'.format(np.min(diff)))
            # print('max: {}'.format(np.max(diff)))


            # Simple diff
            # diff = ((abs(prev_gray - gray) > h*255)*255).astype("uint8")

            # https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/

            #thresh = cv2.threshold(diff, int(h*255), 255, cv2.THRESH_BINARY)[1]

            # perform a series of erosions and dilations to remove
            # any small blobs of noise from the thresholded image
            #thresh = cv2.erode(thresh, None, iterations=2)
            #thresh = cv2.dilate(thresh, None, iterations=4)

            image = diff

            # perform a connected component analysis on the thresholded
            # image, then initialize a mask to store only the "large"
            # components
            labels = measure.label(image, neighbors=8, background=0)
            mask = np.zeros(image.shape, dtype="uint8")
            
            # loop over the unique components
            for label in np.unique(labels):
                # if this is the background label, ignore it
                if label == 0:
                    continue
            
                # otherwise, construct the label mask and count the
                # number of pixels 
                labelMask = np.zeros(image.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)
            
                # if the number of pixels in the component is sufficiently
                # large, then add it to our mask of "large blobs"
                if numPixels > min_blob_size:
                    blob = Blob(numPixels)
                    # mask = cv2.add(mask, labelMask)

                    # find the contours in the mask, then sort them from left to
                    # right
                    cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    if len(cnts) > 0:
                        cnts = contours.sort_contours(cnts)[0]
                
                    # print("Contours: {}".format(len(cnts)))
                    # loop over the contours
                    for (_, c) in enumerate(cnts):
                        # draw the bright spot on the image
                        ((cX, cY), _) = cv2.minEnclosingCircle(c)
                    blob.centroid = (cX, cY)

                    # https://answers.opencv.org/question/51602/has-opencv-built-in-functions-to-calculate-circularity-compactness-etc-for-contoursblobs/
                    ca = cv2.contourArea(c)
                    al = cv2.arcLength(c, True)
                    blob.compactness = ca/al

                    video_blobs.append(blob)
            
            # Update previous frame
            prev_gray = gray
            # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        except Exception as e:
            print(e)
            break

    vc.release()

    video_blobs.sort(key=lambda x: x.size, reverse=True)
    top_blobs = video_blobs[:K]
    if len(top_blobs) > 0:
        blob_features = functools.reduce(operator.concat, [b.features3() for b in top_blobs])
        distances = list(OrderedDict.fromkeys([b1.distance(b2) for b1 in top_blobs for b2 in top_blobs if b1 != b2]))
        # print("Distances: ", result)
        blob_features.extend(distances)
        return blob_features
    else:
        return []