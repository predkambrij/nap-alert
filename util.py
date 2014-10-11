# Import Libraries
import time
import math
from collections import deque
import cProfile

import numpy
import cv2
import cv2.cv as cv
import Image
import ImageOps
import ImageEnhance

#from facedetector import FaceDetector
#from display import Display


#from scipy.cluster import vq
#import matplotlib
#import matplotlib.pyplot as plt

 
from GlobalConstants import GlobalConstants
CAMERA_INDEX = GlobalConstants.CAMERA_INDEX
SCALE_FACTOR = GlobalConstants.SCALE_FACTOR
FACE_CLASSIFIER_PATH = GlobalConstants.FACE_CLASSIFIER_PATH
EYE_CLASSIFIER_PATH = GlobalConstants.EYE_CLASSIFIER_PATH
FACE_MIN_SIZE = GlobalConstants.FACE_MIN_SIZE
EYE_MIN_SIZE = GlobalConstants.EYE_MIN_SIZE

DISPLAY_SCALE = GlobalConstants.DISPLAY_SCALE
FACE_SCALE = GlobalConstants.FACE_SCALE
EYE_SCALE = GlobalConstants.EYE_SCALE



class Util:

    @staticmethod
    def contrast(img, amount='auto'):
        """
        Modify image contrast
        
        Args:
            img (numpy array)            Input image array
            amount (float or string)      Either number (e.g. 1.3) or 'auto'
        """
        
        pilIMG = Image.fromarray(img);
        
        if amount is 'auto':
            pilEnhancedIMG = ImageOps.autocontrast(pilIMG, cutoff = 0);
            return numpy.asarray(pilEnhancedIMG);
        else:
            pilContrast = ImageEnhance.Contrast(pilIMG);
            pilContrasted = pilContrast.enhance(amount);
            return numpy.asarray(pilContrasted);

    @staticmethod
    def threshold(img, thresh):
        """Threshold an image"""
        
        pilIMG1 = Image.fromarray(img);
        pilInverted1 = ImageOps.invert(pilIMG1);
        inverted = numpy.asarray(pilInverted1);
        r, t = cv2.threshold(inverted, thresh, 0, type=cv.CV_THRESH_TOZERO);
        pilIMG2 = Image.fromarray(t);
        pilInverted2 = ImageOps.invert(pilIMG2);
        thresholded = numpy.asarray(pilInverted2);
        return thresholded;

    
    @staticmethod
    def equalizeHSV(img, equalizeH=False, equalizeS=False, equalizeV=True):
        """
        Equalize histogram of color image using BSG2HSV conversion
        By default only equalizes the value channel
        
        Note: OpenCV's HSV implementation doesn't capture all hue info, see:
        http://opencv.willowgarage.com/wiki/documentation/c/imgproc/CvtColor
        http://www.shervinemami.info/colorConversion.html
        """

        imgHSV = cv2.cvtColor(img,cv.CV_BGR2HSV);
        h,s,v = cv2.split(imgHSV);
        
        if equalizeH:
            h = cv2.equalizeHist(h);
        if equalizeS:
            s = cv2.equalizeHist(s);
        if equalizeV:
            v = cv2.equalizeHist(v);
        
        hsv = cv2.merge([h,s,v]);
        bgr = cv2.cvtColor(hsv,cv.CV_HSV2BGR);
        return bgr;
