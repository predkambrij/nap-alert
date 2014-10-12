#!/usr/bin/env python2.7

"""
nap-alert
Authors: Brandon Jackson, Alojzij Blatnik
"""

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

from facedetector import FaceDetector
from facemodel import FaceModel
from display import Display
from util import Util


#from scipy.cluster import vq
#import matplotlib
#import matplotlib.pyplot as plt

 
# Constants
from globalconstants import GlobalConstants
CAMERA_INDEX = GlobalConstants.CAMERA_INDEX
SCALE_FACTOR = GlobalConstants.SCALE_FACTOR
FACE_CLASSIFIER_PATH = GlobalConstants.FACE_CLASSIFIER_PATH
EYE_CLASSIFIER_PATH = GlobalConstants.EYE_CLASSIFIER_PATH
FACE_MIN_SIZE = GlobalConstants.FACE_MIN_SIZE
EYE_MIN_SIZE = GlobalConstants.EYE_MIN_SIZE

DISPLAY_SCALE = GlobalConstants.DISPLAY_SCALE
FACE_SCALE = GlobalConstants.FACE_SCALE
EYE_SCALE = GlobalConstants.EYE_SCALE



			

class Capture:

	camera = cv2.VideoCapture(CAMERA_INDEX);
	height = 0;
	width = 0;
	
	def __init__(self, scaleFactor=1):
		# set resolution for the webcam
		self.camera.set(3,1280)
		self.camera.set(4,720)
		
		# get webcam dimensions
		self.height = self.camera.get(cv.CV_CAP_PROP_FRAME_HEIGHT);
		self.width = self.camera.get(cv.CV_CAP_PROP_FRAME_WIDTH);
		
		# Reduce Video Size to make Processing Faster
		if scaleFactor is not 1:
			scaledHeight = self.height / scaleFactor;
			scaledWidth = self.width / scaleFactor;
			self.camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT,scaledHeight);
			self.camera.set(cv.CV_CAP_PROP_FRAME_WIDTH,scaledWidth);
	
	def read(self):
		retVal, colorFrame = self.camera.read();
		displayFrame = cv2.resize(colorFrame,None,fx=DISPLAY_SCALE,fy=DISPLAY_SCALE);
		
		grayFrame = cv2.equalizeHist(cv2.cvtColor(colorFrame,cv.CV_BGR2GRAY));
		
		faceFrame = cv2.resize(grayFrame,None,fx=FACE_SCALE,fy=FACE_SCALE);
		
		eyesFrame = cv2.resize(cv2.equalizeHist(cv2.cvtColor(colorFrame,cv.CV_BGR2GRAY)),None,fx=EYE_SCALE,fy=EYE_SCALE);
		
		frames = {
			'color': colorFrame,
			'display': displayFrame,
			#'gray': grayFrame,
			'face': faceFrame,
			'eyes': eyesFrame
		};
		
		return frames;

def main():
	# Instantiate Classes
	detector = FaceDetector(FACE_CLASSIFIER_PATH, EYE_CLASSIFIER_PATH);
	model = FaceModel();
	display = Display();
	capture = Capture();
	
	oldTime = time.time();
	i = 0;
	frames_num=0
	delta_sum = 0
	while True:
		# escape key for exit, in linux display is not working without that
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			return
		
		# Calculate time difference (dt), update oldTime variable
		newTime = time.time();
		dt =  newTime - oldTime;
		oldTime = newTime;
		
		frames_num += 1
		delta_sum += dt
		if frames_num % 100 == 0:
			print "delta:",delta_sum/float(frames_num),"frames pr sec:",1/float(delta_sum/float(frames_num))
			frames_num=0
			delta_sum = 0
		
		# Grab Frames
		frames = capture.read();	
		
		# Detect face 20% of the time, eyes 100% of the time
		if i % 5 is 0:
			rects = detector.detect(frames);
		else:
			rects = detector.detect(frames,model.getPreviousFaceRects());
		i += 1;
		
		# Add detected rectangles to model
		model.add(rects);
		
		display.renderScene(frames['display'],model,rects);
		display.renderEyes(frames['color'],model);

cProfile.run('main()','profile.o','cumtime');
