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

from util import Util


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



class FaceDetector:

    """
    FaceDetector is a wrapper for the cascade classifiers.
    Must be initialized using faceClassifierPath and eyeClassifierPath, and 
    should only be initialized once per program instance. The only "public"
    method is detect().
    """

    def __init__(self, faceClassifierPath, eyeClassifierPath):
        """
        Initialize & Load Haar Cascade Classifiers.
        
        Args:
            faceClassifierPath (string): path to face Haar classifier
            eyeClassifierPath (string): path to eye Haar classifier
        """
        self.faceClassifier = cv2.CascadeClassifier(faceClassifierPath);
        self.eyeClassifier = cv2.CascadeClassifier(eyeClassifierPath);
    
    def detect(self,frames, faceRect=False):
        """
        Detect face and eyes. 
        Runs Haar cascade classifiers. Sometimes it is desirable to speed up 
        processing by using a previously-found face rectangle. To do this, pass 
        the old faceRect as the second argument.
        
        Args:
            frames (dict of numpy array): dictionary containing images with different scales
            faceRect (numpy array): array of face rectangle. Face detected if 
                                     omitted.
        Returns:
            a dictionary with three elements each representing a rectangle
        """

        # Data structure to hold frame info
        rects = {
            'face': numpy.array([],dtype=numpy.int32),
            'eyeLeft': numpy.array([],dtype=numpy.int32),
            'eyeRight': numpy.array([],dtype=numpy.int32)
        };
        
        # Detect face if old faceRect not provided
        if faceRect is False or len(faceRect) is 0:
            faceIMG = frames['face'];
            #cv.ShowImage("w1", cv.fromarray(faceIMG))
            
            faceRects = self.classifyFace(faceIMG);
            
            # Ensure a single face found
            if len(faceRects) is 1:
                faceRect = faceRects[0];
            else:
                # TODO throw error message
                ##print "No Faces / Multiple Faces Found!";
                pass
                return rects;
            
        rects['face'] = faceRect;

        # Extract face coordinates, calculate center and diameter
        x1,y1,x2,y2 = rects['face'];
        faceCenter = (((x1+x2)/2.0), ((y1+y2)/2.0));
        faceDiameter = y2-y1;
        
        # Extract eyes region of interest (ROI), cropping mouth and hair
        eyeBBox = numpy.array([x1,
                              (y1 + (faceDiameter*0.24)),
                              x2,
                              (y2 - (faceDiameter*0.40))],dtype=numpy.int32);
        
                            
#        eyesY1 = (y1 + (faceDiameter * 0.16));
#        eyesY2 = (y2 - (faceDiameter * 0.32));
#        eyesX1 = x1 * EYE_SCALE;
#        eyesX2 = x2 * EYE_SCALE;
#        eyesROI = img[eyesY1:eyesY2, x1:x2];

        # Search for eyes in ROI
        eyeRects = self.classifyEyes(frames['eyes'],eyeBBox);
#        print eyeRects;
        
        # Ensure (at most) two eyes found
        if len(eyeRects) > 2:
            # TODO throw error message (and perhaps return?)
            print "Multiple Eyes Found!";
            # TODO get rid of extras by either:
            #    a) using two largest rects or
            #    b) finding two closest matches to average eyes
            

        # Loop over each eye
        for e in eyeRects:
            # Adjust coordinates to be in faceRect's coordinate space
#            e += numpy.array([eyesX1, eyesY1, eyesX1, eyesY1],dtype=numpy.int32);
                        
            # Split left and right eyes. Compare eye and face midpoints.
            eyeMidpointX = (e[0]+e[2])/2.0;
            if eyeMidpointX < faceCenter[0]:
                rects['eyeLeft'] = e; # TODO prevent overwriting
            else:
                rects['eyeRight'] = e;
        # TODO error checking
        # TODO calculate signal quality
        print 'final rects=',rects
        
        return rects;

    def classify(self, img, cascade, minSizeX=40):
        """Run Cascade Classifier on Image"""
        minSizeX = int(round(minSizeX));
#        print 'minSizeX:',minSizeX
        # Run Cascade Classifier
        rects = cascade.detectMultiScale(
                img, minSize=(minSizeX,minSizeX), 
                flags=cv.CV_HAAR_SCALE_IMAGE);
        
        # No Results
        if len(rects) == 0:
            return numpy.array([],dtype=numpy.int32);
        
        rects[:,2:] += rects[:,:2]; # ? ? ? 
        rects = numpy.array(rects,dtype=numpy.int32);
        return rects;
    
    def classifyFace(self,img):
        """Run Face Cascade Classifier on Image"""
        rects = self.classify(img,self.faceClassifier,img.shape[1]*FACE_MIN_SIZE);
        return rects/FACE_SCALE;
    
    def classifyEyes(self,img,bBox):
        """Run Eyes Cascade Classifier on Image"""
        EYE_MIN_SIZE = 0.15;
        bBoxScaled = bBox*EYE_SCALE;
        eyesROI = img[bBoxScaled[1]:bBoxScaled[3], bBoxScaled[0]:bBoxScaled[2]];
        
        eyesROI = cv2.equalizeHist(eyesROI);
        
#        print 'eyesROI dimensions: ',eyesROI.shape;
        minEyeSize = eyesROI.shape[1]*EYE_MIN_SIZE;
#        print 'minEyeSize:',minEyeSize;
        cv2.imshow("eyesROI",eyesROI);
        rectsScaled = self.classify(eyesROI, self.eyeClassifier, 
                                    minEyeSize);
        
#        print rectsScaled;
        # Scale back to full size
        rects = rectsScaled / EYE_SCALE;
        
        # Loop over each eye
        for eye in rects:
            # Adjust coordinates to be in faceRect's coordinate space
            eye += numpy.array([bBox[0],bBox[1],bBox[0],bBox[1]]);

        return rects;
