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



class FaceModel:

    """
    FaceModel integrates data from the new frame into a model that keeps track of where the eyes are. To do this it uses:
        - A moving average of the most recent frames
        - Facial geometry to fill in missing data
    The resulting model generates a set of two specific regions of interest (ROI's) where blinking is expected to take place.
    """
    
    # TODO flush eye history whenever faceRect midpoint changes
    # TODO flush eye history whenever eye rectangle outside of faceRect bbox
    # TODO make sure that eye rectangles don't overlap

    QUEUE_MAXLEN = 50;
    
    QUALITY_QUEUE_MAXLEN = 30;
    qualityHistory = {
        'face':deque(maxlen=QUALITY_QUEUE_MAXLEN),
        'eyeLeft':deque(maxlen=QUALITY_QUEUE_MAXLEN),
        'eyeRight':deque(maxlen=QUALITY_QUEUE_MAXLEN)
    };
    
    # Queues storing most recent position rectangles, used to calculate
    # moving averages
    rectHistory = {
        'face': deque(maxlen=QUEUE_MAXLEN),
        'eyeLeft': deque(maxlen=QUEUE_MAXLEN),
        'eyeRight': deque(maxlen=QUEUE_MAXLEN)
    };
    
    # Moving average of position rectangles
    rectAverage = {
        'face': numpy.array([]),
        'eyeLeft': numpy.array([]),
        'eyeRight': numpy.array([])
    };
    
    def add(self,rects):
        """Add new set of rectangles to model
        Examples:
        final rects= {'eyeRight': array([714, 283, 790, 359], dtype=int32), 'eyeLeft': array([517, 274, 599, 356], dtype=int32), 'face': array([414, 158, 847, 591], dtype=int32)}
        final rects= {'eyeRight': array([689, 279, 769, 359], dtype=int32), 'eyeLeft': array([520, 276, 599, 355], dtype=int32), 'face': array([414, 158, 847, 591], dtype=int32)}
        final rects= {'eyeRight': array([694, 282, 769, 357], dtype=int32), 'eyeLeft': array([521, 278, 595, 352], dtype=int32), 'face': array([421, 169, 841, 589], dtype=int32)}
        """
        
        # Checks to see if face has moved significantly. If so, resets history.
        if(self._faceHasMoved(rects['face'])):
            self.clear();
                
        # Loop over rectangles, adding non-empty ones to history
        for key,rect in rects.items():
            if len(rect) is not 4:
                self.qualityHistory[key].append(0);
                continue;
            self.rectHistory[key].append(rect);
            self.qualityHistory[key].append(1);
#            print 'appended to qHist[',key,']';
        
        # Update moving average stats
        self._updateAverages();

    def getPreviousFaceRects(self):
        if len(self.rectHistory['face']) is 0:
            return numpy.array([],dtype=numpy.int32);
        else:
            return self.rectHistory['face'][-1];
    
    def getEyeRects(self):
        """Get array of eye rectangles"""
        return [self.rectAverage['eyeLeft'], self.rectAverage['eyeRight']];
    
    def getFaceRect(self):
        """Get face rectangle"""
        return self.rectAverage['face'];

    def getEyeLine(self):
        """Returns Points to create line along axis of eyes"""
        left,right = self.getEyeRects();
        
        if len(left) is not 4 or len(right) is not 4:
            return [(0,0),(0,0)];
        
        leftPoint = (left[0], ((left[1] + left[3])/2));
        rightPoint = (right[2], ((right[1] + right[3])/2));
        return [leftPoint,rightPoint];
        
    def clear(self):
        """ Resets Eye History"""
        for key,value in self.rectAverage.items():
            self.rectAverage[key] = numpy.array([],dtype=numpy.int32);
            self.rectHistory[key].clear();
            self.qualityHistory[key].clear();

    def _faceHasMoved(self, recentFaceRect):
        """Determines if face has just moved, requiring history reset"""
    
        # If no face found, return true
        if(len(recentFaceRect) is not 4):
            return True;

        history = self.rectHistory['face'];
        
        if len(history) is not self.QUEUE_MAXLEN:
            return False;

        old = history[self.QUEUE_MAXLEN - 10];
        oldX = (old[0] + old[2]) / 2.0;
        oldY = (old[1] + old[3]) / 2.0;
        recentX = (recentFaceRect[0] + recentFaceRect[2]) / 2.0;
        recentY = (recentFaceRect[1] + recentFaceRect[3]) / 2.0;
        change = ((recentX-oldX)**2 + (recentY-oldY)**2)**0.5; # sqrt(a^2+b^2)
        return True if change > 15 else False;

    def _updateAverages(self):
        """Update position rectangle moving averages"""
        for key,queue in self.rectHistory.items():
            if len(queue) is 0:
                continue;
            self.rectAverage[key] = sum(queue) / len(queue);
        
        faceQ = numpy.mean(self.qualityHistory['face']);
        eyeLeftQ = numpy.mean(self.qualityHistory['eyeLeft']);
        eyeRightQ = numpy.mean(self.qualityHistory['eyeRight']);
        
#        print 'Quality:    ', faceQ, eyeLeftQ, eyeRightQ;
#        print 'QHistory: ', self.qualityHistory['face'], self.qualityHistory['eyeLeft'], self.qualityHistory['eyeRight'];
#        print '--------------';

        #print 'QHistSizes: ', len(self.qualityHistory['face']), len(self.qualityHistory['eyeLeft']), len(self.qualityHistory['eyeRight']);

