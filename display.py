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

from util import Util


 
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



class Display:

    def renderScene(self, frame, model, rects=False):
        """Draw face and eyes onto image, then display it"""
        
        # Get Coordinates
        eyeRects = model.getEyeRects();
        faceRect = model.getFaceRect();
        linePoints = model.getEyeLine();
    
        # Draw Shapes and display frame
        self.drawLine(frame, linePoints[0],linePoints[1],(0, 0, 255));
        self.drawRectangle(frame, faceRect, (0, 0, 255));
        self.drawRectangle(frame, eyeRects[0], (0, 255, 0));
        self.drawRectangle(frame, eyeRects[1], (0, 255, 0));
        
        if rects is not False:
            self.drawRectangle(frame, rects['eyeLeft'], (152,251,152));
            self.drawRectangle(frame, rects['eyeRight'],(152,251,152));
        
        cv2.imshow("Video", frame);
    
    def renderEyes(self, frame, model):
    
        eyeRects = model.getEyeRects();
        
        if len(eyeRects[0]) is 4:
            cropTop = 0.2;
            cropBottom = 0.2;
            eyeLeftHeight = eyeRects[0][3] - eyeRects[0][1];
            eyeLeftWidth = eyeRects[0][2] - eyeRects[0][0];
            eyeLeftIMG = frame[(eyeRects[0][1]+eyeLeftHeight*cropTop):(eyeRects[0][3]-eyeLeftHeight*cropBottom), eyeRects[0][0]:eyeRects[0][2]];
            eyeLeftExpanded =             frame[(eyeRects[0][1]+eyeLeftHeight*(cropTop/2)):(eyeRects[0][3]-eyeLeftHeight*(cropBottom/2)), (eyeRects[0][0]-eyeLeftWidth*cropTop):(eyeRects[0][2]+eyeLeftWidth*cropTop)];
            
            #eyeLeftExpanded = cv2.resize(eyeLeftExpanded,None,fx=0.5,fy=0.5);
            eyeLeftExpanded = cv2.cvtColor(eyeLeftExpanded,cv.CV_BGR2GRAY);
            eyeLeftExpanded = cv2.equalizeHist(eyeLeftExpanded);
            eyeLeftExpanded = cv2.GaussianBlur(eyeLeftExpanded,(7,7),4);
            
            cv2.imshow("eyeLeftExpanded",eyeLeftExpanded);
            cv2.moveWindow("eyeLeftExpanded",0, 500);

            
            # Grayscale Eye
            eyeLeftBW = cv2.cvtColor(eyeLeftIMG,cv.CV_BGR2GRAY);

            # Equalize Eye and find Average Eye
            eyeLeftEqualized = cv2.equalizeHist(eyeLeftBW);
            #eyeLeftAvg = ((eyeLeftBW.astype(numpy.float32) + eyeLeftEqualized.astype(numpy.float32)) / 2.0).astype(numpy.uint8);


            # Eye Contrast Enhancement
            eyeLeftContrasted = Util.contrast(eyeLeftIMG,1.5);
             #eyeLeftHiContrast = Util.contrast(eyeLeftIMG,2);
            
            # Blur Eye
            eyeLeftBlurredBW = cv2.GaussianBlur(eyeLeftEqualized,(7,7),1);
            eyeLeftBlurThreshBW = Util.threshold(eyeLeftBlurredBW,100);
            
            # Split into blue, green and red channels
            B,G,R = cv2.split(eyeLeftIMG);
            B = cv2.equalizeHist(B);
            BBlurred = cv2.GaussianBlur(B,(7,7),1);
            #G = cv2.equalizeHist(G);
            #R = cv2.equalizeHist(R);
            
            # Thresholding
#            thresholded = Util.threshold(B,200);

            # Good Features To Track
            eyeFeatures = cv2.goodFeaturesToTrack(eyeLeftExpanded,10,0.3,10);
            eyeLeftFeatureMap = cv2.cvtColor(eyeLeftExpanded,cv.CV_GRAY2BGR);
            if eyeFeatures is not None:
                for c in eyeFeatures:
                    if len(c) is 0:
                        continue;
                    corner = c[0].astype(numpy.int32);#*2;
                    
                    center = (corner[0], corner[1]);
                    cv2.circle(eyeLeftFeatureMap,center,2,(0, 255, 0),-1);
                    
            cv2.imshow("eyeLeftFeatures",eyeLeftFeatureMap);
            cv2.moveWindow("eyeLeftFeatures",0,600);
            
            # Harris Corner Detection
#             cornerMap = cv2.cornerHarris(eyeLeftEqualized,2,3,0.004);
#             eyeLeftCorners = cv2.cvtColor(eyeLeftEqualized,cv.CV_GRAY2BGR);
#             size = eyeLeftBlurredBW.shape;
#     #             print size
#     #             
#     #             cornerValues = cornerMap.flatten();
#     # 
#     #             hist, bins = numpy.histogram(cornerValues,bins = 50)
#     #             width = 0.7*(bins[1]-bins[0])
#     #             center = (bins[:-1]+bins[1:])/2
#     #             plt.bar(center, hist, align = 'center', width = width)
#     #             plt.show()
#             
#             for i in range(0,size[0]):
#                 for j in range(0,size[1]):
#                     
#                     if cornerMap[i][j] > 0.00025:
#                         cv2.circle(eyeLeftCorners,(i,j),2,(0, 255, 0),-1);
#             
#             cv2.imshow("eyeLeftCorners",eyeLeftCorners);
#             cv2.moveWindow("eyeLeftCorners",0,750);

            
            
            
            
            # Hough Transformation
            irisMinRadius = int(round(eyeLeftEqualized.shape[1]*0.1));
            irisMaxRadius = int(round(eyeLeftEqualized.shape[1]*0.25));
            # TODO update this based on previously-found iris radii
            minDistance = irisMaxRadius*2;
            circles = cv2.HoughCircles(eyeLeftBlurredBW, cv.CV_HOUGH_GRADIENT, 2.5, minDistance, param1=30, param2=30,minRadius=irisMinRadius,maxRadius=irisMaxRadius);
            
            eyeLeftBW_C = cv2.cvtColor(B,cv.CV_GRAY2BGR);
            if circles is not None and len(circles)>0:
                #print circles
                for c in circles[0]:
                    c = c.astype(numpy.int32);
                    
                    center = (c[0], c[1]);
                    #print 'center=',center,', radius=',c[2];
                    cv2.circle(eyeLeftBW_C,(c[0],c[1]),c[2],(0, 255, 0));
            
            cv2.imshow("eyeLeftBW_C",eyeLeftBW_C);
            cv2.moveWindow("eyeLeftBW_C",150,600);
            
            # Display Original Eye Image
            cv2.imshow("eyeLeft",eyeLeftIMG);
            cv2.moveWindow("eyeLeft",0,350);
            
            # Display Blurred Images
#            cv2.imshow("eyeLeftBW",eyeLeftBW);
#             cv2.moveWindow("eyeLeftBW",0,475);
#            cv2.imshow("eyeLeftBlurredBW",eyeLeftBlurredBW);
#             cv2.moveWindow("eyeLeftBlurredBW",150,475);
#            cv2.imshow("eyeLeftBlurThreshBW",eyeLeftBlurThreshBW);
#             cv2.moveWindow("eyeLeftBlurThreshBW",300,475);
             
 
            cv2.imshow("edges",cv2.Canny(eyeLeftBW,15,30));
            cv2.moveWindow("edges",0,550);
            cv2.imshow("blurrededges",cv2.Canny(eyeLeftBlurredBW,15,30));
            cv2.moveWindow("blurrededges",150,550);
#            cv2.imshow("blurredthreshedges",cv2.Canny(eyeLeftBlurThreshBW,15,30));
#            cv2.moveWindow("blurredthreshedges",300,550);

            
            # Display B, G, R Channels
#             cv2.imshow("B",B);
#             cv2.moveWindow("B",0,475);
#             cv2.imshow("G",G);
#             cv2.moveWindow("G",150,475);
#             cv2.imshow("R",R);
#             cv2.moveWindow("R",300,475);            
            
            # Display Thresholded Eye
#            cv2.imshow("eyeLeftThresh",thresholded);
#            cv2.moveWindow("eyeLeftThresh",300,750);

            # Display Histogram
        #    self.drawHistogram(eyeLeftContrasted);
            
            # Display Contrasted Images
#             cv2.imshow("eyeLeftContrasted",eyeLeftContrasted);
#             cv2.moveWindow("eyeLeftContrasted",0, 750);
#             cv2.imshow("eyeLeftHiContrast",eyeLeftHiContrast);
#             cv2.moveWindow("eyeLeftHiContrast",150, 750);

            
            # HSV Equalization
#             eyeLeftEQ = Util.equalizeHSV(eyeLeftIMG);
#             cv2.imshow("eyeLeftEQ",eyeLeftEQ);
#             cv2.moveWindow("eyeLeftEQ",0,500);
            
            # K-Means Color Quantization/Clustering
#             z = eyeLeftEQ.reshape((-1,3))
#             k = 4;           # Number of clusters
#             center,dist = vq.kmeans(z,k)
#             code,distance = vq.vq(z,center)
#             res = center[code]
#             eyeLeftQ = res.reshape((eyeLeftEQ.shape))
#             cv2.imshow("eyeLeftQ",eyeLeftQ);
#             cv2.moveWindow("eyeLeftQ",0,650);

        if len(eyeRects[1]) is 4:
            eyeRightIMG = frame[eyeRects[1][1]:eyeRects[1][3], eyeRects[1][0]:eyeRects[1][2]];
            cv2.imshow("eyeRight",eyeRightIMG);
            cv2.moveWindow("eyeRight",200,350);

    @staticmethod
    def drawHistogram(img,color=True,windowName='drawHistogram'):
        h = numpy.zeros((300,256,3))
         
        bins = numpy.arange(256).reshape(256,1)
        
        if color:
            channels =[ (255,0,0),(0,255,0),(0,0,255) ];
        else:
            channels = [(255,255,255)];
        
        for ch, col in enumerate(channels):
            hist_item = cv2.calcHist([img],[ch],None,[256],[0,255])
            #cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
            hist=numpy.int32(numpy.around(hist_item))
            pts = numpy.column_stack((bins,hist))
            #if ch is 0:
            cv2.polylines(h,[pts],False,col)
         
        h=numpy.flipud(h)
         
        cv2.imshow(windowName,h);
    
    @staticmethod
    def drawLine(img, p1, p2, color):
        """Draw lines on image"""
        p1 = (int(p1[0]*DISPLAY_SCALE), int(p1[1]*DISPLAY_SCALE));
        p2 = (int(p2[0]*DISPLAY_SCALE), int(p2[1]*DISPLAY_SCALE));
        cv2.line(img, p1, p2,(0, 0, 255));
    
    @staticmethod
    def drawRectangle(img, rect, color):
        """Draw rectangles on image"""
        
        if len(rect) is not 4:
            # TODO throw error
            return;
        rect = rect * DISPLAY_SCALE;
        x1, y1, x2, y2 = rect.astype(numpy.int32);
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2);
