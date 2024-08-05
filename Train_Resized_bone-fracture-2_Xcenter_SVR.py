# Author Alfonso Blanco

import cv2
import time
Ini=time.time()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import os
import re

import imutils

import pickle #to save the model

dirname= "bone-fracture-2\\trainvalid\\images"
dirnameLabels="bone-fracture-2\\trainvalid\\labels"

imageSize=640


########################################################################
def loadimages(dirname):

     imgpath = dirname + "\\"
     imgpathlabels = dirnameLabels + "\\"
     images = []
     TabFileName=[]
   
    
     print("Reading imagenes from ",imgpath)
     
     
     cont=0
     contRejected=0
     Yxmidpoint=[]
     Yymidpoint=[]
     Ywmidpoint=[]
     Yhmidpoint=[]
     
     for root, dirnames, filenames in os.walk(imgpath):
        
         
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                             
                 image = cv2.imread(filepath)
                 
                 filenameLabel=filename[0:len(filename)-3]+ "txt"
                 filenameLabel=imgpathlabels + filenameLabel
                 #print( filenameLabel)
                                
                 f=open(filenameLabel,"r")
                                
                 xywh=""
                 # label file may be empty, in that case ignore
                 SwEmpty=0
                 for linea in f:
                 
                      cont= cont +1
                     
                      xywh=linea[2:]
                      
                      SwEmpty=1
                      xywh=xywh.split(" ")
                      
                      Yxmidpoint.append(str(float(xywh[0])))
                      Yymidpoint.append(str(float(xywh[1])))
                      Ywmidpoint.append(str(float(xywh[2])))
                      Yhmidpoint.append(str(float(xywh[3])))   
                     
                     
                      break
                 if SwEmpty==0 :
                      contRejected=contRejected+1
                      print(" REJECTED, HAS NO LABELS " +  filename)
                      continue

                 

                  
                 result= cv2.resize(image, (imageSize,imageSize), interpolation = cv2.INTER_AREA)

                 
                 

                 # TO REDUCE MEMORY PROBLEMS, CONVERT TO GRAY, No TRY WITH 3 Channels
                 cv2.imwrite("pp.jpg", result)

                 result= cv2.imread("pp.jpg", cv2.IMREAD_GRAYSCALE)
                 
                 result=result.flatten()
                 #print(len(image))
                 images.append(result)
                 TabFileName.append(filename)
                 
                 #cont+=1
                 #if cont > 9: break
     print ("Total rejected " + str(contRejected))
     return images, TabFileName, Yxmidpoint, Yymidpoint


###########################################################
# MAIN
##########################################################

X_train, TabFileName, Yxmidpoint, Yymidpoint=loadimages(dirname)

print("Number of images to train : " + str(len(X_train)))

# https://medium.com/@niousha.rf/support-vector-regressor-theory-and-coding-exercise-in-python-ca6a7dfda927

#from sklearn.model_selection import train_test_split

#X_train, X_test = train_test_split(X_train, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

### When using StandardScaler(), fit() method expects a 2D array-like input
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

from sklearn.svm import SVR

#svr_lin = SVR(kernel = 'linear')
#svr_lin = SVR(kernel = 'poly')

from sklearn.svm import SVC
import pickle #to save the model

from sklearn.multiclass import OneVsRestClassifier


#https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
svr_lin =  OneVsRestClassifier(SVC(kernel='linear', probability=True,  max_iter=1000)) #Creates model instance here
# probar esto
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#svr_rbf = SVR(kernel = 'rbf')
#svr_poly = SVR(kernel = 'poly')


svr_lin.fit(X_train_scaled, Yxmidpoint)
#svr_rbf.fit(X_train_scaled, y_train)
#svr_poly.fit(X_train_scaled, y_train)

# https://stackoverflow.com/questions/23582489/python-pickle-protocol-choice
#pickle.dump(svr_lin, open("svr_lin_Yymidpoint.pickle", 'wb')) #save model as a pickled file
pickle.dump(svr_lin,open("svr_lin_Yxmidpoint.pickle",'wb'), protocol=5) #save model as a pickled file

