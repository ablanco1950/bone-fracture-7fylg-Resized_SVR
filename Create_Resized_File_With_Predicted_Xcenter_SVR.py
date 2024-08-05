# Author Alfonso Blanco

import cv2

import numpy as np

import os
import re

import pickle #to save, load  the model

dirname= "bone-fracture-2\\test\\images"
dirnameLabels="bone-fracture-2\\test\\labels"
imageSize=640
########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     imgpathlabels = dirnameLabels + "\\"
     images = []
     TabFileName=[]
     TabImagesCV=[]
    
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
                 SwEmpty=0
                 for linea in f:
                 
                      cont= cont +1
                      #print(cont)
                      xywh=linea[2:]
                      #print(len(xywh))
                      
                      #print(xywh)
                      
                      SwEmpty=1
                      xywh=xywh.split(" ")
                     
                      Yxmidpoint.append(xywh[0])
                      Yymidpoint.append(xywh[1])
                      Ywmidpoint.append(xywh[2])
                      Yhmidpoint.append(xywh[3])
                     
                     
                      break
                 if SwEmpty==0 :
                      contRejected=contRejected+1
                      print(" REJECTED HAS NO LABELS " + filename)
                      continue

                 

                 
                 result = cv2.resize(image, (imageSize,imageSize), interpolation = cv2.INTER_AREA)
                 TabImagesCV.append(result)
                 # TO REDUCE MEMORY PROBLEMS, CONVERT TO GRAY
                 cv2.imwrite("pptest.jpg", result)

                 result= cv2.imread("pptest.jpg", cv2.IMREAD_GRAYSCALE)
                 result=result.flatten()
                 #print(len(image))
                 images.append(result)
                 TabFileName.append(filename)
                 
                 
                 cont+=1
     
     return  TabImagesCV, images, TabFileName, Yxmidpoint, Yymidpoint

###########################################################
# MAIN
##########################################################

#TabFileLabelsName, Yxmidpoint, Yymidpoint, Ywmidpoint, Yhmidpoint= loadlabels(dirnameLabels)
imagesCV, X_test, TabFileName, Yxmidpoint, Yymidpoint=loadimages(dirname)

print("Number of images to test : " + str(len(X_test)))

#imagesCV, X_test, TabFileName=loadimages(dirname)

#print("Number of images to test : " + str(len(TabFileLabelsName)))


# https://medium.com/@niousha.rf/support-vector-regressor-theory-and-coding-exercise-in-python-ca6a7dfda927


from sklearn.preprocessing import StandardScaler

### When using StandardScaler(), fit() method expects a 2D array-like input
scaler = StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)

model_svr_lin_Yxmidpoint= pickle.load( open("svr_lin_Yxmidpoint.pickle", 'rb'))

import numpy as np
from sklearn import metrics

#### Test dataset - metrics ####
y_test_pred_Yxmidpoint = model_svr_lin_Yxmidpoint.predict(X_test_scaled)


print("predicted values for Xcenter:")
print(y_test_pred_Yxmidpoint)
print("true values for Xcenter:")
print(Yxmidpoint)
print("===")


with open( "Predicted_True_Xcenter.txt" ,"w") as  w:
    for i in range (len(y_test_pred_Yxmidpoint)):
          
             
                lineaw=[]
                lineaw.append(y_test_pred_Yxmidpoint[i]) 
                lineaw.append(Yxmidpoint[i])
                lineaWrite =','.join(lineaw)
                lineaWrite=lineaWrite + "\n"
                w.write(lineaWrite)
             
