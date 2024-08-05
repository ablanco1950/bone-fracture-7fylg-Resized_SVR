# bone-fracture-7fylg-Resized_SVR

From dataset https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg a model is obtained, based on ML (SVR), with that custom dataset, to indicate fractures in x-rays.

The train file should be downloaded from https://universe.roboflow.com/roboflow-100/bone-fracture-7fylg, but as SVR does not consider a process with train and valid, to get the maximum number of train images the folders with images and labels of train and valid should be combined into a single trainvalid folder.
This process can be avoided by downloading the attached bone-fracture-2.zip file and unzipping it so that there is only one bone-fracture-2 folder (be careful if unzipping creates two nested bone-fracture-2 folders, you would have to delete one)

Requirements:

Download all project datasets to a folder on disk.

You only need to have 50GB of free disk space (which is normal), you may need more than 16GB of RAM

You don't need a GPU

But because of this RAM limitation, you have to run the process in several steps:

1-
Create a model that obtains the Xcenter coordinate of each fracture, by running:

Train_Resized_bone-fracture-2_Xcenter_SVR.py

it takes less than 2 hours and creates the svr_lin_Yxmidpoint.pickle model with a size of about 24 Gb

2-
Create a model that obtains the Ycenter coordinate of each fracture, by running:

Train_Resized_bone-fracture-2_Ycenter_SVR.py

it takes less than 2 hours and creates the svr_lin_Yymidpoint.pickle model with a size of about 24 Gb

3-
From the test images and labels (test folder in bone-fracture.2) and based on the svr_lin_Yxmidpoint.pickle model obtained in step 1- , create a .txt file with the predicted Xcenter coordinates for each fracture of the test images (Predicted_True_Xcenter.txt file)

by running:

Create_Resized_File_With_Predicted_Xcenter_SVR.py

4-

From the test images and labels (test folder in bone-fracture.2) and based on the svr_lin_Yymidpoint.pickle model obtained in step 2- , create a .txt file with the predicted Ycenter coordinates for each fracture of the test images (Predicted_True_Ycenter.txt file)

by running:

Create_Resized_File_With_Predicted_Ycenter_SVR.py

5- create a Evaluation by running:

Evaluate_Resized_bone-fracture-2_Xcenter_Ycenter_SVR.py

The test images appear on the screen with a blue circle indicating the predicted fracture and a green rectangle indicating where the fracture was indicated when the image was labeled.

The model is obtained in a short time with the resources of a laptop and the results do not differ much from those obtained by consulting each image at the address https://universe.roboflow.com/science-research/science-research-2022:-bone-fracture-detection (you may need to be a roboflow user, which is achieved at no cost or difficulty)

The images have been resized, since SVR does not support images of different sizes, so when presented they appear a little deformed (to be improved)
