import numpy as np
import nibabel as nib
import glob
import os

trainX = []
files = sorted(glob.glob("Brain_Image/*.nii"))
print len(files)
for f in files:
	img = nib.load(f)
	img_data = img.get_data()
	img_data = img_data[:, :, 0:146]
	img_data = np.asarray(img_data)
#	if(img_data.shape[2]<146):
#		print f
	trainX.append(img_data)
print len(trainX)
#trainX1 = trainX[:100]
#trainX1 = np.asarray(trainX1)
#print trainX1.shape
#trainX = np.asarray(trainX)
#print trainX.shape
#print len(trainX)

