
from utils import model_configure
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


def plot_data(origImage, origMask, Mask_predict):
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(Mask_predict)
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	figure.tight_layout()
	figure.show() 
    
def predict_clouds(model, imagePath):
	model.eval()
	with torch.no_grad():
		image = cv2.imread(imagePath)
		
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0

		orig = image.copy()

		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(model_configure.MASK_Real_Images_Path,filename)
		groundTruthPath = groundTruthPath[:-4]+'.png'

		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (model_configure.img_rows,model_configure.img_rows))
        
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(model_configure.DEVICE)

		Mask_predict = model(image).squeeze()
		Mask_predict = torch.sigmoid(Mask_predict)
		Mask_predict = Mask_predict.cpu().numpy()
		Mask_predict = (Mask_predict > model_configure.THRESHOLD) * 255
		Mask_predict = Mask_predict.astype(np.uint8)
		return Mask_predict, gtMask
        
        
imagePaths = open(model_configure.test_imgs_path).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size = 1014)

unet = torch.load(model_configure.save_model_path).to(model_configure.DEVICE)
predicts = []
GTMasks = []
for path in imagePaths:
	predict, gtmask = predict_clouds(unet, path) 
	predicts.append(predict)
	GTMasks.append(gtmask)
