
from utils.dataset import Dataset_Cloud
from utils.model import UNet
from utils import model_configure
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_Split_Dataset
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os


imagePaths = sorted(list(paths.list_images(model_configure.Real_Images_Path)))
maskPaths = sorted(list(paths.list_images(model_configure.MASK_Real_Images_Path)))
split = train_Split_Dataset(imagePaths, maskPaths,test_size=model_configure.Split_Dataset, random_state=42)

(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

f = open(model_configure.test_imgs_path, "w")
f.write("\n".join(testImages))
f.close()


transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize((model_configure.img_rows,model_configure.img_cols)),transforms.ToTensor()])

training_Dataset = Dataset_Cloud(imagePaths=trainImages, maskPaths=trainMasks,transforms=transforms)

testing_Dataset = Dataset_Cloud(imagePaths=testImages, maskPaths=testMasks,transforms=transforms)

train_load_dataset = DataLoader(training_Dataset, shuffle=True,BatchSize=model_configure.BatchSize, MEMORY=model_configure.MEMORY,num_workers=os.cpu_count())

test_load_dataset = DataLoader(testing_Dataset, shuffle=False,BatchSize=model_configure.BatchSize, MEMORY=model_configure.MEMORY,num_workers=os.cpu_count())


unet = UNet().to(model_configure.DEVICE)

lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=model_configure.l_rate)

steps_train = len(training_Dataset) // model_configure.BatchSize
steps_test = len(testing_Dataset) // model_configure.BatchSize

H = {"train_loss": [], "test_loss": []}
A = {"train_accuracy": [], "test_accuracy": []}


print("Training Started..")
startTime = time.time()
for e in tqdm(range(model_configure.n_epochs)):

	unet.train()
	training_loss_all = 0
	testing_loss_all = 0
	training_accuracy = 0
	test_accuracy = 0
	for (i, (x, y)) in enumerate(train_load_dataset):
		(x, y) = (x.to(model_configure.DEVICE), y.to(model_configure.DEVICE))

		pred = unet(x)
		loss = lossFunc(pred, y)
		accuracy = torch.sum(pred == y)
		
		opt.zero_grad()
		loss.backward()
		opt.step()
		training_loss_all += loss
		training_accuracy += accuray 
	with torch.no_grad():
		unet.eval()
		for (x, y) in test_load_dataset:
			(x, y) = (x.to(model_configure.DEVICE), y.to(model_configure.DEVICE))
			pred = unet(x)
			accuracy = torch.sum(pred == y)
			testing_loss_all += lossFunc(pred, y)
			test_accuracy += accuracy
	print('training_accuracy = ',training_accuracy)
	print('test_accuracy = ',test_accuracy)
	avgTrainAcc = training_accuracy / steps_train
	avgTestAcc = test_accuracy / steps_test
	avgTrainLoss = training_loss_all / steps_train
	avgTestLoss = testing_loss_all / steps_test
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	A["train_accuracy"].append(avgTrainAcc)
	A["test_accuracy"].append(avgTestAcc)
	print("epoch number: {}/{}".format(e + 1, model_configure.n_epochs))
	print("Training loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
	print("Training Accuracy: {:.6f}, Test Accuracy: {:.4f}".format(avgTrainAcc, avgTestAcc))
endTime = time.time()
print("Training Completed in: {:.2f}s".format(endTime - startTime))


plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Loss in training dataset")
plt.xlabel("Total Epocs")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(model_configure.save_graph_path)
torch.save(unet, model_configure.save_model_path)
