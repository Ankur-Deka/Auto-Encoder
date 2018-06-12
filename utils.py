# Code for handling datasets for traing and evaluation
# By Ankur Deka
# Data: 12th June, 2018

import os
import torch
from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms

root = './data'
if not os.path.exists(root):
	os.mkdir(root)

batch_size = 100

# if does not exist, download mnist dataset
trans=transforms.Compose([transforms.Resize(size=(60,60)),transforms.ToTensor()])
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
valid_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
				dataset=train_set,
				batch_size=batch_size,
				shuffle=True)
valid_loader = torch.utils.data.DataLoader(
				dataset=valid_set,
				batch_size=batch_size,
				shuffle=False)

# function to load images at test time
resize_trans=transforms.Compose([transforms.Resize(size=(60,60)),transforms.ToTensor()])
def load_test(path):
	if not os.path.exists(path):
		print('File {} doesn\'t exist'.format(path))
	else:	
		img = Image.open(path)
		img = img.convert(mode='L')
		img = trans(img)
		x = img.view(1,1,60,60)
		return(img.view(60,60).numpy(),x)
