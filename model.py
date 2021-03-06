# Auto encoder-decoder model for 60x60 grayscale images
# Code by: Ankur Deka
# Date: 11th June, 2018

import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder(nn.Module):
	def __init__(self):
		super(encoder,self).__init__()
		self.conv1=nn.Conv2d(1,64,5,padding=2)	#padding is the padding size
		self.conv2=nn.Conv2d(64,32,3,padding=1)
		self.conv3=nn.Conv2d(32,8,3,padding=1)
		self.fc=nn.Linear(392,128)

	def forward(self,x):
		x=F.max_pool2d(F.relu(self.conv1(x)),2)	#2 means 2x2 max pool
		x=F.max_pool2d(F.relu(self.conv2(x)),2)
		x=F.max_pool2d(F.relu(self.conv3(x)),2)
		x=x.view(-1,self.num_flat_features(x))
		x=F.relu(self.fc(x))
		return(x)

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class decoder(nn.Module):
	def __init__(self):
		super(decoder,self).__init__()
		self.fc=nn.Linear(128,392)
		self.unpool1=nn.ConvTranspose2d(8,32,kernel_size=3,stride=2)
		self.unpool2=nn.ConvTranspose2d(32,64,kernel_size=5,stride=2,padding=2,output_padding=1)
		self.unpool3=nn.ConvTranspose2d(64,1,kernel_size=5,stride=2,padding=2,output_padding=1)

	def forward(self,y):
		y=self.fc(y)
		y=y.view(-1,8,7,7)
		y=F.relu(self.unpool1(y))
		y=F.relu(self.unpool2(y))
		y=F.relu(self.unpool3(y))
		return(y)

class AE(nn.Module):
	def __init__(self):
		super(AE,self).__init__()
		self.encoder=encoder()
		self.decoder=decoder()

	def forward(self,x):
		encoded=self.encoder(x)
		decoded=self.decoder(encoded)
		return(encoded,decoded)
