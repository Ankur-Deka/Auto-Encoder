# Code for tresting the trained auto encoder
# By  Ankur Deka
# Date: 12th June, 2018

import argparse
import os
import torch
import utils
from model import AE
from matplotlib import pyplot as plt
import time

def get_args():
	parser=argparse.ArgumentParser()
	# epoch
	parser.add_argument('--epoch',type=int,default=4,help='epoch number to load')
	# test_image
	parser.add_argument('-l','--test_images', nargs='+',help='paths to test_images',required='True')
	args=parser.parse_args()
	return(args)

def main():
	# Get command line arguments
	args=get_args()

	# Create the model
	ae=AE()

	# Load the trained model weights
	load_dir='./save'
	checkpoint_path = os.path.join(load_dir,str(args.epoch)+'.tar')
	if os.path.isfile(checkpoint_path):
		print('Loading checkpoint')
		checkpoint = torch.load(checkpoint_path)
		model_epoch = checkpoint['epoch']
		ae.load_state_dict(checkpoint['state_dict'])
		print('Loaded checkpoint at {}'.format(model_epoch))
	else:
		print('Checkpoint {} not available'.format(args.epoch))

	# Evaluate
	for path in args.test_images:
		img, x = utils.load_test(path)
		print(img.shape,x.shape)
		start_time=time.clock()
		enc, dec = ae(x)
		end_time=time.clock()		
		print('Tested in {} seconds'.format(end_time-start_time))
		dec = dec.view(60,60).data.numpy()
		plt.subplot(121)
		plt.imshow(img,cmap='gray')
		plt.subplot(122)
		plt.imshow(dec,cmap='gray')
		plt.show()


if __name__=='__main__':
	main()