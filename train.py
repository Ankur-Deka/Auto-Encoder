# Training the auto encoder

import os
import torch
import torch.nn as nn
import utils
from model import AE

#Get the data
train_loader=utils.train_loader
valid_loader=utils.valid_loader
print('Total trainning batch number: {}'.format(len(train_loader)))
print('Total validing batch number: {}'.format(len(valid_loader)))

#Save directory
save_dir='./save'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

#Log directory
log_dir='./log'
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
log_file = open(os.path.join(log_dir, 'loss.txt'), 'w')


#Define model's training parameters
lr=0.0005
num_epochs=5
ae=AE()
criterion=nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(),
			lr=lr,
			weight_decay=1e-5)


#Validation Loss
def eval_loss():
	val_loss=[]
	for batch_id,(x,label) in enumerate(train_loader):
		enc,dec=ae(x)
		loss=criterion(dec,x).item()
		val_loss.append(loss)
		avg_loss=sum(val_loss)/len(val_loss)
		return(avg_loss)

#Train the model
for epoch in range(num_epochs):
	for batch_id,(x,label) in enumerate(train_loader):
		optimizer.zero_grad()
		enc,dec=ae(x)
		loss=criterion(dec,x)
		loss.backward()
		optimizer.step()
		val_loss=eval_loss()
		print('Epoch: {}, Batch ID: {}, Training Loss: {}, Validation Loss: {}'.format(epoch, batch_id, loss.item(), val_loss))
		log_file.write(str(epoch)+','+str(batch_id)+','+str(loss.item())+','+str(val_loss)+'\n')
			
	#save model after every epoch
	print('Saving model')
	save_loc=save_dir+'/'+str(epoch)+'.tar'
	torch.save({'epoch': epoch,
				'state_dict': ae.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()
				}, save_loc)

log_file.close()