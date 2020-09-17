import torch
import torch.nn       as     nn
import models         as     my_model
import numpy          as     np
from   torch.autograd import Variable
from   scipy          import stats
from datetime import datetime
import os, sys

####################################################
############### Input parameres ####################
####################################################

INPUT1  = "../distance_1/saved_distances.npy"
INPUT2  = "../distance_2/saved_distances.npy"
INPUT3  = "../distance_3/saved_distances.npy"
INPUT4  = "../distance_4/saved_distances.npy"

with open(INPUT1, 'rb') as f:
	data1 = np.load(f)

[M1, M2, N_each] = data1.shape 

with open(INPUT2, 'rb') as f:
	data2 = np.load(f)

with open(INPUT3, 'rb') as f:
	data3 = np.load(f)

with open(INPUT4, 'rb') as f:
	data4 = np.load(f)

N_each = 4160
N = 4 * N_each
M = M1 * M2
data = np.zeros((N, M))
inputDim     = M 
outputDim    = 2 
epochs       = 5000
learningRate = 0.002
restart      = 0
PATH         = "./models/07-02-2020_11-35-47_AM.pt"

print ("------------------------------------------")
print ("The initial data set dimansion is : " + str(N) + " by " + str(M))
print ("Epochs : " + str(epochs))
print ("Learning Rate : " + str(learningRate))
print ("------------------------------------------")

OUTPUT1 = open("latent_space_1.txt"  , "w")
OUTPUT2 = open("latent_space_2.txt"  , "w")
OUTPUT3 = open("latent_space_3.txt"  , "w")
OUTPUT4 = open("latent_space_4.txt"  , "w")
OUTPUT5 = open("loss.txt"            , "w")
####################################################

for k in range(0, M1):
	for j in range(0, M2):
		for i in range(0, N_each):
			data[i][k * M2 + j]                   = data1[k][j][i]
			data[i + N_each][k * M2 + j]          = data2[k][j][i]
			data[i + int(N_each * 2)][k * M2 + j] = data3[k][j][i]
			data[i + int(N_each * 3)][k * M2 + j] = data4[k][j][i]

print (data.shape)

for j in range(0, M):
	data[:,j] = stats.zscore(data[:,j])

if restart == 1:
	model = torch.load(PATH)
	model.eval()
	print ("Using the prvious model ...")
else:
	model    = my_model.autoencoder(inputDim, outputDim)
	print ("Using a new model ...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
model.to(device)

criterion = nn.SmoothL1Loss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, amsgrad=True)

inputs = Variable(torch.from_numpy(data).float())
labels = Variable(torch.from_numpy(data).float())

inputs = inputs.to(device)
labels = inputs.to(device)

for epoch in range(epochs):
	optimizer.zero_grad()
	outputs = model(inputs)[0]
	loss = criterion(outputs, labels)
	loss.backward()
	optimizer.step()	
	if epoch % 100 == 0:
		print('epoch {}, loss {}'.format(epoch, loss.item()))
		OUTPUT5.write('{}    '.format(epoch))
		OUTPUT5.write('{}  \n'.format(loss.item()))

with torch.no_grad(): # we don't need gradients in the testing phase
	predicted = model(outputs)[1].data.cpu().numpy()

#[4 * N, n] = predicted.shape

for i in range(0, N_each):
	OUTPUT1.write('{}    '.format(predicted[i][0]))
	OUTPUT1.write('{}    '.format(predicted[i][1]))
	OUTPUT1.write('{}  \n'.format(i))

for i in range(N_each, 2 * N_each):
 	OUTPUT2.write('{}    '.format(predicted[i][0]))
	OUTPUT2.write('{}    '.format(predicted[i][1]))
 	OUTPUT2.write('{}  \n'.format(i - N_each))

for i in range(2 * N_each, 3 * N_each):
 	OUTPUT3.write('{}    '.format(predicted[i][0]))
 	OUTPUT3.write('{}    '.format(predicted[i][1]))
 	OUTPUT3.write('{}  \n'.format(i - 2 * N_each))

for i in range(3 * N_each, 4 * N_each):
 	OUTPUT4.write('{}    '.format(predicted[i][0]))
 	OUTPUT4.write('{}    '.format(predicted[i][1]))
 	OUTPUT4.write('{}  \n'.format(i - 3 * N_each))

model_name = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")

# Path to be created
PATH = "./models/" + str(model_name) + ".pt"

torch.save(model, PATH)












