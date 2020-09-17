import torch
import torch.nn as nn
##########################################################
############# Class for VAE with 3 layers ################
##########################################################

class autoencoder(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        H = 512
        K = 256
        L = 128
        self.hidden1 = nn.Linear(inputSize, H)
        self.hidden2 = nn.Linear(H, K)
        self.hidden3 = nn.Linear(K, L)
        self.hidden4 = nn.Linear(L, outputSize)
        self.hidden5 = nn.Linear(outputSize, L)
        self.hidden6 = nn.Linear(L, K)
        self.hidden7 = nn.Linear(K, H)
        self.hidden8 = nn.Linear(H, inputSize)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.hidden3(x)
        x = self.sigmoid(x)
        y = self.hidden4(x)
        x = self.hidden5(y)
        x = self.sigmoid(x)
        x = self.hidden6(x)
        x = self.sigmoid(x)
        x = self.hidden7(x)
        x = self.sigmoid(x)
        x = self.hidden8(x)

        return x, y