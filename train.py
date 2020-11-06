import json
import numpy as np
from utils import tokenize,stem,bow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet



with open('intentions.json','r') as file:
    intentions = json.load(file)


all_words,tags,xy = [],[],[]
ignore = ['?',',','!','.','-','\'']


for intent in intentions['intentions']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        tokenied_sentence = tokenize(pattern)
        xy.append((tokenied_sentence, tag))

        stemmed_words = [stem(w) for w in tokenied_sentence if w not in ignore]
        all_words.extend(stemmed_words)

all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_train, y_train = [],[]
for (tokenied_sentence, tag) in xy:
    bag = bow(tokenied_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))
    
X_train, y_train = np.array(X_train), y_train


class MyDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]    

    def __len__(self):
        return self.n_samples


input_size = len(all_words)
hidden_size = 16
ouput_size = len(tags)
lr = 0.001
num_epochs = 1000


dataset = MyDataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=2)


device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, ouput_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for (X, y) in train_loader:
        X = X.to(device)
        y = y.to(device)

        y_hat = model(X)
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%100 == 0:
        print("Epoch : ",epoch+1, "/", num_epochs," loss : ", loss.item())

print("Final Loss : ", loss.item())

