
import random
import json
import torch
from model import NeuralNet
from utils import bow, tokenize, stem

device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

with open('intentions.json','r') as file:
    intentions = json.load(file)

data=torch.load( "saved_model.pth")

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
model_state = data["model_state"]
all_words = data["all_words"]
tags = data["tags"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bow(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intentions['intentions']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")