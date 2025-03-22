import torch 
from torch import nn, optim

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class XOR_NN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden = nn.Linear(2, 4) # hidden layer with 4 neurons
        self.output = nn.Linear(4, 1) # output layer with 1 neuron 
        self.activation = nn.ReLU() # activation function for hidden layer 
        self.sigmoid = nn.Sigmoid() # activation function for output layer 

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x 
    
model = XOR_NN()

# loss function and optimizer 
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# train model 
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad() # zero the optimizer
    output = model(X)     # using the model forward function with the X dataset
    loss = criterion(output, y) # calculate loss
    loss.backward() # backward propagation to adjust the model weights
    optimizer.step() # updating the model weights

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model 

with torch.no_grad():
    predictions = model(X)
    print('--------------------')
    print('Predictions:')
    for i in range(len(X)):
        print(f'Input: {X[i].tolist()} -> Predicted: {predictions[i].item():.4f}, Rounded: {round(predictions[i].item())}, Excpected: {y[i].item()}')