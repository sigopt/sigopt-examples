import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import orchestrate.io

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

batch_size = 100
train_val_split = int(len(train_dataset) * 0.9)
indices = list(range(len(train_dataset)))
train_idx, valid_idx = indices[:train_val_split], indices[train_val_split:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=valid_sampler)

activation_functions = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

optimizers = {
    'gradient_descent': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
}

def create_model():
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            conv1_output = orchestrate.io.assignment('conv1_output', default=20)
            conv1_kernel = orchestrate.io.assignment('conv1_kernel', default=5)
            conv1_act = orchestrate.io.assignment('conv1_act', default='relu')
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, conv1_output, kernel_size=conv1_kernel),
                nn.BatchNorm2d(conv1_output),
                activation_functions[conv1_act],
                nn.MaxPool2d(2)
            )
            conv2_output = orchestrate.io.assignment('conv2_output', default=20)
            conv2_kernel = orchestrate.io.assignment('conv2_kernel', default=5)
            conv2_act = orchestrate.io.assignment('conv2_act', default='relu')
            self.layer2 = nn.Sequential(
                nn.Conv2d(conv1_output, conv2_output, kernel_size=conv2_kernel),
                nn.BatchNorm2d(conv2_output),
                activation_functions[conv2_act],
                nn.MaxPool2d(2)
            )
            fc1_input_dim = (((((28 - conv1_kernel + 1) // 2) - conv2_kernel + 1) // 2) *
                             ((((28 - conv1_kernel + 1) // 2) - conv2_kernel + 1) // 2) *
                             conv2_output)
            fc1_hidden = orchestrate.io.assignment('fc1_hidden', default=500)
            fc1_act = orchestrate.io.assignment('fc1_act', default='sigmoid')
            self.fc1 = nn.Linear(fc1_input_dim, fc1_hidden)
            self.fc1_act = activation_functions[fc1_act]
            self.fc2 = nn.Linear(fc1_hidden, 10)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc1_act(out)
            out = self.fc2(out)
            return out

    cnn = CNN()
    cost = nn.CrossEntropyLoss()
    optimizer = optimizers[orchestrate.io.assignment('optimizer', default='adam')](
      cnn.parameters(),
      lr=10**orchestrate.io.assignment('log_learning_rate', default=-3.)
    )

    for _ in range(orchestrate.io.assignment('epochs', default=10)):
        for (images, labels) in train_loader:
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()

    return cnn

def evaluate_model():
    correct = 0
    total = 0
    cnn = create_model()
    cnn.eval()
    for images, labels in val_loader:
        images = Variable(images)
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return float(correct) / total

if __name__ == '__main__':
    orchestrate.io.log_metric('accuracy', evaluate_model())
