from FindModel import findModel
import json
import torch
from torch import nn
from torch import optim

from torchvision import datasets as d
from torchvision import transforms as t
from torchvision.transforms import Compose as c
from torch.utils.data import DataLoader as DL
from collections import OrderedDict
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("dir", default='./flowers', help="Image Folder")
parser.add_argument("--arch", default='vgg16', help="CNN Model Architecture")
parser.add_argument("--learning_rate", default=0.001, help="learning rate of the optimizer")
parser.add_argument("--hidden_units", default=2048, help="hidden units for classifier's neural network")
parser.add_argument("--save_dir", default=".", help="checkpoint save ")
parser.add_argument("--epochs", default=1, help="epoch of training")
parser.add_argument("--gpu", action='store_true', default=False, help="use gpu for training")
parser.parse_args()

save_dir = parser.parse_args().save_dir
data_dir = parser.parse_args().dir
arch = parser.parse_args().arch
epochs = parser.parse_args().epochs
learning_rate = parser.parse_args().learning_rate
gpu = parser.parse_args().gpu
hidden_units = parser.parse_args().hidden_units

torchdevice = "cuda" if gpu and torch.cuda.is_available() else "cpu"
device = torch.device(torchdevice)
print('----------------------------------------------')
print(f"using the {torchdevice}")
print('----------------------------------------------')

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

training_data = c([
    t.RandomRotation(60),
    t.RandomResizedCrop(224),
    t.RandomHorizontalFlip(),
    t.ToTensor(),
    t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_data = c([
    t.Resize(255),
    t.CenterCrop(224),
    t.ToTensor(),
    t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_data = c([
    t.Resize(255),
    t.CenterCrop(224),
    t.ToTensor(),
    t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

training_dataset = d.ImageFolder(train_dir, transform=training_data)
valid_dataset = d.ImageFolder(valid_dir, transform=valid_data)
test_dataset = d.ImageFolder(test_dir, transform=test_data)

train_loader = DL(training_dataset, batch_size=32, shuffle=True)
validation_loader = DL(valid_dataset, batch_size=32)
test_loader = DL(test_dataset, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# TODO: Build and train your network
# Load a pre-trained network
# (If you need a starting point, the VGG networks work great and are straightforward to use)


def buildModel(arch=arch, hidden_units=hidden_units):
    model = findModel(arch)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 5024)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.20)),
        ('fc2', nn.Linear(5024, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))]))

    return model


model = buildModel().to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


def trainIt(epoch=int(epochs)):
    # Train the classifier layers using backpropagation using the pre-trained network to get the features

    step = 0
    running_loss = 0
    print_every = 15
    for e in range(epoch):
        for images, labels in train_loader:
            step += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps, labels)
                    ps = torch.exp(logps)
                    topps, top_class = ps.topk(1, dim=1)
                    equallity = top_class == labels.view(*top_class.shape)
                    if torchdevice == 'cuda':
                        accuracy += torch.mean(equallity.type(torch.cuda.FloatTensor))
                    else:
                        accuracy += torch.mean(equallity.type(torch.FloatTensor))

                print(f"epoch:{e}/{epoch}..\t"
                      f"training loss:{running_loss / print_every}..\t"
                      f"test loss:{test_loss / len(test_loader)}..\t"
                      f"test accuracy:{accuracy / len(test_loader)}")


trainIt()
print('saving checkpoint')
# TODO: Save the checkpoint
checkpoint = {
    'state_dict': model.state_dict(),
    'classifier': model.classifier,
    'class_to_idx': test_dataset.class_to_idx,
    "arch": arch
}

torch.save(checkpoint, f"{save_dir}/checkpoint_{arch}.pth")
