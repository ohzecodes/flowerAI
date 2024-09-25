from FindModel import findModel
import json
import torch
from torch import nn
from torch import optim
from torchvision import datasets as d
from Transformations import get_transformations
from torch.utils.data import DataLoader as DL
from collections import OrderedDict
from argparse import ArgumentParser

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--dir", default='./flowers', help="Image Folder")
parser.add_argument("--arch", default='vgg16', help="CNN Model Architecture")
parser.add_argument("--learning_rate", default=0.001, help="learning rate of the optimizer")
parser.add_argument("--hidden_units", default=2048, help="hidden units for classifier's neural network")
parser.add_argument("--save_dir", default=".", help="checkpoint save")
parser.add_argument("--epochs", default=1, help="epoch of training")
parser.add_argument("--gpu", action='store_true', default=False, help="use gpu for training")
args = parser.parse_args()

save_dir = args.save_dir
data_dir = args.dir
arch = args.arch
epochs = int(args.epochs)
learning_rate = float(args.learning_rate)
gpu = args.gpu
hidden_units = int(args.hidden_units)

torchdevice = "cuda" if gpu and torch.cuda.is_available() else "cpu"
device = torch.device(torchdevice)
print('----------------------------------------------')
print(f"using the {torchdevice}")
print('----------------------------------------------')

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transformations
train_transform, valid_transform, test_transform = get_transformations(arch)

# Load datasets
training_dataset = d.ImageFolder(train_dir, transform=train_transform)
valid_dataset = d.ImageFolder(valid_dir, transform=valid_transform)
test_dataset = d.ImageFolder(test_dir, transform=test_transform)

train_loader = DL(training_dataset, batch_size=32, shuffle=True)
validation_loader = DL(valid_dataset, batch_size=32)
test_loader = DL(test_dataset, batch_size=32)

# Load category names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Function to check output shape
def get_output_shape(model, input_size=(3, 224, 224)):
    dummy_input = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        output = model.features(dummy_input)  # Use the feature extractor part of the model
    return output.view(output.size(0), -1).size(1)  # Flatten output for classifier


# Function to build the model
def buildModel(arch, hidden_units):
    model = findModel(arch)
    for param in model.parameters():
        param.requires_grad = False

    # Get the output size from the feature extractor
    output_size = get_output_shape(model)

    # Modify the classifier
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(output_size, hidden_units)),  # Use dynamic output size
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.20)),
        ('fc2', nn.Linear(hidden_units, 102)),  # Assuming 102 classes
        ('output', nn.LogSoftmax(dim=1))
    ]))

    return model


model = buildModel(arch, hidden_units).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


# Training function
def trainIt(epoch=epochs):
    step = 0
    running_loss = 0
    print_every = 15
    for e in range(epoch):
        print(e)
        model.train()  # Set model to training mode
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
                model.eval()  # Set model to evaluation mode
                test_loss = 0
                accuracy = 0
                with torch.no_grad():  # No gradients needed for validation
                    for images, labels in validation_loader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()

                        ps = torch.exp(logps)
                        topps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch: {e + 1}/{epoch}.. "
                      f"Training loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {test_loss / len(validation_loader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(validation_loader):.3f}")
                running_loss = 0


# Train the model
trainIt()

# Save the checkpoint
print('Saving checkpoint...')
checkpoint = {
    'state_dict': model.state_dict(),
    'classifier': model.classifier,
    'class_to_idx': training_dataset.class_to_idx,
    "arch": arch
}

torch.save(checkpoint, f"{save_dir}/checkpoint_{arch}.pth")
print('Checkpoint saved.')
