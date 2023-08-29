# Here begins the code for the neural network model

# Load packages
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import time
import os
import tqdm
import PIL.Image as Image
from IPython.display import display

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.get_device_name(device))

## Load and transform the data

# Transform the data and labels here
# The 224x224 images are processed with random horizontal flip, random rotation and normalization

dataset_dir = "stanford_car_dataset/car_data/car_data/"

# data transformation, you can try different transformation/ data augmentation here
# note: no data augmentation for test data

width, height = 224, 224
train_tfms = transforms.Compose([transforms.Resize((width, height)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # mean and std from imagenet dataset
test_tfms = transforms.Compose([transforms.Resize((width, height)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(15),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# create datasets
dataset = torchvision.datasets.ImageFolder(root=dataset_dir + "train", transform = train_tfms)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir + "test", transform = test_tfms)
testloader = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=False, num_workers=2)

# Train and Test the Model
def train_model(model, criterion, optimizer, scheduler, n_epochs=5):

    losses = []
    accuracies = []
    test_accuracies = []

    # set the model to train mode initially
    model.train()
    for epoch in tqdm.tqdm(range(n_epochs)):
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs and assign them to cuda
            inputs, labels = data
            #inputs = inputs.to(device).half() # half precision model to quickly check
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            # torch.cuda.amp.autocast() # for half precision model
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate loss & acc

            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_duration = time.time() - since
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 / 32 * running_correct / len(trainloader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        # switch the model to eval mode to evaluate on test data
        model.eval()
        test_acc = eval_model(model)
        test_accuracies.append(test_acc)

        # re-set the model to train mode after validating
        model.train()
        scheduler.step(test_acc)
        since = time.time()
    print('Finished Training')
    return model, losses, accuracies, test_accuracies

# Evaluate the Model on training data
def eval_model(model):
    correct = 0.0
    total = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            #images = images.to(device).half() # half precision model
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (
        test_acc))
    return test_acc

## Tuning the Model - AlexNet

# define parameters

NUM_CAR_CLASSES = 196

# use alexnet as the base model
model_ft_an = models.alexnet(pretrained=True)

# Freeze model parameters and define the FC layer to be attached to the model,
# loss function and the optimizer.

# put the model on the GPUs
for param in model_ft_an.parameters():
    param.require_grad = False

# replace the last fc layer with an untrained one (requires grad)

num_ftrs_an = model_ft_an.classifier[6].in_features
model_ft_an.classifier[6] = nn.Linear(num_ftrs_an, NUM_CAR_CLASSES)

model_ft_an = model_ft_an.to(device)

# half precision model
# model_ft_an = model_ft_an.half()

# for layer in model_ft_an.modules():
#     if isinstance(layer, nn.BatchNorm2d):
#         layer.float()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft_an.parameters(), lr=0.01, momentum=0.9)

lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

# model training
model_ft_an, training_losses, training_accs, test_accs = train_model(model_ft_an, criterion, optimizer, lrscheduler, n_epochs=20)

# plot the stats

f, axarr = plt.subplots(2,2, figsize = (12, 8))
axarr[0, 0].plot(training_losses)
axarr[0, 0].set_title("Training loss, AlexNet")
axarr[0, 1].plot(training_accs)
axarr[0, 1].set_title("Training acc, AlexNet")
axarr[1, 0].plot(test_accs)

axarr[1, 0].set_title("Test acc, AlexNet")

# Evaluate the model on single images

# tie the class indices to their names 

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
classes, c_to_idx = find_classes(dataset_dir + "train")

# test the model on random images

# switch the model to evaluation mode to make dropout and batch norm work in eval mode
model_ft_an.eval()

# transforms for the input image
loader = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
image = Image.open(dataset_dir+"test/Mercedes-Benz C-Class Sedan 2012/01977.jpg")
image = loader(image).float()
image = torch.autograd.Variable(image, requires_grad=True)
image = image.unsqueeze(0)
image = image.cuda()
output = model_ft_an(image)
conf, predicted = torch.max(output.data, 1)

# get the class name of the prediction
display(Image.open(dataset_dir+"test/Mercedes-Benz C-Class Sedan 2012/01977.jpg"))
print(classes[predicted.item()], "confidence: ", conf.item())

## Tuning the model - ResNet34

# define parameters

NUM_CAR_CLASSES = 196

# use resnet as the base model
model_ft_rn = models.resnet34(pretrained=True)

# Freeze model parameters and define the FC layer to be attached to the model,
# loss function and the optimizer.

# put the model on the GPUs
for param in model_ft_rn.parameters():
    param.require_grad = False

# replace the last fc layer with an untrained one (requires grad)

# resnet34
num_ftrs_rn = model_ft_rn.fc.in_features
model_ft_rn.fc = nn.Linear(num_ftrs_rn, NUM_CAR_CLASSES)

model_ft_rn = model_ft_rn.to(device)

# half precision model
# model_ft_rn = model_ft_rn.half()
# for layer in model_ft_rn.modules():
#     if isinstance(layer, nn.BatchNorm2d):
#         layer.float()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft_rn.parameters(), lr=0.01, momentum=0.9)

lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

# model training

model_ft_rn, training_losses, training_accs, test_accs = train_model(model_ft_rn, criterion, optimizer, lrscheduler, n_epochs=20)

# plot the stats

f, axarr = plt.subplots(2,2, figsize = (12, 8))
axarr[0, 0].plot(training_losses)
axarr[0, 0].set_title("Training loss, ResNet34")
axarr[0, 1].plot(training_accs)
axarr[0, 1].set_title("Training acc, ResNet34")
axarr[1, 0].plot(test_accs)

axarr[1, 0].set_title("Test acc, ResNet34")

# Evaluate the model on single images
# test the model on random images

# switch the model to evaluation mode to make dropout and batch norm work in eval mode
model_ft_rn.eval()

# transforms for the input image
loader = transforms.Compose([transforms.Resize((400, 400)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
image = Image.open(dataset_dir+"test/Mercedes-Benz C-Class Sedan 2012/01977.jpg")
image = loader(image).float()
image = torch.autograd.Variable(image, requires_grad=True)
image = image.unsqueeze(0)
image = image.cuda()
output = model_ft_rn(image)
conf, predicted = torch.max(output.data, 1)

# get the class name of the prediction
display(Image.open(dataset_dir+"test/Mercedes-Benz C-Class Sedan 2012/01977.jpg"))
print(classes[predicted.item()], "confidence: ", conf.item())

# Save and Load the Models
PATH_an = 'car_model_an.pth'
torch.save(model_ft_an.state_dict(), PATH_an)

PATH_rn = 'car_model_an.pth'
torch.save(model_ft_rn.state_dict(), PATH_rn)

model_loaded_an = torch.load(PATH_an)
model_loaded_rn = torch.load(PATH_rn)

