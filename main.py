import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(">> using", device)

# hyperparams
n_epochs = 200
threshold = .8
batch_size = 10

print(">> loading data...")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# second, fetch raw data
data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# labeled training data - this never changes
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
X_train, y_train = next(iter(dataloaders['train']))
X_train = X_train.to(device)
y_train = y_train.to(device)
print("dataset", y_train)

# unlabeled training data
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size * 7, shuffle=True) for x in ['train', 'val']}
X_unlabeled, _ = next(iter(dataloaders['train']))
X_unlabeled = X_unlabeled.to(device)

# test data
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=250, shuffle=True) for x in ['train', 'val']}
X_test, y_test = next(iter(dataloaders['val']))
X_test = X_test.to(device)
y_test = y_test.to(device)


###########
print(">> loading model...")
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)  # pin the model on the proper device
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)


print(">> supervised learning...")


for epoch in range(n_epochs):
    
    model_ft.train()
    
    # step 1: train on labeled data, which is never updated
    optimizer_ft.zero_grad()
    output = model_ft(X_train)
    real_loss = criterion(output, y_train)
    real_loss.backward()
    optimizer_ft.step()
    
    # validate
    model_ft.eval()    
    correct = 0

    output = model_ft(X_test)
    probs = F.softmax(output, dim=1)
    preds = probs.argmax(dim=1, keepdim=True)
    correct += preds.eq(y_test.view_as(preds)).sum().item()

    if epoch % (n_epochs / 20) == 0:
        print("epoch:", epoch, "accuracy", correct / len(y_test), "loss", real_loss.item())
###########

print(">> loading model...")
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)  # pin the model on the proper device
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)


print(">> semisupervised learning...")

for epoch in range(n_epochs):
    
    model_ft.train()
    
    # step 1: train on labeled data, which is never updated
    optimizer_ft.zero_grad()
    output = model_ft(X_train)
    real_loss = criterion(output, y_train)
    
    # step 2: view probabilities on unlabeled data
    output = model_ft(X_unlabeled)
    probs = F.softmax(output, dim=1)
    preds = probs.argmax(dim=1, keepdim=True)

    # step 3: train against pseudo labels only if it's high confidence
    fake_loss = 0
    counter = 0
    for prob, pred, c in zip(probs, preds, X_unlabeled):
        if (prob[pred] > threshold):
            counter += 1
            
            # step 4: generate strong augmentation data only if we'll use it
            X_strong = transforms.RandomErasing(p=1, ratio=(1, 1), scale=(0.01, 0.01), value=.5)(c)

            # step 5: learn against pseudo labels
            output = model_ft(X_strong.unsqueeze(0))
            fake_loss += criterion(output, pred)
    if counter > 0:
        fake_loss = fake_loss / counter  # take average fake loss
            
    total_loss = real_loss + 2*fake_loss
    total_loss.backward()
    optimizer_ft.step()
    
        
    # validate
    model_ft.eval()    
    correct = 0

    output = model_ft(X_test)
    probs = F.softmax(output, dim=1)
    preds = probs.argmax(dim=1, keepdim=True)
    correct += preds.eq(y_test.view_as(preds)).sum().item()

    if epoch % (n_epochs / 20) == 0:
        print("epoch:", epoch, "accuracy", correct / len(y_test), "loss", total_loss.item())