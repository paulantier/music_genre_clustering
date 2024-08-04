import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, leaves_list
from torchvision.models import resnet18, ResNet18_Weights
import csv
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


# Path to the CSV file
csv_file = "dataset_v2.csv"
len_labels=33

# Lists to store image names and soft labels
image_names = []
soft_labels = []

# Read the CSV file
with open(csv_file, 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    
    for row in csvreader:
        soft_labels.append([float(x) for x in row[:len_labels]])  # Soft labels are the res
        image_names.append(row[len_labels:])  # The image name is the last element

class ImageDataset(Dataset):
    def __init__(self, image_folder, image_names, soft_labels, transform=None):
        self.image_folder = image_folder
        self.image_names = image_names
        self.soft_labels = soft_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        if len(img_name)==1:
            img_name = img_name[0]
        img_path = os.path.join(self.image_folder, img_name + '.png')
        
        # Open image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Get corresponding soft labels
        labels = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        
        return image, labels

def save_checkpoint(epoch, model, optimizer, scheduler, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', None)
    
    return epoch, optimizer, scheduler

checkpoint_dir = 'checkpoints_mel_sp_v2'
os.makedirs(checkpoint_dir, exist_ok=True)

def plot_confusion_matrix(cm, classes, epoch):
    # Normalize the confusion matrix
    cm_normalized = 100.0 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Use hierarchical clustering to find the best order for the classes
    linkage_matrix = linkage(cm_normalized, method='ward')
    sorted_indices = leaves_list(linkage_matrix)
    
    # Reorder the confusion matrix and class labels
    cm_reordered = cm_normalized[sorted_indices, :][:, sorted_indices]
    classes_reordered = [classes[i] for i in sorted_indices]
    
    # Plot the reordered confusion matrix
    plt.figure(figsize=(40, 40))
    sns.heatmap(cm_reordered, annot=True, fmt=".1f", cmap="Blues", xticklabels=classes_reordered, yticklabels=classes_reordered)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix at Epoch {epoch}')
    plt.savefig(f'conf_mats/confusion_matrix_epoch_{epoch}.png')
    plt.close()

class PermuteColumnsRandom:
    def __init__(self, block_size, p, iter_max):
        self.block_size_min, self.block_size_max = block_size
        self.iteration_max = iter_max
        self.probability = p

    def __call__(self, image):

        if random.uniform(0,1)<self.probability:
            nb_iter = random.randint(0, self.iteration_max)
            for _ in range(nb_iter):
                width = image.shape[2]
                block_size = random.randint(self.block_size_max, self.block_size_max)  # Taille des blocs à permuter

                # Choix aléatoire de la position de la première slice
                start1 = random.randint(0, width - block_size)
                end1 = start1 + block_size

                # Choix aléatoire de la position de la deuxième slice, en s'assurant qu'elle ne chevauche pas la première slice
                start2 = random.randint(0, width - block_size)
                #while abs(start1 - start2) < block_size:
                #    start2 = random.randint(0, width - block_size)
                end2 = start2 + block_size

                # Permutation des colonnes
                image[:, :, start1:end1], image[:, :, start2:end2] = image[:, :, start2:end2].clone(), image[:, :, start1:end1].clone()
        return image

# Définir les transformations
transform = transforms.Compose([
    transforms.ColorJitter(brightness=.5, saturation=.5, contrast=0.5),
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.4)),
    transforms.ToTensor(),
    PermuteColumnsRandom(block_size=[10,80], iter_max=3, p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""
transform = transforms.Compose([
    #transforms.ColorJitter(brightness=.5, saturation=.5, contrast=0.5),
    transforms.ToTensor(),
    #PermuteColumnsRandom(block_size=[10,80], iter_max=3, p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""

image_folder = "MELs"


train_image_names, test_image_names, train_soft_labels, test_soft_labels = train_test_split(
    image_names, soft_labels, test_size=0.2, random_state=1
)

# Create the train and test datasets
train_dataset = ImageDataset(image_folder, train_image_names, train_soft_labels, transform=transform)
test_dataset = ImageDataset(image_folder, test_image_names, test_soft_labels, transform=test_transform)


batch_size=16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Instancier le modèle
class MyResModel(torch.nn.Module):
    def __init__(self):
        super(MyResModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128,len_labels),
        )

    def forward(self, x):
        return self.classifier(x)


model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = MyResModel()
model = model.to(device)

num_epochs = 100
load_model=True

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=30, T_mult=2)


if load_model:
    checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_17.pth' 
    start_epoch, optimizer, scheduler = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
else:
    start_epoch=0

# Entraînement

all_classes = []
with open("genre_frequencies.txt", 'r') as f:
    for line in f:
        class_name, _ = line.split(':')
        all_classes.append(class_name.strip())

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    nb_elements=0
    train_loader_tqdm = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        nb_elements += labels.shape[0]
        train_loader_tqdm.set_postfix(loss=running_loss / (nb_elements/batch_size))
    
    avg_train_loss = running_loss / len(train_loader)
    scheduler.step(epoch + running_loss / len(train_loader))

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    save_checkpoint(epoch+1, model, optimizer, scheduler, avg_train_loss, checkpoint_path)

    
    if (epoch+1)%4==0:
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        #cm = confusion_matrix(all_labels, all_preds)
        #plot_confusion_matrix(cm, classes=test_dataset.classes, epoch=epoch+1)

        print(f'Loss : {running_loss / len(test_loader):.4f}') # - Test Accuracy: {100 * correct / total:.2f}%')
