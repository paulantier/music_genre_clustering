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


from torchvision.models import resnet18, ResNet18_Weights


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

checkpoint_dir = 'checkpoints_melspects'
os.makedirs(checkpoint_dir, exist_ok=True)

def plot_confusion_matrix(cm, classes, epoch):
    cm_normalized = 100.0 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix at Epoch {epoch}')
    plt.savefig(f'conf_mats/melspect/confusion_matrix_epoch_{epoch}.png')
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
test_transform = transforms.Compose([
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform = transforms.Compose([
    #transforms.Resize([224,224]),
    transforms.ColorJitter(brightness=.5, saturation=.5, contrast=0.5),
    transforms.ToTensor(),
    PermuteColumnsRandom(block_size=[10,80], iter_max=3, p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dir = 'MELSPECTs_train'
test_dir = 'MELSPECTs_test'

# Load datasets with respective transformations
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

batch_size = 32
# Créer des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Instancier le modèle
model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

num_epochs = 300
load_model=True

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=38, T_mult=2)


if load_model:
    checkpoint_path = 'checkpoints_melspects/checkpoint_epoch_8.pth'  # Example path, adjust accordingly
    start_epoch, optimizer, scheduler = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
else:
    start_epoch=0

# Entraînement

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

    if epoch%4==3:
        # Évaluation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, classes=test_dataset.classes, epoch=epoch+1)

        print(f'Loss : {running_loss / len(test_loader):.4f} - Test Accuracy: {100 * correct / total:.2f}%')
