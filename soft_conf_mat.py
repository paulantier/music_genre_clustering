import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
import csv
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

# Normalize the confusion matrix
def normalize_confusion_matrix(matrix):
    return matrix / matrix.sum(axis=1, keepdims=True)

# Visualize with a heatmap
def save_heatmap(matrix, class_labels, filename):
    plt.figure(figsize=(16, 16))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=class_labels, yticklabels=class_labels, cmap='viridis')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Soft Confusion Matrix')
    plt.savefig(filename)
    plt.close() 

# Path to the CSV file
csv_file = "dataset.csv"
len_labels=615

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


def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', None)
    
    return epoch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def soft_confusion_matrix(y_true, y_pred):
    """
    Compute the soft confusion matrix.
    
    :param y_true: Array of true soft labels, shape (n_samples, n_classes)
    :param y_pred: Array of predicted soft labels, shape (n_samples, n_classes)
    :return: Soft confusion matrix, shape (n_classes, n_classes)
    """
    n_classes = y_true.shape[1]
    matrix = np.zeros((n_classes, n_classes))
    
    for yt, yp in zip(y_true, y_pred):
        matrix += np.outer(yt, yp)
    
    return matrix

all_classes = []
with open("genre_frequencies.txt", 'r') as f:
    for line in f:
        class_name, _ = line.split(':')
        all_classes.append(class_name.strip())



image_folder = "MELs"


train_image_names, test_image_names, train_soft_labels, test_soft_labels = train_test_split(
    image_names, soft_labels, test_size=0.2, random_state=1
)

# Create the train and test datasets
train_dataset = ImageDataset(image_folder, train_image_names, train_soft_labels, transform=transform)
test_dataset = ImageDataset(image_folder, test_image_names, test_soft_labels, transform=transform)


batch_size=16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
            nn.Linear(128,33),
        )

    def forward(self, x):
        return self.classifier(x)


model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = MyResModel()
model = model.to(device)

num_epochs = 50
load_model=True

if load_model:
    checkpoint_path = 'checkpoints_mel_sp/checkpoint_epoch_71.pth' 
    start_epoch = load_checkpoint(checkpoint_path, model)
else:
    start_epoch=0

model.eval()

criterion = nn.CrossEntropyLoss()
running_loss=0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        probabilities = torch.softmax(outputs, dim=1)
        # Append batch predictions and labels to lists
        all_preds.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
y_pred = np.array(all_preds)
y_true = np.array(all_labels)

class_labels = all_classes
#cm = confusion_matrix(all_labels, all_preds)
#plot_confusion_matrix(cm, classes=test_dataset.classes, epoch=epoch+1)

print(f'Loss : {running_loss / len(dataloader):.4f}') # - Test Accuracy: {100 * correct / total:.2f}%')

num_selected_classes = 40
selected_classes = np.random.choice(len_labels, num_selected_classes, replace=False)

def filter_classes(y, selected_classes):
    return y[:, selected_classes]

filtered_y_true = filter_classes(y_true, selected_classes)
filtered_y_pred = filter_classes(y_pred, selected_classes)

# Compute and save the soft confusion matrix for the filtered classes
filtered_conf_matrix = soft_confusion_matrix(filtered_y_true, filtered_y_pred)
filtered_normalized_matrix = normalize_confusion_matrix(filtered_conf_matrix)
selected_class_labels = [class_labels[i] for i in selected_classes]

save_heatmap(filtered_normalized_matrix, selected_class_labels, 'soft_conf_mat_subset.png')