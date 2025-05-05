import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import gdown

# Google Drive file ID and local path
MODEL_ID = "1V0ghKNZjk_6Bjox7Qd9-N9Ncqft6zD_0"
MODEL_PATH = "model/saved_model.pth"

# Class labels
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

# Define the model architecture
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

# Predict function
trained_model = None

def predict(image_path):
    global trained_model

    if trained_model is None:
        download_model()
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        trained_model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
