import os
import urllib.request
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "model/saved_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1V0ghKNZjk_6Bjox7Qd9-N9Ncqft6zD_0"  # Direct download link

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

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

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        download_model()  # Automatically download if not present
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
