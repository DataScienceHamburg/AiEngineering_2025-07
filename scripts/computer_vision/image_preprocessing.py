#%% packages
import torch
from torchvision import transforms
from PIL import Image

#%% import image
img = Image.open("sample_image.jpg")
img

#%% preprocessing steps
preprocess_steps = transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=1),
    transforms.RandomRotation(degrees=60),
    transforms.CenterCrop(size=300),
    transforms.Resize(size=(300, 300)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
preprocess_steps(img)