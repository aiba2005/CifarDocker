import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import io
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pydantic import BaseModel

classes = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

class CifarClassifaction(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),#32x32
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),#16x16
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),#8x8
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    self.second = nn.Sequential(
        nn.Flatten(),#4x4
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

  def forward(self, image):
    image = self.first(image)
    image = self.second(image)
    return image

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CifarClassifaction()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

cifar_app = FastAPI()


@cifar_app.post('/predict/')
async def check_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='File not found')

        img = Image.open(io.BytesIO(data))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            result = pred.argmax(dim=1).item()
        return {'class': classes[result]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(cifar_app, host='127.0.0.1', port=8000)

