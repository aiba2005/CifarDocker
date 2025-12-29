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

class ChecCifarVgg(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            conv_block(3, 64), conv_block(64, 64),
            nn.MaxPool2d(2, 2),

            conv_block(64, 128), conv_block(128, 128),
            nn.MaxPool2d(2, 2),

            conv_block(128, 256), conv_block(256, 256),
            conv_block(256, 256), conv_block(256, 256),
            nn.MaxPool2d(2, 2),

            conv_block(256, 512), conv_block(512, 512),
            conv_block(512, 512), conv_block(512, 512),
            nn.MaxPool2d(2, 2),

            conv_block(512, 512), conv_block(512, 512),
            conv_block(512, 512), conv_block(512, 512),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChecCifarVgg()
model.load_state_dict(torch.load('model_mc.pth', map_location=device))
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

