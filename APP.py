from flask import Flask, render_template, request
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition

# Define the Model class
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        # to reduce size of network and overfitting
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Define the validation_dataset class
class ValidationDataset(Dataset):
    def __init__(self, video_path, sequence_length=60, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        vidObj = cv2.VideoCapture(self.video_path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                faces = face_recognition.face_locations(image)
                if faces:  # Proceed only if faces are detected
                    top, right, bottom, left = faces[0]
                    image = image[top:bottom, left:right, :]
                    if self.transform:
                        image = self.transform(image)
                    frames.append(image)
                    if len(frames) == self.count:
                        break
        # Pad frames if there are not enough
        while len(frames) < self.count:
            frames.append(torch.zeros_like(frames[0]))
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

# Load the model
model = Model(2).cuda()
path_to_model = 'D:\DEEP FAKE COLOB\model_87_acc_20_frames_final_data.pt'
model.load_state_dict(torch.load(path_to_model))
model.eval()

def predict(model, img, path='./'):
    fmap, logits = model(img.to('cuda'))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = nn.Softmax(dim=1)(logits)
    confidence = logits[:, 1].item() * 100  # Confidence for class 1 (index 1)
    print('Confidence of prediction:', confidence)
    prediction_idx = torch.argmax(logits, dim=1).item()
    
    return [prediction_idx, confidence]

app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)

            # Define transformation for video frames
            im_size = 112
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            train_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((im_size, im_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

            # Create dataset
            video_dataset = ValidationDataset(video_path, sequence_length=20, transform=train_transforms)

            # Make prediction
            prediction = predict(model, video_dataset[0], './')
            if prediction[0] == 1:
                result = "REAL"
            else:
                result = "FAKE"

            return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = './uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
