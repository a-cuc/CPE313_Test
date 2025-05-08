import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
import cv2
from PIL import Image
import numpy as np


torch.classes.__path__ = []

class CustomFixedFeatureModel_Hyper_Result(nn.Module):
    def __init__(self, base_model, lstm_h=32, n_layers=2):
        super(CustomFixedFeatureModel_Hyper_Result, self).__init__()
        self.base_model = base_model
        self.lstm_layer_1 = nn.LSTM(input_size=1280, hidden_size=lstm_h, num_layers=n_layers, batch_first=True)
        self.lstm_layer_2 = nn.LSTM(input_size=lstm_h, hidden_size=16, num_layers=n_layers, batch_first=True)
        self.final_classifier = nn.Linear(16, 1) 

    def forward(self, x):
        x = self.base_model(x)
        x = x.unsqueeze(1) # Adding a sequence dimension for LSTM input
        output, (h_n, c_n) = self.lstm_layer_1(x)
        output, (h_n, c_n) = self.lstm_layer_2(output)
        x = self.final_classifier(h_n[-1])
        return x.squeeze(-1)
    
# Create an instance of the custom model
model = models.efficientnet_v2_s(weights='DEFAULT')
model.classifier = nn.Identity()  # Remove the final classification layer
model = CustomFixedFeatureModel_Hyper_Result(model)
model.load_state_dict(torch.load('model2_ffe7_weights.pth', weights_only=True))

st.write("""
# Drowsiness Detection System by Angelo"""
)
file=st.file_uploader("Choose photo from computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    model.eval()
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),  # Convert to float
    ])
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)
    predicted_classes = (prediction > 0).float().numpy()
    return int(predicted_classes[0])

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Drowsy', 'Normal']
    string="OUTPUT : "+ class_names[prediction]
    st.success(string)
