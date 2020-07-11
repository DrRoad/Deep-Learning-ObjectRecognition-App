#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import torch

from torchvision import models
from PIL import Image
import lib.tool as tool

st.title("Object Recognition App using Deep Learning")


# set device to CUDA if available, else to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


#--------------------------------------------------------#

# Load an input image for testing

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    img = (Image.open(img_file_buffer))


else:
    demo_image = "data/bird.jpeg"
    img = (Image.open(demo_image))


batch_t = tool.preprocess_ImageNet(img).to(device)

#--------------------------------------------------------#

# Load resnet architecture and download weights
model = models.resnet101(pretrained=True)  # Trained on ImageNet
model.eval()
model.to(device)


#--------------------------------------------------------#

# Inference
output = model(batch_t)

#--------------------------------------------------------#

# Prediction
pred = tool.postprocess(output, topPred=5)

#--------------------------------------------------------#

# Plot
import plotly.express as px
fig = px.bar(x=pred['class'], y=pred['conf'])
st.image(img, use_column_width=True)
st.plotly_chart(fig)
