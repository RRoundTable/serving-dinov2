import torch
import torch.nn as nn
import gradio as gr
import glob
from typing import List
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.decomposition import PCA
import sklearn
import numpy as np


# Constants
patch_h = 40
patch_w = 40

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# DINOV2
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

# Trasnforms
transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Empty Tenosr
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14)


# PCA
pca = PCA(n_components=3)

# Min-Max Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(clip=True)

def query_image(
    img1, img2, img3, img4,
    background_threshold,
    is_foreground_larger_than_threshold,
) -> List[np.ndarray]:

    # Transform
    imgs = [img1, img2, img3, img4]
    for i, img in enumerate(imgs):
        img = np.transpose(img, (2, 0, 1)) / 255
        imgs_tensor[i] = transform(torch.Tensor(img))

    # Get feature from patches
    with torch.no_grad():
        features_dict = model.forward_features(imgs_tensor)
        features = features_dict['x_prenorm'][:, 1:]

    features = features.reshape(4 * patch_h * patch_w, -1)
    # PCA Feature
    pca.fit(features)
    pca_features = pca.transform(features)
    scaler.fit(pca_features)
    pca_feature = scaler.transform(pca_features)

    # Foreground/Background
    if is_foreground_larger_than_threshold:
        pca_features_bg = pca_features[:, 0] < background_threshold
    else:
        pca_features_bg = pca_features[:, 0] > background_threshold
    pca_features_fg = ~pca_features_bg

    # PCA with only foreground
    pca.fit(features[pca_features_fg])
    pca_features_rem = pca.transform(features[pca_features_fg])

    # Min Max Normalization
    scaler.fit(pca_features_rem)
    pca_features_rem = scaler.transform(pca_features_rem)

    pca_features_rgb = np.zeros((4 * patch_h * patch_w, 3))
    pca_features_rgb[pca_features_bg] = 0
    pca_features_rgb[pca_features_fg] = pca_features_rem
    pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)

    return [pca_features_rgb[i] for i in range(4)]

description = """
DINOV2 PCA demo for <a href="https://arxiv.org/abs/2304.07193">DINOv2: Learning Robust Visual Features without Supervision(Figure 1)</a>

How to Use:

1. Enter 4 images that have clean background and similar object.
2. Edit threshold and checkbox to split background/foreground.

Method:
1. Compute the features of patches from 4 images. We can get a feature that have (4 * patch_w * patch_h, feature_dim) shape.
2. PCA the feature with 3 dims. After PCA, Min-Max normalization is performed.
3. Use first component to split foreground and background. (threshold and checkbox)
4. All the feature of patches included in the background are set to 0.
5. PCA is performed based on the remaining features. Afer PCA, Min-Max normalization is performed.
6. Visualize
"""
demo = gr.Interface(
    query_image,
    inputs=[gr.Image(), gr.Image(), gr.Image(), gr.Image(), gr.Slider(-1, 1, value=0.1), gr.Checkbox(label="foreground is larger than threshold", value=True) ],
    outputs=[gr.Image(), gr.Image(), gr.Image(), gr.Image()],
    title="DINOV2 PCA",
    description=description,
    examples=[
        ["assets/1.png", "assets/2.png","assets/3.png","assets/4.png", 0.9, True],
        ["assets/5.png", "assets/6.png","assets/7.png","assets/8.png", 0.6, True],
        ["assets/9.png", "assets/10.png","assets/11.png","assets/12.png", 0.6, True],
    ]
)
demo.launch()
