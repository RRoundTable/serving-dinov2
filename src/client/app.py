import gradio as gr
from typing import List
from sklearn.decomposition import PCA
import numpy as np
import os
import cv2
import tritonclient.grpc as grpcclient
from sklearn.preprocessing import MinMaxScaler


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

# Constants
patch_h = 40
patch_w = 40
patch_size = 14
INFERENCE_URL = os.environ.get("INFERENCE_URL", "localhost:20000/infer")

# triton client
triton_client = grpcclient.InferenceServerClient(url=INFERENCE_URL)

# PCA
pca = PCA(n_components=3)

# Min-Max Scaler
scaler = MinMaxScaler(clip=True)


def transforms(image: np.ndarray) -> np.ndarray:
    resized_image = cv2.resize(
        image,
        (patch_h * patch_size, patch_w * patch_size)
    ) / 255
    normalized_image = (
        resized_image - np.array([0.485, 0.456, 0.406])
    ) / np.array([0.229, 0.224, 0.225])
    normalized_image = normalized_image.transpose(2, 0, 1)

    return np.expand_dims(normalized_image, 0).astype("float32")


def request_inference(images: List[np.ndarray]) -> np.ndarray:
    inputs = []
    for img in images:
        inputs.append(transforms(img))
    inputs = np.concatenate(inputs)
    triton_inputs = [grpcclient.InferInput("input", inputs.shape, "FP32")]
    triton_inputs[0].set_data_from_numpy(inputs)
    triton_outputs = [grpcclient.InferRequestedOutput("output")]

    params = {
        "model_name": "dinov2_vitl14",
        "model_version": "1",
    }

    result = triton_client.infer(
        inputs=triton_inputs,
        outputs=triton_outputs,
        **params,
    )
    output = result.as_numpy("output")

    return output


def query_image(
    img1, img2, img3, img4,
    background_threshold,
    is_foreground_larger_than_threshold,
) -> List[np.ndarray]:

    # Transform
    imgs = [img1, img2, img3, img4]
    patch_features = request_inference(imgs)

    patch_features = patch_features.reshape(len(imgs) * patch_h * patch_w, -1)

    # PCA Feature
    pca.fit(patch_features)
    pca_features = pca.transform(patch_features)

    # Scaling
    scaler.fit(pca_features)
    pca_features = scaler.transform(pca_features)

    # Foreground/Background
    if is_foreground_larger_than_threshold:
        pca_features_bg = pca_features[:, 0] < background_threshold
    else:
        pca_features_bg = pca_features[:, 0] > background_threshold
    pca_features_fg = ~pca_features_bg

    # PCA with only foreground
    pca.fit(patch_features[pca_features_fg])
    pca_features_rem = pca.transform(patch_features[pca_features_fg])

    # Min Max Normalization
    scaler.fit(pca_features_rem)
    pca_features_rem = scaler.transform(pca_features_rem)

    pca_features_rgb = np.zeros((4 * patch_h * patch_w, 3))
    pca_features_rgb[pca_features_bg] = 0
    pca_features_rgb[pca_features_fg] = pca_features_rem
    pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)

    return [pca_features_rgb[i] for i in range(4)]


if __name__ == "__main__":
    demo = gr.Interface(
        query_image,
        inputs=[
            gr.Image(),
            gr.Image(),
            gr.Image(),
            gr.Image(),
            gr.Slider(-1, 1, value=0.1),
            gr.Checkbox(label="foreground is larger than threshold", value=True),
        ],
        outputs=[gr.Image(), gr.Image(), gr.Image(), gr.Image()],
        title="DINOV2 PCA",
        description=description,
        examples=[
            ["assets/1.png", "assets/2.png", "assets/3.png", "assets/4.png", 0.4, True],
            ["assets/5.png", "assets/6.png", "assets/7.png", "assets/8.png", 0.4, True],
            ["assets/9.png", "assets/10.png", "assets/11.png", "assets/12.png", 0.4, True],
        ]
    )
    demo.launch(server_name="0.0.0.0")
