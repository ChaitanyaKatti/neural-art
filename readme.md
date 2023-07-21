# Neural Style Transfer

<p align="center">
  <img src="https://github.com/ChaitanyaKatti/neural-art/assets/96473570/0c925a25-13d7-4763-bb4f-9e746a5d0cce" alt="Image Description" width="400" height="300" />
</p>


Welcome to our Neural Style Transfer GitHub repository! This project focuses on the implementation of Neural Style Transfer, a fascinating technique that combines the content of one image with the artistic style of another image using deep neural networks. Neural Style Transfer has gained significant attention for its ability to create captivating and artistic images.

## Table of Contents

1. [Introduction](#introduction)
2. [Neural Style Transfer Overview](#neural-style-transfer-overview)
3. [Methodology](#methodology)
4. [How to Use](#how-to-use)
5. [Examples](#examples)
6. [References](#references)

## Introduction

Neural Style Transfer is an exciting technique that merges the content of one image, such as a photograph, with the visual style of another image, like a famous painting. The result is a visually stunning synthesis, where the content of the input image is preserved, but the style is transformed into the artistic patterns of the style image.

## Neural Style Transfer Overview

The concept of Neural Style Transfer was first introduced in the paper "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. The paper showcases how to use Convolutional Neural Networks (CNNs) to achieve the style transfer process. You can access the paper [here](https://arxiv.org/abs/1508.06576).

## Methodology

The Neural Style Transfer technique involves the following key steps:

1. **Content Representation**: A pre-trained CNN is used to extract high-level features from the content image. The intermediate feature maps, usually from deeper layers of the network, capture the content information.

2. **Style Representation**: Similarly, the style image's features are extracted from the pre-trained CNN. The Gram matrix is computed from these feature maps, representing the style of the image.

3. **Loss Function**: To optimize the output image, a loss function is defined, which consists of two components: the content loss and the style loss. The content loss measures the difference between the content image and the generated image's feature maps. The style loss calculates the difference in the Gram matrices between the style image and the generated image.

4. **Optimization**: The optimization process aims to minimize the overall loss function by adjusting the generated image's pixel values. This process iteratively updates the generated image to merge the content and style effectively.

## How to Use

To run the Neural Style Transfer algorithm on your own images, follow these steps:

1. Clone the repository: `git clone https://github.com/ChaitanyaKatti/neural-art.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Prepare your content image and style image and place them in the `images` directory.
4. Run the style transfer script: `python main.py` of use the `main.ipynb`

## Examples

Here are some impressive examples of Neural Style Transfer outputs:

<p align="center">
  <img src="https://github.com/ChaitanyaKatti/neural-art/assets/96473570/2d16f692-0276-4664-ba71-559e1bf893d5" alt="Image Description" width="500" height="500" />
  
</p>

## References

1. [A Neural Algorithm of Artistic Style - Original Paper](https://arxiv.org/abs/1508.06576)
2. [Neural Style Transfer with TensorFlow Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)

For more details and visual examples, refer to the `images/generated` directory in the repository.

Enjoy creating your own artistic images with Neural Style Transfer! :art:
