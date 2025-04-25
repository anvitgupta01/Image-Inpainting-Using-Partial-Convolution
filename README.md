This project implements image inpainting using Partial Convolutions, a technique introduced by NVIDIA researchers to address the challenge of reconstructing missing or corrupted regions in images. Unlike traditional methods that rely on filling missing areas with constant values, partial convolutions condition the convolution operation on valid pixels, leading to more realistic and coherent inpainting results.

## How to Use
1. Clone the github repository using the following command -

   ```
   git clone https://github.com/anvitgupta01/Image-Inpainting-Using-Partial-Convolution/
   ```

2. Load `model.pth` file in the same directory in which the `src` folder is present from https://drive.google.com/drive/folders/1R9J-UdGKu97kL7nExm6dR-a_hQSpXXB6

3. Run `run_inpainting.ipynb`.

## Features
1. **Irregular Masking** - Irregular shaped masking with varied shapes and sizes have been done to ensure model learns to handle wide variety of shapes and sizes and real world occlusion scenarios
2. **Use of Partial Convolution** - Partial Convolution is used so that result of convolution will only depend on valid pixels.
3. **Out Of Distribution Inference** - Model is trained for 1 epoch (1400 steps) and it shows impressive results on Out Of Distribution examples which are not present in the dataset, marking the zero-shot behavior.
