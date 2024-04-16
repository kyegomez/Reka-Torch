[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Reka Torch
Implementation of the model: "Reka Core, Flash, and Edge: A Series of Powerful Multimodal Language Models" in PyTorch. [PAPER LINK](https://publications.reka.ai/reka-core-tech-report.pdf)

## Install
`pip3 install -U reka-torch`

## Usage
```python
import torch  # Importing the torch library
from reka_torch.model import Reka  # Importing the Reka model from the reka_torch package

text = torch.randint(0, 10000, (2, 512))  # Generating a random tensor of shape (2, 512) with values between 0 and 10000

img = torch.randn(2, 3, 224, 224)  # Generating a random tensor of shape (2, 3, 224, 224) with values

audio = torch.randn(2, 1000)  # Generating a random tensor of shape (2, 1000) with values

video = torch.randn(2, 3, 16, 224, 224)  # Generating a random tensor of shape (2, 3, 16, 224, 224) with values

model = Reka(512)  # Creating an instance of the Reka model with input size 512

out = model(text, img, audio, video)  # Forward pass through the model with the input tensors

print(out.shape)  # Printing the shape of the output tensor

```

# License
MIT
