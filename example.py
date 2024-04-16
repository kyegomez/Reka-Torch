import torch  # Importing the torch library
from reka_torch.model import Reka  # Importing the Reka model from the reka_torch package

text = torch.randint(0, 10000, (2, 512))  # Generating a random tensor of shape (2, 512) with values between 0 and 10000

img = torch.randn(2, 3, 224, 224)  # Generating a random tensor of shape (2, 3, 224, 224) with values from a normal distribution

audio = torch.randn(2, 1000)  # Generating a random tensor of shape (2, 1000) with values from a normal distribution

video = torch.randn(2, 3, 16, 224, 224)  # Generating a random tensor of shape (2, 3, 16, 224, 224) with values from a normal distribution

model = Reka(512)  # Creating an instance of the Reka model with input size 512

out = model(text, img, audio, video)  # Forward pass through the model with the input tensors

print(out.shape)  # Printing the shape of the output tensor
