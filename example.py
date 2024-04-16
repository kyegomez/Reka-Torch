import torch
from reka_torch.model import Reka

text = torch.randint(0, 10000, (2, 512))

img = torch.randn(2, 3, 224, 224)

audio = torch.randn(2, 1000)

video = torch.randn(2, 3, 16, 224, 224)

model = Reka(512)

out = model(text, img, audio, video)
print(out.shape)
