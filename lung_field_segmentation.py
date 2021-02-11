import torch
import torchvision
import numpy as np
import warnings
from unet import PretrainedUNet
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm

warnings.filterwarnings('ignore')


# Load the model
unet = PretrainedUNet(
    in_channels=1,
    out_channels=2,
    batch_norm=True,
    upscale_mode="bilinear"
)

unet.load_state_dict(torch.load('data/lung_seg_net.pt'))
unet.eval()

# This step is important for the segment quality
img_resize = torchvision.transforms.Resize(512)

# Load the names of all the sites
img = Image.open('image name')  # Change this to other load method if you use dicom input
img.load()

# pre-process the input image
img_rs = img_resize(img)
img_ori = np.asarray(img_rs, dtype='float')
img_norm = (img_ori - np.min(img_ori)) / np.ptp(img_ori)
img_norm = img_norm - 0.5
if len(img_norm.shape) == 2:
    img_norm = np.repeat(np.expand_dims(img_norm, axis=2), 3, axis=2)
img_tensor = torch.tensor(img_norm)
img_tensor = img_tensor.transpose(2, 0).transpose(2, 1)
img_tensor = img_tensor.type(torch.FloatTensor)

# segment the lung field
out = unet(img_tensor[0, :, :].unsqueeze(0).unsqueeze(0))
softmax = torch.nn.functional.log_softmax(out, dim=1)
out = torch.argmax(softmax, dim=1)
out = out[0]

# save the output of segmentation in the scale of 256 * 256
plt.imsave('img_seg.jpg', out[1::2, 1::2], cmap=cm.gray)


