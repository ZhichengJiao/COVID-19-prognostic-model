import torch
import torch.nn as nn
import numpy as np
import warnings
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import hdf5storage as hds
import cv2
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# Define the severity prediction net
class NetSever(nn.Module):
    def __init__(self):
        super(NetSever, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, stride=1, bias=False)
        self.pool2 = nn.AvgPool2d(2)

        self.drop_fc = nn.Dropout(0.5)
        self.activ = nn.ReLU()
        self.activ_pred = nn.Sigmoid()
        self.surv_pred1 = nn.Linear(256, 32, bias=False)
        self.surv_pred2 = nn.Linear(32, 2, bias=False)

    def forward(self, x):
        x = self.activ(self.conv1(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x_ = self.drop_fc((x.view(x.shape[0], -1)))
        x_ = self.activ(self.surv_pred1(x_))
        x_ = self.activ_pred(self.surv_pred2(x_))
        x_ = F.softmax(x_, dim=1)
        return x, x_


def return_cam(feat_val, mask_res, model_pred_params):

    conv_1 = model_pred_params['conv1.weight'].detach().cpu().numpy()
    fc_1 = model_pred_params['surv_pred1.weight'].detach().cpu().numpy()
    fc_2 = model_pred_params['surv_pred2.weight'].detach().cpu().numpy()

    fc_2 = np.squeeze(fc_2[1, :])
    contr_1 = fc_2.dot(fc_1)
    contr_2 = contr_1.dot(np.sum(conv_1, axis=(2, 3)))
    feat_weighted = feat_val.detach().cpu().numpy() * np.expand_dims(contr_2, axis=(0, 2, 3))
    cam = np.sum(feat_weighted.squeeze(), axis=0)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_out = cv2.resize(cam_img, (512, 512))
    cam_out = cam_out * mask_res
    return cam_out


# Load the nrrd files of image and related lung mask
data_test = hds.loadmat('data/img_and_mask_for_vis_cam.mat')
# Load the Efficient model for feature representation
model_eff = EfficientNet.from_pretrained('efficientnet-b0')
model_eff.eval()

# Params for visualization
model_pred_params = torch.load('data/critical_classification_model.pt')
# Extract features of the masked image

img_one_ori = data_test['img_vis']
mask_one = data_test['mask_vis']
img_masked = img_one_ori * mask_one
img_masked = np.expand_dims(img_masked, axis=0)
img_masked = np.expand_dims(img_masked, axis=0)
img_masked = np.repeat(img_masked, 3, axis=1)
img_masked = torch.tensor(img_masked)
img_one = np.expand_dims(img_one_ori, axis=0)
img_one = np.expand_dims(img_one, axis=0)
img_one = np.repeat(img_one, 3, axis=1)
img_one = torch.tensor(img_one)
with torch.no_grad():
    feat_vis = model_eff.extract_features(torch.FloatTensor(img_one.float()))

cam_out = return_cam(feat_vis, mask_one, model_pred_params)
cam_out = np.squeeze(cam_out.astype('uint8'))
heatmap = cv2.applyColorMap(cam_out, cv2.COLORMAP_JET)
img_rgb = np.repeat(np.expand_dims(img_one_ori, axis=2), 3, axis=2)

# Generate the overlay image
map_overlap = heatmap * 0.3 + img_rgb * 0.5 * 255

# plot the original image
ax1 = plt.subplot(121)
ax1.set_title('CXR image')
plt.imshow(img_one_ori, cmap='gray')

# plot the overlay image
ax2 = plt.subplot(122)
ax2.set_title('Overlay attention')
plt.imshow(np.fliplr(map_overlap.reshape(-1, 3)).reshape(map_overlap.shape).astype('uint8'))

# show the results
plt.show()


