import torch
import torch.nn as nn
import numpy as np
import warnings
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import hdf5storage as hds
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


# load the test image and mask
data_test = hds.loadmat('data/img_and_mask_for_test.mat')
img_critical = data_test['img_crit']
img_non_critical = data_test['img_non_crit']
mask_critical = data_test['mask_crit']
mask_non_critical = data_test['mask_non_crit']

# the pre-trained EfficientNet model for features extraction
model_eff = EfficientNet.from_pretrained('efficientnet-b0')
model_eff.eval()
model_eff.cuda()

# the critical classification model
model_pred = NetSever()
model_pred.load_state_dict(torch.load('data/critical_classification_model.pt'))
model_pred.cuda().eval()

# predict the critical patient
pred_all_critcal = np.zeros((img_critical.shape[2], 1))
for ind_img in range(img_critical.shape[2]):
    img_one = img_critical[:, :, ind_img]
    mask_one = mask_critical[ind_img, :, :]
    img_masked = img_one * mask_one
    img_masked = np.expand_dims(img_masked, axis=0)
    img_masked = np.expand_dims(img_masked, axis=0)
    img_masked = np.repeat(img_masked, 3, axis=1)
    img_masked = torch.tensor(img_masked)

    # extract features from EfficientNet
    with torch.no_grad():
        feat_one = model_eff.extract_features(img_masked.type(torch.FloatTensor).cuda())

    # predict the severity score
    with torch.no_grad():
        _, output_one = model_pred(feat_one)

    severity_one = output_one[:, 1].detach().cpu().numpy().squeeze()

    if output_one[:, 1] >= output_one[:, 0]:
        print('This is a severe patient ')
        pred_all_critcal[ind_img, 0] = 1
    else:
        print('This is a non-severe patient ')

# predict the non_critical patient
pred_all_non_critcal = np.zeros((img_non_critical.shape[2], 1))
for ind_img in range(img_non_critical.shape[2]):
    img_one = img_non_critical[:, :, ind_img]
    mask_one = mask_non_critical[ind_img, :, :]
    img_masked = img_one * mask_one
    img_masked = np.expand_dims(img_masked, axis=0)
    img_masked = np.expand_dims(img_masked, axis=0)
    img_masked = np.repeat(img_masked, 3, axis=1)
    img_masked = torch.tensor(img_masked)

    # extract features from EfficientNet
    with torch.no_grad():
        feat_one = model_eff.extract_features(img_masked.type(torch.FloatTensor).cuda())

    # predict the severity score
    with torch.no_grad():
        _, output_one = model_pred(feat_one)

    severity_one = output_one[:, 1].detach().cpu().numpy().squeeze()

    if output_one[:, 1] >= output_one[:, 0]:
        print('This is a critical patient ')
        pred_all_non_critcal[ind_img, 0] = 1
    else:
        print('This is a non-critical patient ')