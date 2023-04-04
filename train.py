import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.optim as optim

from functools import partial
from argparse import ArgumentParser
from unet import double_unet
from unet.unet import UNet2D
from unet.MGunet import MGUNet2D
from unet.model import Model
from unet.utils import MetricList
from unet.metrics import jaccard_index, f1_score,LogNLLLoss#, DiceBCELoss
from unet.dataset import JointTransform2D, ImageToImage2D, Image2D
#import pydevd_pycharm
#pydevd_pycharm.settrace('132.69.238.212 ', port=$SERVER_PORT, stdoutToServer=True, stderrToServer=True)

parser = ArgumentParser()
parser.add_argument('--model_type', required=True, type=str, default="unet")
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
#parser.add_argument('--predict_dataset', type=str)
parser.add_argument('--checkpoint_path', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--in_channels', default=4, type=int) #4 for neuron segmentation, 3 for carvana
parser.add_argument('--out_channels', default=2, type=int)
parser.add_argument('--depth', default=5, type=int)
parser.add_argument('--width', default=16, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--save_freq', default=0, type=int)
parser.add_argument('--save_model', default=0, type=int)
parser.add_argument('--model_name', type=str, default='model_test')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--crop', type=int, default=None)
args = parser.parse_args()

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
model = None
if args.model_type == "unet":
    conv_depths = [int(args.width*(2**k)) for k in range(args.depth)]
    model = UNet2D(args.in_channels, args.out_channels, conv_depths)

elif args.model_type == "double_unet":
    model = double_unet.build_doubleunet()

else:
    hyper_depth = 4
    conv_depths = [[int(args.width*(2**k)) for k in range(args.depth)] for i in range(hyper_depth)]
    mid_out_channels = [16 for i in range(hyper_depth-1)]
    model = MGUNet2D(args.in_channels, args.out_channels, mid_out_channels=mid_out_channels, conv_depths=conv_depths)
loss = LogNLLLoss() #DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

results_folder = os.path.join(args.checkpoint_path, args.model_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

metric_list = MetricList({'jaccard': partial(jaccard_index),
                          'f1': partial(f1_score)})

model = Model(model, loss, optimizer, results_folder, device=args.device)
model.fit_dataset(train_dataset, n_epochs=args.epochs, n_batch=args.batch_size,
                  shuffle=True, val_dataset=val_dataset, save_freq=args.save_freq,
                  save_model=args.save_model, predict_dataset=predict_dataset,
                  metric_list=metric_list, verbose=True)
