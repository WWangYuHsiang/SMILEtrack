# import
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml

cfg_path = r"cfg/deploy/yolov7-PRB-e6e.yaml"
ckpt_path = r"yolov7-PRB-e6e_training.pt"  # checkpoints of yolov7-prb with IDetect layer
save_path = r"yolov7-PRB-e6e.pt"  # output file path

device = select_device('5', batch_size=1)
# model trained by cfg/training/*.yaml
ckpt = torch.load(ckpt_path, map_location=device)
# reparameterized model in cfg/deploy/*.yaml
model = Model(cfg_path, ch=3, nc=80).to(device)

with open(cfg_path) as f:
    yml = yaml.load(f, Loader=yaml.SafeLoader)
anchors = len(yml['anchors'])

# copy intersect weights
state_dict = ckpt['model'].float().state_dict()
exclude = []
intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
model.load_state_dict(intersect_state_dict, strict=False)
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

idx = 168
idx2 = 172

# copy weights of lead head
model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

# reparametrized YOLOR
for i in range((model.nc+5)*anchors):
    model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
    model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
    model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
    model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()

# model to be saved
ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
        'optimizer': None,
        'training_results': None,
        'epoch': -1}

# save reparameterized model
torch.save(ckpt, save_path)