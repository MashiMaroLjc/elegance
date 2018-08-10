# coding:utf-8
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from torch.autograd import Variable
_mean = [0.5, 0.5, 0.5]
_std = [0.5, 0.5, 0.5]


def _default_loader(path):
    return Image.open(path).convert('RGB')

def read_img(path):
    img = _default_loader(path)
    return img

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(_mean, _std)])


def tensor_to_numpyuint8(ts):
    batch, c, h, w = ts.shape
    imgs = np.zeros(shape=(batch, h, w, c), dtype="float64")
    for i, t in enumerate(ts):
        try:
            t = t.cpu().data.numpy()
        except:
            t = t.cpu().numpy()
        for ch in range(c):
            t[ch] = t[ch] * _std[ch] + _mean[ch]
        t = np.transpose(t, [1, 2, 0])
        imgs[i] = t
    return np.clip(imgs * 255, 0, 255).astype("uint8")




def get_plugin_name(dir):
    plugins = []
    for d in os.listdir(dir):
        plugins.append(d)
    return plugins

def get_net_and_preprocess(plugin):
    """

    :param plugin:  str
    :return: g net
    :rtype: nn.Module
    """
    import torch
    net_path = "plugin.{}.net".format(plugin)
    weight_path = os.path.join("plugin", plugin, "G.pth")
    plug = __import__(net_path, fromlist=(plugin, "net"))
    generate = plug.GNet()
    generate.load_state_dict(torch.load(weight_path))
    pipeline = plug.Preprocess()
    return generate,pipeline

def predict_save(generate, path, savepath, max_size=1280, use_gpu=False, preprocess=None,combine=False):
    generate.eval()
    img = read_img(path)
    origin_img = img.copy()
    a, b = img.size
    origin_max = max([a,b])
    print("input size:", (a, b))
    if preprocess is not None:
        img = preprocess(img)
    max_ = max(img.size)
    max_ = max_size if max_ > max_size else max_
    is_big = (origin_max > max_size ) or (max_ > max_size)

    img = img.resize((max_, max_))
    img = data_transform(img)
    c, h, w = img.size()
    img = img.view(1, c, h, w)
    if use_gpu:
        img = img.cuda()
    img = Variable(img)
    fake_img = generate(img)
    fake_img = tensor_to_numpyuint8(fake_img)
    new_img = fake_img[0]
    img = Image.fromarray(new_img)
    if is_big:
        na = a
        nb = b
    else:
        na,nb =img.size
        if na > nb:
            nb = int(nb * (b/a))
        else:
            na = int(na * (a/b))
    print("output size:",(na,nb))
    if combine:
        origin_img = origin_img.resize((na, nb))
        img = img.resize((na, nb))
        img = np.concatenate((origin_img,img),axis=1)
        Image.fromarray(img).save(savepath)
    else:
        img = img.resize((na, nb))
        img.save(savepath)