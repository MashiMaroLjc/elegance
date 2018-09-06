# coding:utf-8
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import glob
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
        if not (".py" in d or "__" in d):
            plugins.append(d)
    return plugins


def load_plugin(plugin_name, params):
    """

    :param plugin:  str
    :return: g net
    :rtype: nn.Module
    """
    import torch
    net_path = "plugin.{}.net".format(plugin_name)
    weight_path = os.path.join("plugin", plugin_name, "G.pth")
    plug = __import__(net_path, fromlist=(plugin_name, "net"))
    transformer = plug.Transfromer(params)
    generator = transformer.get_generator()

    generator.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
    preprocess = transformer.get_preprocess()
    postprocess = transformer.get_postprocess()
    return generator, preprocess, postprocess


def _forward(img, use_gpu, generate):
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
    return img


def _load_img(path, preprocess):
    img = read_img(path)
    origin_img = img.copy()
    a, b = img.size
    print("[INFO]Input size:", (a, b))
    if preprocess is not None:
        img = preprocess(img)
    print("[INFO]After proproces size:", img.size)
    return img, origin_img


def _save_img(img, origin_img, savepath, combine, postprocess):
    if postprocess is not None:
        img = postprocess(img)
    na, nb = img.size
    print("[INFO]After postproces size:", (na, nb))
    if combine:
        origin_img = origin_img.resize((na, nb), Image.BICUBIC)
        img = img.resize((na, nb), Image.BICUBIC)
        b = np.array(img).astype("float32")
        a = np.array(origin_img).astype("float32")
        r = b - a
        r *= 0.5
        r += 127.5
        r = r.astype("uint8")
        img = np.concatenate((origin_img, img, r), axis=1)
        Image.fromarray(img).save(savepath)
    else:
        img = img.resize((na, nb), Image.BICUBIC)
        img.save(savepath)


def predict_save(generate, path, savepath, split_size=np.inf, use_gpu=False,
                 preprocess=None, combine=False, postprocess=None):
    # img,origin_img = _load_img(path,preprocess)
    # img = _forward(img,use_gpu,generate)
    # print("Output size:", img.size)
    # _save_img(img,origin_img,savepath,combine,postprocess)
    _predict_spilt(generate, path, split_size, savepath, use_gpu, preprocess, combine, postprocess)


def predict_dir(generate, path, savepath, split_size=np.inf, use_gpu=False,
                preprocess=None, combine=False, postprocess=None):
    img_dir = list(glob.glob(path + "*.jpg")) + list(glob.glob(path + "*.png"))
    length = len(img_dir)
    for i in range(length):
        print("[INFO][{}/{}]".format(i, length))
        out_path = img_dir[i].replace(path, savepath)
        predict_save(generate, img_dir[i], out_path, split_size, use_gpu, preprocess, combine, postprocess)


def _predict_spilt(generate, path, split_size, savepath, use_gpu=False,
                   preprocess=None, combine=False, postprocess=None):
    # if max()
    img, origin_img = _load_img(path, preprocess)
    oa,ob = img.size
    if max(img.size) <= split_size:
        img = _forward(img, use_gpu, generate)
    else:
        img_buffer = []
        img_h, img_w = img.size
        # 确保可以整除
        img_h = img_h - img_h % split_size
        img_w = img_w - img_w % split_size
        img = img.resize((img_h, img_w))
        # 先完成列 后完成行
        for i in range(0, img_h, split_size):
            buff_ = []
            for j in range(0, img_w, split_size):
                print("[INFO]Split ({},{})/({},{})".format(i+split_size,
                                                           j+split_size,
                                                          img_h, img_w))
                crop = [i, j, i + split_size, j + split_size]
                img_crop = img.crop(crop)
                img_crop = _forward(img_crop, use_gpu, generate)
                buff_.append(img_crop)
            buff_ = np.concatenate(buff_, axis=0)
            img_buffer.append(buff_)
        img = np.concatenate(img_buffer, axis=1)
        img = Image.fromarray(img).resize((oa,ob),Image.BICUBIC)
    print("[INFO]Output size:", img.size)
    _save_img(img, origin_img, savepath, combine, postprocess)


def str2dict(s,eq="="):
    if s is None:
        return {}
    return dict([elem.split(eq) for elem in s])
