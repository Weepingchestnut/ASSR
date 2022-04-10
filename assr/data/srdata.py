import os
import pickle
import random

import imageio
import torch.utils.data as data
from skimage.transform import resize

from assr.data import common
from assr.utils.utility import calc_psnr_numpy


class SRData(data.Dataset):
    def __init__(self, cfg, name='', train=True, benchmark=False):
        self.cfg = cfg
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = False
        self.scale = cfg.DATASET.DATA_SCALE
        self.idx_scale = 0

        self._set_filesystem(cfg.DATASET.DATA_DIR)
        if cfg.DATASET.DATA_EXT.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')  # '/home/lzk/workspace/rcan-it/datasets/DIV2K/bin'
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()     # get hr, lr path list
        if cfg.DATASET.DATA_EXT.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            # list_hr, list_lr = self._scan()
            self.images_hr = self._check_and_load(
                cfg.DATASET.DATA_EXT, list_hr, self._name_hrbin()
            )
            self.images_lr = [
                self._check_and_load(cfg.DATASET.DATA_EXT,
                                     l, self._name_lrbin(s))
                for s, l in zip(self.scale, list_lr)    # s: 4  l:
            ]
        else:
            if cfg.DATASET.DATA_EXT.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = list_hr, list_lr
            elif cfg.DATASET.DATA_EXT.find('sep') >= 0:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                for s in self.scale:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{:.2f}'.format(s)
                        ),
                        exist_ok=True
                    )

                self.images_hr, self.images_lr = [], [[] for _ in self.scale]
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(
                        cfg.DATASET.DATA_EXT, [h], b, verbose=True, load=False
                    )

                for i, ll in enumerate(list_lr):
                    for l in ll:
                        b = l.replace(self.apath, path_bin)
                        b = b.replace(self.ext[1], '.pt')
                        self.images_lr[i].append(b)
                        self._check_and_load(
                            cfg.DATASET.DATA_EXT, [l], b,  verbose=True, load=False
                        )

        if train:
            self.n_train_samples = cfg.SOLVER.ITERATION_TOTAL * cfg.SOLVER.SAMPLES_PER_BATCH    # 200000 * batch size 16
            n_patches = cfg.SOLVER.SAMPLES_PER_BATCH * cfg.SOLVER.TEST_EVERY    # batch size 16 * test every 1000
            n_images = len(cfg.DATASET.DATA_TRAIN) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def _set_filesystem(self, dir_data):    # '/home/lzk/workspace/rcan-it/datasets'
        self.apath = os.path.join(dir_data, self.name)          # '.../rcan-it/datasets/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'HR')            # '.../rcan-it/datasets/DIV2K/HR'
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')    # '.../rcan-it/datasets/DIV2K/LR_bicubic'
        if self.input_large:
            self.dir_lr += 'L'
        self.ext = ('.png', '.png')     # hr img and lr img ext, such as hr:0001.png, lr:0001x4.png

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        for i in range(self.begin, self.end + 1):
            filename = '{:0>4}'.format(i)   # '0001'
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext[0]))   # '.../rcan-it/datasets/DIV2K/DIV2K_train_HR/0001.png'
            for si, s in enumerate(self.scale):     # s: 4    si: 0
                list_lr[si].append(os.path.join(
                    self.dir_lr,    # 先执行子类div2k的_set_filesystem ==> self.dir_lr '.../rcan-it/datasets/DIV2K/DIV2K_train_LR_bicubic'
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext[1])
                ))  # [['.../rcan-it/datasets/DIV2K/DIV2K_train_LR_bicubic/X4/0001x4.png']]

        return list_hr, list_lr

    def _check_and_load(self, ext, l, f, verbose=True, load=True):  # ext: 'bin', l: list of img path, f: '.../rcan-it/datasets/DIV2K/bin/train_bin_HR.pt'
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose:
                    print('Loading {}...'.format(f))    # Loading /home/lzk/workspace/rcan-it/datasets/DIV2K/bin/train_bin_HR.pt...
                with open(f, 'rb') as _f:
                    ret = pickle.load(_f)   # too slow
                return ret[:len(l)]     # 读取指定图像数
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            if ext.find('bin') >= 0:
                print('Bin pt file with name and image')
                b = [{
                    'name': os.path.splitext(os.path.basename(_l))[0],
                    'image': imageio.imread(_l)
                } for _l in l]
                with open(f, 'wb') as _f:
                    pickle.dump(b, _f)

                return b
            else:
                print('Direct pt file without name or image')
                b = imageio.imread(l[0])
                with open(f, 'wb') as _f:
                    pickle.dump(b, _f)

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )   # '.../rcan-it/datasets/DIV2K/bin/train_bin_HR.pt'

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.split, scale)
        )   # '.../rcan-it/datasets/DIV2K/train_bin_LR_X4.pt'

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)   # tuple(lr_patch, hr_patch)
        pair = common.set_channel(*pair, n_channels=self.cfg.DATASET.CHANNELS)
        pair_t = common.np2Tensor(*pair, rgb_range=self.cfg.DATASET.RGB_RANGE)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return self.n_train_samples
        else:
            return len(self.images_hr)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]      # get one hr_img from total 800 hr_imgs e.g. dict{'name':'0084','image':Array(1368, 2040, 3)}
        f_lr = self.images_lr[self.idx_scale][idx]      # get lr_img for corresponding hr_img e.g. dict{'name':'0084x4','image':Array(342, 510, 3)}

        if self.cfg.DATASET.DATA_EXT.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))  # os.path.basename() will return the last name of path
            if self.cfg.DATASET.DATA_EXT == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
            elif self.cfg.DATASET.DATA_EXT.find('sep') >= 0:
                with open(f_hr, 'rb') as _f:
                    hr = pickle.load(_f)
                with open(f_lr, 'rb') as _f:
                    lr = pickle.load(_f)

        return lr, hr, filename

    def _get_index(self, idx):
        if not self.train:
            return idx

        idx = random.randrange(self.n_train_samples)
        return idx % len(self.images_hr)

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if not self.train:
            ih, iw = lr.shape[:2]
            hr_patch = hr[0:ih * scale, 0:iw * scale]
            lr_patch = lr
            return lr_patch, hr_patch

        # rejection sampling for training
        while True:
            lr_patch, hr_patch = common.get_patch(
                lr, hr, patch_size=self.cfg.DATASET.OUT_PATCH_SIZE,
                scale=scale, multi=(len(self.scale) > 1),
                input_large=self.input_large
            )

            rej_cfg = self.cfg.DATASET.REJECTION_SAMPLING
            if not rej_cfg.ENABLED:
                break

            bicub_sr = resize(lr_patch, hr_patch.shape, order=3,    # bicubic
                              preserve_range=True, anti_aliasing=False)
            bicub_psnr = calc_psnr_numpy(bicub_sr, hr_patch, scale,
                                         float(self.cfg.DATASET.RGB_RANGE))
            if bicub_psnr < rej_cfg.MAX_PSNR or random.random() < rej_cfg.PROB:
                break

        aug_cfg = self.cfg.AUGMENT
        if aug_cfg.ENABLED:
            lr_patch, hr_patch = common.augment(
                lr_patch, hr_patch, invert=aug_cfg.INVERT,
                c_shuffle=aug_cfg.CHANNEL_SHUFFLE)

        return lr_patch, hr_patch   # Array 0~255











