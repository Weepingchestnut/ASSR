import os
import pickle
import random

import imageio
from skimage.transform import resize

from assr.config import get_cfg_defaults
from assr.data import srdata, common, Data
from assr.utils.utility import calc_psnr_numpy


class ASDIV2K(srdata.SRData):
    def __init__(self, cfg, name='ASDIV2K', train=True, benchmark=False):
        data_range = cfg.DATASET.DATA_RANGE
        if train:
            data_range = data_range[0]
        else:
            if cfg.SOLVER.TEST_ONLY and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = data_range

        super().__init__(
            cfg, name=name, train=train, benchmark=benchmark
        )
        # print("as_path_bin:", as_path_bin)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        for i in range(self.begin, self.end + 1):
            filename = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext[0]))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{:.2f}/{}{}'.format(s, filename, self.ext[1])
                ))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        # super(ASDIV2K, self)._set_filesystem(dir_data)
        self.apath = os.path.join(dir_data, 'DIV2K')  # '.../rcan-it/datasets/DIV2K'
        if self.split == 'test':
            self.dir_hr = os.path.join(self.apath, 'DIV2K_valid_HR')  # '.../rcan-it/datasets/DIV2K/DIV2K_train_HR'
            self.dir_lr = os.path.join(self.apath,
                                       'LR_as_bicubic/DIV2K_valid_LR_bicubic_1-4')  # '.../ASSR/datasets/DIV2K/Scale-Arbitrary/DIV2K_train_LR_bicubic_1-4/LR_bicubic'
        else:
            self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')  # '.../rcan-it/datasets/DIV2K/DIV2K_train_HR'
            self.dir_lr = os.path.join(self.apath,
                                       'LR_as_bicubic/DIV2K_train_LR_bicubic_1-4')  # '.../ASSR/datasets/DIV2K/Scale-Arbitrary/DIV2K_train_LR_bicubic_1-4/LR_bicubic'
        if self.input_large:
            self.dir_lr += 'L'
        self.ext = ('.png', '.png')  # hr img and lr img ext, such as hr:0001.png, lr:0001x4.png

    def __getitem__(self, idx):
        # self.set_scale(self.rand_scale())
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)   # tuple(lr_patch, hr_patch)
        pair = common.set_channel(*pair, n_channels=self.cfg.DATASET.CHANNELS)
        pair_t = common.np2Tensor(*pair, rgb_range=self.cfg.DATASET.RGB_RANGE)

        return pair_t[0], pair_t[1], filename, self.scale

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

    def rand_scale(self):
        if len(self.scale) > 1 and self.train:
            idx_scale = random.randrange(0, len(self.scale))
        else:
            idx_scale = 0
        return idx_scale

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if not self.train:
            ih, iw = lr.shape[:2]
            hr_patch = hr[0:ih * scale, 0:iw * scale]
            lr_patch = lr
            return lr_patch, hr_patch

        # rejection sampling for training
        while True:
            lr_patch, hr_patch = common.get_aspatch(
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


if __name__ == '__main__':
    scale = [s / 10 for s in list(range(11, 41, 1))]
    print(scale)

    cfg = get_cfg_defaults()
    cfg.DATASET.DATA_EXT = 'sep'
    cfg.DATASET.DATA_TRAIN = ['ASDIV2K']
    cfg.DATASET.DATA_VAL = ['ASDIV2K']
    cfg.DATASET.DATA_RANGE = [[1, 800], [801, 805]]
    cfg.DATASET.DATA_SCALE = scale
    print(cfg)

    loader = Data(cfg)
    loader_train = iter(loader.loader_train)
    loader_test = loader.loader_test

    lr, hr, _, scale = next(loader_train)
    print(f"lr batch shape: {lr.size()}")
    print(f"hr batch shape: {hr.size()}")
    print("current scale:", scale)



    # for i in scale:
    #     filename = '{:0<4}'.format(i)
    #     print(filename)
    #
    # ext = ['.png', '.png']
    #
    # for i in range(1, 800 + 1):
    #     filename = '{:0>4}'.format(i)
    #     # list_hr.append(os.path.join(self.dir_hr, filename + self.ext[0]))
    #     for si, s in enumerate(scale):
    #         print(os.path.join(
    #             'datasets/DIV2K/Scale-Arbitrary/DIV2K_train_LR_bicubic_1-4/LR_bicubic',
    #             'X{:.2f}/{}{}'.format(s, filename, ext[1])
    #         ))

