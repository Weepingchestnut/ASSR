from importlib import import_module

from torch.utils.data import dataloader, ConcatDataset

from assr.data.asdataloader import ASDataLoader


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train  # datasets: [<assr.data.div2k.DIV2K object at 0x7fe23e213a00>]

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'):
                d.set_scale(idx_scale)


class Data:
    def __init__(self, cfg):
        if cfg.DATASET.FINETUNE.ENABLED:
            data_train = cfg.DATASET.FINETUNE.DATA
            print("Using Dataset(s): " + ','.join(data_train) + " for fine-tuning\n")
        else:
            data_train = cfg.DATASET.DATA_TRAIN
            print("Using Dataset(s): " + ','.join(data_train) + " for training\n")

        self.loader_train = None
        if not cfg.SOLVER.TEST_ONLY:
            datasets = []
            for d in data_train:
                if "2K" in d:
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('assr.data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(cfg, name=d))

                if d in ['Set5', 'Set14C', 'B100', 'Urban100', 'Manga109']:
                    m = import_module('assr.data.benchmark')
                    datasets.append(getattr(m, 'Benchmark')(cfg, train=True, name=d))

            if data_train[0].find('AS') >= 0:
                self.loader_train = ASDataLoader(
                    scale=cfg.DATASET.DATA_SCALE,
                    dataset=datasets[0],
                    batch_size=cfg.SOLVER.SAMPLES_PER_BATCH,
                    shuffle=True,
                    pin_memory=bool(cfg.SYSTEM.NUM_GPU),
                    # distribute workers among multiple processes
                    num_workers=cfg.SYSTEM.NUM_CPU // cfg.SYSTEM.NUM_GPU
                )
            else:
                self.loader_train = dataloader.DataLoader(
                    MyConcatDataset(datasets),
                    batch_size=cfg.SOLVER.SAMPLES_PER_BATCH,
                    shuffle=True,
                    pin_memory=bool(cfg.SYSTEM.NUM_GPU),
                    # distribute workers among multiple processes
                    num_workers=cfg.SYSTEM.NUM_CPU // cfg.SYSTEM.NUM_GPU
                )

        self.loader_test = []
        datatest = []

        if cfg.SOLVER.TEST_EVERY and not cfg.SOLVER.TEST_ONLY:
            datatest = cfg.DATASET.DATA_VAL
        elif cfg.SOLVER.TEST_ONLY:
            datatest = cfg.DATASET.DATA_TEST

        for d in datatest:
            if d in ['Set5', 'Set14C', 'B100', 'Urban100', 'Manga109']:
                m = import_module('assr.data.benchmark')
                testset = getattr(m, 'Benchmark')(cfg, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('assr.data.' + module_name.lower())
                testset = getattr(m, module_name)(cfg, train=False, name=d)

            if data_train[0].find('AS') >= 0:
                self.loader_test.append(
                    ASDataLoader(
                        scale=cfg.DATASET.DATA_SCALE,
                        dataset=testset,
                        batch_size=1,
                        shuffle=False,
                        pin_memory=bool(cfg.SYSTEM.NUM_GPU),
                        num_workers=cfg.SYSTEM.NUM_CPU // cfg.SYSTEM.NUM_GPU))
            else:
                self.loader_test.append(
                    dataloader.DataLoader(
                        testset,
                        batch_size=1,
                        shuffle=False,
                        pin_memory=bool(cfg.SYSTEM.NUM_GPU),
                        num_workers=cfg.SYSTEM.NUM_CPU // cfg.SYSTEM.NUM_GPU))
