import os
from common import config
import re

class Dataset:
    def __init__(self, dataset_name='train'):
        assert dataset_name in ['train', 'test']
        self.ds_name = dataset_name
        self.size = config.train_size if self.ds_name == 'train' else config.test_size
        self.dataset, self.train, self.test = [], [], []

    def loaddata(self):
        file_name = os.path.join(config.data_dir, 'cmn.txt')
        pattern = re.compile(r'[\.\?\!\"]')
        with open(file_name) as f:
            while True:
                f_ = f.readline()
                if f_:
                    index = re.finditer(pattern, f_)
                    for i_ in index:
                        index_ = i_.span()
                    self.dataset.append((f_[:index_[-1]], f_[index_[-1]:]))
                else:
                    break

