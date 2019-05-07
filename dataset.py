#!/usr/bin/env python
# coding:utf-8
import os
from common import config
import re
import random
import numpy as np
import pickle


class DataSet:
    def __init__(self, dataset_name, read_from_pkl=True):
        assert dataset_name in ['train', 'test']
        self.ds_name = dataset_name
        self.dataset = []
        self.batch_size = config.train_batch_size if self.ds_name == 'train' else config.test_batch_size
        with open(os.path.join(config.data_dir, 'cmn.txt'), 'r') as f:
            while True:
                f_ = f.readline()
                if f_:
                    indexs = [m.start() for m in re.finditer(re.compile(r'[\u4e00-\u9fa5]'), f_)]
                    self.dataset.append([f_[:indexs[0]].replace('\t', ''), f_[indexs[0]:].replace('\n', '')])
                else:
                    break
        random.shuffle(self.dataset)
        self.train, self.test = self.dataset[:config.train_size], self.dataset[config.train_size:]
        self.size = config.train_size if self.ds_name == 'train' else config.test_size
        if read_from_pkl:
            with open(os.path.join(config.data_dir, 'dictionary.pkl'), 'rb') as f:
                self.dict = pickle.load(f)
        else:
            self.map_dic()
            with open(os.path.join(config.data_dir, 'dictionary.pkl'), 'wb') as f:
                pickle.dump(self.dict, f)

    def map_dic(self):
        """construct dictionary"""
        chineseset, englishset = set(), set()
        for bi_tuple in self.dataset:
            for str_ in bi_tuple[0]:
                englishset.add(str_)
        englishdict = dict(zip(englishset, range(len(englishset))))
        for bi_tuple in self.dataset:
            for str_ in bi_tuple[1]:
                chineseset.add(str_)
        chinesedict = dict(zip(chineseset, range(len(chineseset))))
        self.dict = dict(englishdict, **chinesedict)

    @staticmethod
    def raw_data_generator(data, dic):
        X, y = [], []
        for idx, sentence in enumerate(data):
            sentence0, sentence1 = sentence[0], sentence[1]
            X.append(np.array([dic[word] for word in sentence0]))
            y.append(np.array([dic[word] for word in sentence1]))
        return np.array(X), np.array(y)

    def one_epoch_generator(self):
        """生成一个epoch"""
        idx = list(range(self.size))
        if self.ds_name == "train":
            X, y = self.raw_data_generator(self.train, self.dict)
            np.random.shuffle(idx)
            start = 0
            while start < self.size:
                end = start + self.batch_size
                yield X[idx[start:end]], y[idx[start:end]]
                start = end
        else:
            X, y = self.raw_data_generator(self.test, self.dict)
            start = 0
            while start < self.size:
                end = start + self.batch_size
                yield X[idx[start:end]], y[idx[start:end]]
                start = end


if __name__ == '__main__':
    d = DataSet('train', read_from_pkl=True)
    for i in d.one_epoch_generator():
        print(i)
