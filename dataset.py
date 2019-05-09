#!/usr/bin/env python
# coding:utf-8
import os
from common import Config
import re
import random
import numpy as np
import tensorflow as tf
import pickle
import nltk

class DataSet:
    def __init__(self, dataset_name, read_from_pkl=True):
        assert dataset_name in ['train', 'test']
        self.ds_name = dataset_name
        self.dataset = []
        self.batch_size = Config.batch_size
        with open(os.path.join(Config.data_dir, 'cmn.txt'), 'r') as f:
            while True:
                f_ = f.readline()
                if f_:
                    indexs = [m.start() for m in re.finditer(re.compile(r'[\u4e00-\u9fa5]'), f_)]
                    self.dataset.append([f_[:indexs[0]].replace('\t', ''), f_[indexs[0]:].replace('\n', '')])
                else:
                    break
        random.shuffle(self.dataset)
        for i, bi_tuple in enumerate(self.dataset):
            self.dataset[i][0] = nltk.word_tokenize(bi_tuple[0])
        self.train, self.test = self.dataset[:Config.train_size], self.dataset[Config.train_size:]
        self.size = Config.train_size if self.ds_name == 'train' else Config.test_size
        if read_from_pkl:
            with open(os.path.join(Config.data_dir, 'english_dictionary.pkl'), 'rb') as f:
                self.english_2_index_dict = pickle.load(f)
            with open(os.path.join(Config.data_dir, 'chinese_dictionary.pkl'), 'rb') as f:
                self.chinese_2_index_dict = pickle.load(f)
        else:
            self.map_dic()
            with open(os.path.join(Config.data_dir, 'chinese_dictionary.pkl'), 'wb') as f:
                pickle.dump(self.chinese_2_index_dict, f)
            with open(os.path.join(Config.data_dir, 'english_dictionary.pkl'), 'wb') as f:
                pickle.dump(self.english_2_index_dict, f)
        self.idx2englishdic = {idx: word for word, idx in self.english_2_index_dict.items()}
        self.idx2chinesedic = {idx: word for word, idx in self.chinese_2_index_dict.items()}

    def map_dic(self):
        """construct dictionary"""
        chineseset, englishset = {'<PAD>', '<UNK>', '<GO>', '<EOS>'}, {'<PAD>', '<UNK>', '<GO>', '<EOS>'}
        for bi_tuple in self.dataset:
            for str_ in bi_tuple[0]:
                englishset.add(str_)
        self.english_2_index_dict = dict(zip(englishset, range(len(englishset))))
        for bi_tuple in self.dataset:
            for str_ in bi_tuple[1]:
                chineseset.add(str_)
        self.chinese_2_index_dict = dict(zip(chineseset, range(len(chineseset))))

    def raw_data_generator(self, data):
        """
        输入 原始的二元组data(list)
        生成一个padding长度相同的数据集
        返回
        input 和 output
        """
        X, y = [], []
        for idx, sentence in enumerate(data):
            X.append([self.english_2_index_dict[word] for word in sentence[0]])
            y.append([self.chinese_2_index_dict[word] for word in sentence[1]])
        return self.padding_sentence(X), self.padding_sentence(y)

    @staticmethod
    def padding_sentence(lec):
        length = max([len(s) for s in lec])
        result = np.zeros((len(lec), length))
        for idx, sentence in enumerate(lec):
            result[idx, :len(lec[idx])] = lec[idx]
        return result

    def process_decoder_input(self, data, vocab_2_index):
        """
        补充<GO>，并移除最后一个字符
        """
        # cut掉最后一个字符
        ending = tf.strided_slice(data, [0, 0], [self.batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], vocab_2_index['<GO>']), ending], 1)
        return decoder_input

    def one_epoch_generator(self):
        """
        生成一个epoch
        """
        idx = list(range(self.size))
        if self.ds_name == "train":
            np.random.shuffle(idx)
            start = 0
            while start < self.size:
                end = start + self.batch_size
                pad_targets_batch, pad_sources_batch = self.raw_data_generator([self.train[i_] for i_ in idx[start:end]])
                # 记录每条记录的长度
                targets_lengths, source_lengths = [], []
                for target, source in [self.train[i_] for i_ in idx[start:end]]:
                    targets_lengths.append(len(target))
                    source_lengths.append(len(source))
                yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths
                start = end
        else:
            start = 0
            while start < self.size:
                end = start + self.batch_size
                pad_targets_batch, pad_sources_batch = self.raw_data_generator(
                    [self.test[i_] for i_ in idx[start:end]])
                # 记录每条记录的长度
                targets_lengths, source_lengths = [], []
                for target, source in [self.test[i_] for i_ in idx[start:end]]:
                    targets_lengths.append(len(target))
                    source_lengths.append(len(source))
                yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths
                start = end


if __name__ == '__main__':
    d = DataSet('train', False)
    for i in d.one_epoch_generator():
        print(i)
        break

