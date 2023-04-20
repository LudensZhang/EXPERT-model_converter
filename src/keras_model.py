import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Activation
import numpy as np
import pandas as pd
from collections import OrderedDict
from tensorflow.keras import backend as K
from livingTree import SuperTree

init = tf.keras.initializers.HeUniform()
#init = tf.keras.initializers.LecunNormal()
sig_init = tf.keras.initializers.GlorotUniform()


class Model(object):

    def __init__(self, restore_from=None, dropout_rate=0, open_set=False):
        self.expand_dims = tf.expand_dims
        self.concat = Concatenate(axis=1)
        self.concat_a2 = Concatenate(axis=2)
        self.dropout = Dropout(dropout_rate)
        self.open_set = open_set
        if restore_from:
            self.__restore_from(restore_from)
            self.n_layers = len(self.spec_outputs)
        else:
            raise ValueError('Please given correct model path to restore.')

    def __restore_from(self, path):
        #mapper_dir = self.__pthjoin(path, 'feature_mapper')
        otlg_dir = self.__pthjoin(path, 'ontology.pkl')
        base_dir = self.__pthjoin(path, 'base')
        inters_dir = self.__pthjoin(path, 'inters')
        integs_dir = self.__pthjoin(path, 'integs')
        outputs_dir = self.__pthjoin(path, 'outputs')
        inter_dirs = [self.__pthjoin(inters_dir, i) for i in sorted(os.listdir(inters_dir), key=lambda x: int(x))]
        integ_dirs = [self.__pthjoin(integs_dir, i) for i in sorted(os.listdir(integs_dir), key=lambda x: int(x))]
        output_dirs = [self.__pthjoin(outputs_dir, i) for i in sorted(os.listdir(outputs_dir), key=lambda x: int(x))]
        self.ontology = load_otlg(otlg_dir)
        self.statistics = pd.read_csv(self.__pthjoin(path, 'statistics.csv'), index_col=0)
        self.labels, self.layer_units = parse_otlg(self.ontology)
        self.base = tf.keras.models.load_model(base_dir)
        self.spec_inters = [tf.keras.models.load_model(dir) for dir in inter_dirs]
        self.spec_integs = [tf.keras.models.load_model(dir) for dir in integ_dirs]
        self.spec_outputs = [tf.keras.models.load_model(dir) for dir in output_dirs]

    def init_encoder_block(self, phylogeny):
        block = tf.keras.Sequential(name='feature_encoder')
        block.add(Encoder(phylogeny))
        return block

    def init_base_block(self, num_features):
        block = tf.keras.Sequential(name='base')
        block.add(Flatten()) # (1000, )
        block.add(Dense(2**10, kernel_initializer=init))
        block.add(Activation('relu')) # (1024, )
        block.add(Dense(2**9, kernel_initializer=init))
        block.add(Activation('relu')) # (512, )
        return block

    def init_inter_block(self, index, name, n_units):
        k = index
        block = tf.keras.Sequential(name=name)
        block.add(Dense(self._get_n_units(8*n_units), name='l' + str(k) + '_inter_fc0', kernel_initializer=init))
        block.add(Activation('relu'))
        block.add(Dense(self._get_n_units(4*n_units), name='l' + str(k) + '_inter_fc1', kernel_initializer=init))
        block.add(Activation('relu'))
        block.add(Dense(self._get_n_units(2*n_units), name='l' + str(k) + '_inter_fc2', kernel_initializer=init))
        block.add(Activation('relu'))
        return block

    def init_integ_module(self, index, name, n_units):
        block = tf.keras.Sequential(name=name)
        k = index
        block.add(Dense(self._get_n_units(3*n_units), name='l' + str(k) + '_integ_fc0', kernel_initializer=sig_init))
        block.add(Activation('tanh'))
        return block

    def init_output_module(self, index, name, n_units):
        k = index
        block = tf.keras.Sequential(name=name)
        block.add(Dense(n_units, name='l' + str(index+2) + 'o_fc', kernel_initializer=sig_init))
        #block.add(Activation('sigmoid'))
        return block


    def _init_bn_layer(self):
        return BatchNormalization(momentum=0.9, scale=False)

    def _get_n_units(self, num):
        return int(num)

    def __pthjoin(self, pth1, pth2):
        return os.path.join(pth1, pth2)
    
def load_otlg(path):
    otlg = SuperTree().from_pickle(path)
    return otlg

def parse_otlg(ontology):
    labels = OrderedDict([(layer, label) for layer, label in ontology.get_ids_by_level().items()
                          if layer > 0])
    layer_units = [len(label) for layer, label in labels.items()]
    return labels, layer_units
