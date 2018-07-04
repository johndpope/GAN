# coding: utf-8
import sys
sys.path.append('../')
import numpy as np
import random
import torch
from src.Constant import Constants
from src.ModelDefine import GANModel
from util.load_data import StyleData
from src.PreTrainDs import indexData2variable
import getopt




if __name__ == "__main__":
    # this is just to build the gan model and save the network to use later
    opt, args = getopt.getopt(sys.argv[1:],"s:o:",["style=", "output="])
#    print opt, args
    for op in opt:
#	print op
        if "--style" in op:
            style_path = op[1]
        if "--output" in op:
            output = op[1]
            
    style = StyleData()
    style.load(style_path)
    const = Constants(n_vocab=style.n_words)
    gan = GANModel(content_represent=const.Content_represent,
                   D_filters=const.D_filters,
                   D_num_filters=const.Ds_num_filters,
                   embedding_size=const.Embedding_size,
                   Ey_filters=const.Ey_filters,
                   Ey_num_filters=const.Ey_num_filters,
                   n_vocab=const.N_vocab,
                   style_represent=const.Style_represent,
                   temper=const.Temper)  # there are 9 parameters of a GAN
    torch.save(gan, output)
    print 'finished'
