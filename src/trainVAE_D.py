# coding: utf-8
import sys
sys.path.append('../')
import getopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random 
from src.Constant import Constants
from util.load_data import StyleData
from src.PreTrainDs import indexData2variable
import time


def trainVAE_D(epoches,batch_size,data,ds_model,ds_emb,pretrainD=False,pretrainVAE=False):
    
    gan = torch.load(gan_path)
    gan = gan.cuda()
    style = StyleData()
    style.load(style_path)
    const = Constants(style.n_words)
    optimizer = optim.Adam(gan.parameters(),lr=const.Lr)
    lamda1 = 1
    lamda2 = 1
    lamda3 = 3
    cross_entropy = nn.CrossEntropyLoss()

    # init the state of some model
    ds_model.train(False)
    ds_emb.train(False)

    
    train_data = indexData2variable(data)
    train_data = build2pairs(train_data)
    
    
    print len(train_data)
    for i in range(epoches):
        print "epoches:\t", i
        if pretrainD:
            print "trainning Discriminator.........."
        else :
            print "trainning Generator.............."
        
        stime = time.time()
        
        shuffleData(train_data)
        
        count = 0
        while count < len(train_data):
            tempdata = train_data[count:count+batch_size]
            
            if tempdata == []:
                break
                
            count += batch_size
            optimizer.zero_grad()
            Lrec = 0
            Lcyc = 0
            Ldis = 0
            Ladv = 0

            Loss = 0

            # before we let the D lead the gradient the D model must be strong enough
            if not pretrainD:
                for seqs in tempdata:
                    dic = gan(seqs[0],seqs[1],D_train=False)

                    Lrec = cross_entropy(dic['x1_hat_noT'],seqs[0])+cross_entropy(dic['x2_hat_noT'],seqs[1])
                    Lcyc = cross_entropy(dic['x1_bar_noT'],seqs[0])+cross_entropy(dic['x2_bar_noT'],seqs[1])
                    emb = ds_emb(seqs[0]).unsqueeze(0).unsqueeze(0)
                    if not pretrainVAE:
                        Ldis = (ds_model(emb)[0][1]*(dic['y1']-dic['y_star'])**2).sum()
                    
                    if not pretrainVAE:
                        Ladv = cross_entropy(dic['D_x1_wl'],Variable(torch.LongTensor([0]).cuda())) + cross_entropy(dic['D_x2_hat'],Variable(torch.LongTensor([1]).cuda()))
                    Loss += Lrec + lamda2*Lcyc
                    
                    if not pretrainVAE:
                        Loss += lamda3*Ldis - lamda1*Ladv
            else:
                for seqs in tempdata:
                    dic = gan(seqs[0],seqs[1],Ez_train=False,Ey_train=False,G_train=False,
                              Lcyc=False, Lrec=False, Ldis = False)
                    
                    Ladv = cross_entropy(dic['D_x1_wl'],Variable(torch.LongTensor([0]).cuda()))+ cross_entropy(dic['D_x2_hat'],Variable(torch.LongTensor([1]).cuda()))
                    Loss += lamda1*Ladv
            
#             print "loss \t\t%.3f" %(Loss.data.cpu().numpy()[0])
            
            Loss.backward()
            optimizer.step()
            
            
            
        if i%10 == 0 or i:
            torch.save(gan, gan_path)
                
            gan.eval()
            acc = get_d_acc(gan, train_data)
            gan.train(True)
            
#             if acc > 0.8:
#                 pretrainD = False
#             if acc < 0.6:
#                 pretrainD = True
            
            
            
        etime = time.time()
        print "cost time \t%.2f mins" % ((etime - stime)/60)
    torch.save(gan, gan_path)
            
                
def build2pairs(train_data):
    data = []
    for i in range(min( len(train_data[0]), len(train_data[1]) )):
           data.append([train_data[0][i], train_data[1][i]])
    return data

def shuffleData(train_data):
    """
    this function don't need to return any value and the list is changed inplace
    """
    if len(train_data) == 2:
        random.shuffle(train_data[0])
        random.shuffle(train_data[1])
    else:
        random.shuffle(train_data)

        
def get_d_acc(gan, train_data):
    
    acc = 0
    min_len = len(train_data)/100
    train_data = train_data[:min_len]
    for i in range(min_len):
        dic = gan(train_data[i][0],train_data[i][1],Ez_train=False,Ey_train=False,G_train=False,
                              Lcyc=False, Lrec=False, Ldis = False)
        if dic['D_x1_wl'].topk(1)[1].cpu().data.numpy() == 0:
            acc += 1
        if dic['D_x2_hat'].topk(1)[1].cpu().data.numpy() == 1:
            acc += 1
    
    print "acc:\t\t %.4f" % (acc/(min_len*2.0))
    return acc/(min_len*2.0)

    
    
if __name__ == "__main__":
    
    """
    you shuld use this script in this way:
    python trainVAE_D.py <epoches> <batch_size> <pretrainD?> <pretrainVAE?> <traindatafilename>  <styledatafilename> ganName

    for instance: 
    python trainVAE_D.py 1000 20 yes/no yes/no ./data/trainDataOfIndex.npy ./data/style ./Model/gan.pkl
    """
    opts, agrvs = getopt.getopt(sys.argv[1:], "", ["epoches=", "batch_size=", "pretrainD", "pretrainVAE", "traindata=", "style=", "gan=", "ds=", "ds_emb="])
    pretrainD = False
    pretrainVAE = False
    for op in opts:
        if "--epoches" in op:
            epoches = int(op[1])
        if "--batch_size" in op:
            batch_size = int(op[1])
        if "--pretrainD" in op:
            pretrainD = True
        if "--pretrainVAE" in op:
            pretrianVAE = True
        if "--traindata" in op:
            traindata = op[1]
        if "--style" in op:
            style_path = op[1]
        if "--gan" in op:
            gan_path = op[1]
        if "--ds" in op:
            ds_path = op[1]
        if "--ds_emb" in op:
            ds_emb_path = op[1]
            
    
    ds = torch.load(ds_path).cuda()
    ds_emb = torch.load(ds_emb_path).cuda()
    
    train_data = np.load(traindata)
    

    trainVAE_D(epoches,batch_size,train_data,ds,ds_emb,pretrainD,pretrainVAE)
    
    print "finished trainning......................."
