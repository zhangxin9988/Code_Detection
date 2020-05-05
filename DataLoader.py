from Create_graph import createast,create_edges,createtree
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
import dgl
import torch
import re
import json
def createpairdata(pathlist):
    datalist=[]
    countlines=1

    for line in pathlist:
        #print(countlines)
        countlines += 1
        root_dir='C:/Users/zx/Downloads/googlejam4'
        pairinfo = line.split()
        code1path = pairinfo[0].replace('\\', '/')
        code1path=os.path.join(root_dir,code1path)
        #print(code1path)
        code2path = pairinfo[1].replace('\\', '/')
        code2path = os.path.join(root_dir, code2path)
        #print(code2path)
        label=int(pairinfo[2])
        if(label==-1):
            label=0
        #print(label)
        datalist.append(((code1path,code2path),label))
    return datalist


class Dataset(torch.utils.data.Dataset):
    def __init__(self,matched_pairs,mode='train'):
        super(Dataset,self).__init__()
        self.pairs=matched_pairs
        vocabdict_fp = r'C:\Users\zx\Downloads\googlejam4\copy\vocabdict.json'
        if(mode=='train'):
            self.astdict, self.vocabsize, self.vocabdict = createast()
            with open(vocabdict_fp,'w') as f:
                json.dump(self.vocabdict,f)
        elif(mode=='eval'):
            self.astdict,_,_= createast()
            with open(vocabdict_fp, 'r') as f:
                self.vocabdict=json.load(f)
            self.vocabsize=8018
    def __getitem__(self, item):
        pair=self.pairs[item]
        graphs=pair[0]
        label=pair[1]
        path=(graphs[0],graphs[1])
        graph_pair = (create_edges(path[0],self.astdict, self.vocabsize, self.vocabdict),create_edges(path[1],self.astdict, self.vocabsize, self.vocabdict))
        return graph_pair,label
    def __len__(self):
        return len(self.pairs)


def collate_fn(samples):
    graph_pairs,labels=map(list,zip(*samples))
    a,b=map(list,zip(*graph_pairs))
    batched_graph_a=dgl.batch(a)
    batched_graph_b=dgl.batch(b)
    #clone_pairs和non_clone_pairs都是tuple形式，存储的是我们选出来的样本图对，我们统一挑出来每个元组的第一个元素组成一个列表
    #同理，我们挑出来每个元组的第二个元素组成另外一个对应的列表，这两个列表对应索引位置的graph是对应的
    return (batched_graph_a,batched_graph_b),torch.tensor(labels).view(-1,1).float()
def Data_supplier(path):   #此处的path指的是保存要使用的Java文件地址的一个文件
    with open(path, 'r', encoding='utf-8') as f:
        text = f.readlines()
    datalist=createpairdata(text)
    dataset = Dataset(datalist,mode='eval')
    dataloader = torch.utils.data.DataLoader(dataset,64, shuffle=False, collate_fn=collate_fn)
    return dataset,dataloader
'''
ifcount:5176
whilecount:726
forcount:6750
blockcount:11709
docount:10
switchcount:22
allnodes:  659534
vocabsize: 8018
数据集包括样例对数： 500
Clone_Detection(
  (graph_encoidng): Encode_Graph(
    (embed): Embedding(8018, 16)
    (gcn1): GraphConv(in=16, out=32, normalization=both, activation=None)
    (gcn2): SAGEConv(
      (feat_drop): Dropout(p=0.0, inplace=False)
      (fc_pool): Linear(in_features=32, out_features=32, bias=True)
      (fc_self): Linear(in_features=32, out_features=64, bias=True)
      (fc_neigh): Linear(in_features=32, out_features=64, bias=True)
    )
    (linear1): Linear(in_features=64, out_features=64, bias=True)
  )
  (linear1): Linear(in_features=64, out_features=128, bias=True)
  (linear2): Linear(in_features=64, out_features=16, bias=True)
  (linear3): Linear(in_features=16, out_features=1, bias=True)
)
Epoch:0 || Loss:2.1748580634593964
Epoch:1 || Loss:0.8720820397138596
Epoch:2 || Loss:0.6453210562467575
Epoch:3 || Loss:0.7311123721301556
Epoch:4 || Loss:0.48274561762809753
Epoch:5 || Loss:0.4338391423225403
Epoch:6 || Loss:0.3862685486674309
Epoch:7 || Loss:0.30031137354671955
Epoch:8 || Loss:0.2769793262705207
Epoch:9 || Loss:0.23969637416303158
Epoch:10 || Loss:0.20886839926242828
Epoch:11 || Loss:0.21750513650476933
Epoch:12 || Loss:0.47193985618650913
Epoch:13 || Loss:0.5380103550851345
Epoch:14 || Loss:0.26567542739212513
Epoch:15 || Loss:0.20406990125775337
Epoch:16 || Loss:0.14347735233604908
Epoch:17 || Loss:0.16939911153167486
Finished!!!
Accuarcy on trainset:83.00%'''