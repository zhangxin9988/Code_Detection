import matplotlib.pyplot as plt
import networkx as nx
import re
import json
import numpy as np
import pandas as pd
import os
import dgl
import torch
from torch import nn
from torch.nn import functional as F
from Create_graph import createast,create_edges
from Create_model import Clone_Detection
#准备训练集和测试集

def create_paths_file(dir_path,data_path):
    '''生成一个数据集路径的DataFrame'''
    samples_count=0
    dir_lists=os.walk(dir_path) #返回的是三元组的迭代器 ，三元组形式（root,dirs,files）,包含top目录以及其子目录
    root,dirs,_=next(dir_lists)
    Data_Path=pd.DataFrame(columns=['file_path','class'])
    for d in dirs:
        datafolder=os.path.join(root,d)
        sub_dir_list=os.walk(datafolder)
        for sub_root,_,sub_files in sub_dir_list:
            for file in sub_files:
                datafiler=os.path.join(sub_root,file)
                Data_Path=Data_Path.append({'file_path':datafiler,'class':int(d)},ignore_index=True)
    Data_Path.to_csv(data_path,encoding='utf-8')
    return data_path

def compute_pairs_num(g):
    return len(g)
def pick_pairs(df,pairs_number):
    indices=df.index #把这个分组的indice放在一个list中
    number_of_samples = len(indices) #该分组的样例个数
    clone_pairs=[]
    if(number_of_samples<pairs_number):
        for i in range(number_of_samples):
            for j in range(i+1,number_of_samples):
                clone_pairs.append((indices[i],indices[j]))
    else:
        np.random.seed(2)
        idx=np.random.choice(indices,(pairs_number,2),replace=True)
        for row in idx:
            clone_pairs.append((row[0],row[1]))
    print('第{}组抽取的数据对数{}'.format(df.name,len(clone_pairs)))
    return clone_pairs
#首先构造克隆对
def create_clone_pairs(Data_Path,pairs_number):
    Data_Path=Data_Path
    grouped = Data_Path.groupby('class')
    g=grouped.apply(pick_pairs,pairs_number)
    output=[]
    for c in g:
      output.extend(c)
    return output
#构造非克隆对
def get_indices(s):
    indices=s.index
    return list(indices)

def create_non_clone_pairs(Data_Path,pairs_number):
    grouped=Data_Path.groupby('class')
    indices=grouped.apply(get_indices)#每个分类对应的数据index
    print(indices.index)
    classes=len(indices) #共有12个类别
    non_clone_pairs=[]
    np.random.seed(0)
    for i in range(1,classes+1):
        for j in range(i+1,classes+1):
            min_len=min(len(indices[i]),len(indices[j]),pairs_number)
            a=indices[i]
            np.random.shuffle(a)
            b=indices[j]
            np.random.shuffle(b)
            pairs=list(zip(a[:min_len],b[:min_len]))
            non_clone_pairs.extend(pairs)
            print('第{}组&第{}组的数据对数：{}'.format(i,j,len(pairs)))
    return non_clone_pairs
#测试运行
def generate_pairs_index(Data_Path,pairs_each_class,pairs_between_classes):
    clone_pairs=create_clone_pairs(Data_Path,pairs_each_class)
    non_clone_pairs=create_non_clone_pairs(Data_Path,pairs_between_classes)
    return clone_pairs,non_clone_pairs
'''
dir_path='C:/Users/zx/Downloads/googlejam4/googlejam4_src'
data_path =r'C:/Users/zx/Downloads/googlejam4/copy/Data_Path.csv'
Data_Path=create_paths_file(dir_path,data_path)
create_clone_pairs(Data_Path,pairs_number=30)
create_non_clone_pairs(Data_Path,pairs_number=20)
'''
class Dataset(torch.utils.data.Dataset):
    def __init__(self,Data_Path,matched_pairs,mode='train'):
        super(Dataset,self).__init__()
        self.pairs=matched_pairs
        self.Data_Path=Data_Path
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
        path=(self.Data_Path['file_path'][graphs[0]],self.Data_Path['file_path'][graphs[1]])
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
    return (batched_graph_a,batched_graph_b),torch.tensor(labels)
pattern=r'[0-9]+'
def str_to_tuple(t):
    a,b=re.findall(pattern,t)
    return (int(a),int(b))
#tuple形式的match_pairs已经得到
def Data_supplier(dir_path,pairs_each_class,pairs_between_classes):
    data_path =r'C:/Users/zx/Downloads/googlejam4/copy/Data_Path.csv'
    # 生成保存所有数据的path的文件
    data_path=create_paths_file(dir_path,data_path)
    Data_Path = pd.read_csv(data_path,engine='python',encoding='utf-8')
    clone_pairs,non_clone_pairs=generate_pairs_index(Data_Path,pairs_each_class,pairs_between_classes)
    df_cp=pd.DataFrame(columns=['pairs','labels'])
    df_cp['pairs']=clone_pairs
    df_cp['labels']=1
    df_ncp=pd.DataFrame(columns=['pairs','labels'])
    df_ncp['pairs']=non_clone_pairs
    df_ncp['labels']=0
    matched_pairs=pd.concat([df_cp,df_ncp],axis=0,ignore_index=True)
    matched_pairs.to_csv('C:/Users/zx/Downloads/googlejam4/copy/matched_pairs.csv',index=False,encoding='utf-8')
    matched_pairs = pd.read_csv('C:/Users/zx/Downloads/googlejam4/copy/matched_pairs.csv',engine='python',encoding='utf-8')
    print(matched_pairs['labels'].value_counts())
    matched_pairs = matched_pairs.apply(lambda row: (str_to_tuple(row['pairs']), row['labels']), axis=1)
    dataset = Dataset(Data_Path, matched_pairs,mode='eval')
    dataloader = torch.utils.data.DataLoader(dataset,32, shuffle=True, collate_fn=collate_fn)
    return dataset,dataloader
dir_path='C:/Users/zx/Downloads/googlejam4/googlejam4_src'
data_path ='C:/Users/zx/Downloads/googlejam4/copy/Data_Path.csv'
model_path=r'C:\Users\zx\Downloads\googlejam4\copy\code_clone_detections' #在这里训练的数据是我自己生成的随机数据
'''
pairs_each_class=30
pairs_between_classes=20
dataset,dataloader=Data_supplier(dir_path,pairs_each_class,pairs_between_classes)
print('数据集包括样例对数：',len(dataset))
model=Clone_Detection()
lr=0.01
epochs=18
def train(lr,epochs,model_path):
    loss_func = nn.BCELoss()
    optimier = torch.optim.Adam(model.parameters(), lr)
    epoch_losses = []
    model.train() #训练模式
    for epoch in range(epochs):
        epoch_loss =0
        for iter, ((bg1, bg2), label) in enumerate(dataloader):
            #bg1,bg2=bg1.to(device),bg2.to(device)
            #label=label.to(device)
            predict = model(bg1, bg2)
            #print('predict:',predict)
            #print('-'*100)
            label=label.float().squeeze()
            #print('label:', label)
            #print('-' * 100)
            loss = loss_func(predict, label)
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimier.step()
            optimier.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= iter + 1
        print('Epoch:{} || Loss:{}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
    torch.save(model.state_dict(),model_path)
    print('Finished!!!')


def evaluate():
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            (bg1, bg2), label = data
            predicted = model(bg1, bg2)
            label=label.squeeze()
            # _,predicted=torch.max(outputs,1)
            total += label.size()[0]
            correct += ((predicted >= 0.5) == label).int().sum().item()
            print('预测：', (predicted >= 0.5).int())
            print('标签：', label)
    accuarcy = correct / total
    print('Accuarcy on trainset:{:.2f}%'.format(accuarcy*100))
'''
#path=r'C:\Users\zx\Downloads\googlejam4\code_clone_detections'
#model.load_state_dict(torch.load(path))
#train(lr,epochs,model_path)
#evaluate()

'''ifcount:5176
whilecount:726
forcount:6750
blockcount:11709
docount:10
switchcount:22
allnodes:  659534
vocabsize: 8018
数据集包括样例对数： 1855
0    944
1    911
Name: labels, dtype: int64

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
)
Epoch:0 || Loss:0.9540570039173653
Epoch:1 || Loss:0.6957423640736218
Epoch:2 || Loss:0.5256753060324438
Epoch:3 || Loss:0.38361055737939376
Epoch:4 || Loss:0.6706605846511906
Epoch:5 || Loss:0.38836485892534256
Epoch:6 || Loss:0.3289910326230115
Epoch:7 || Loss:0.3348811338173932
Epoch:8 || Loss:0.2539746192251814
Epoch:9 || Loss:0.1880559859604671
Epoch:10 || Loss:0.19458011043225898
Epoch:11 || Loss:0.2127864545789258
Epoch:12 || Loss:0.1230661412379865
Finished!!!
Accuarcy on trainset:96.98%'''
