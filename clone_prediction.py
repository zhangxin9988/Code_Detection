import torch
import torch.nn as nn
import json
from Create_graph import create_edges,createast
from Create_model import Clone_Detection,Encode_Graph
from DataLoader_copy import Data_supplier
import dgl
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
model_path=r'C:\Users\zx\Downloads\googlejam4\copy\model_results'
model=Clone_Detection()
model=torch.load(model_path)

#输入代码对进行预测
#这两个是克隆对

classes={'0':'非克隆对','1':'克隆对'}
path1=r'C:\Users\zx\Downloads\googlejam4\googlejam4_src\1\googlejam1.p003.Mushroom.java'
path2=r'C:\Users\zx\Downloads\googlejam4\googlejam4_src\1\googlejam1.p006.A.java'
path3=r'C:\Users\zx\Downloads\googlejam4\googlejam4_src/1\googlejam1.p750.second.java'
path4=r'C:\Users\zx\Downloads\googlejam4\googlejam4_src/6\googlejam6.p254.RankAndFile.java'
vocabdict_fp = r'C:\Users\zx\Downloads\googlejam4\copy\vocabdict.json'
#vocabdict_fp0 = r'C:\Users\zx\Downloads\googlejam4\vocabdict.json'
def predict(path1,path2):  #做预测
    astdict,_,_= createast()
    vocabsize=8018
    with open(vocabdict_fp, 'r') as f:
        vocabdict = json.load(f)
    g1=create_edges(path1,astdict,vocabsize,vocabdict)
    g2=create_edges(path2,astdict,vocabsize,vocabdict)
    similarity=model(g1,g2)
    print('g1和g2的相似度为{:.2f}%'.format(similarity.item() * 100))
    if(similarity>=0.5):
        print('g1和g2是克隆对')
    else:
        print('g1和g2不是克隆对')
    return g1,g2,similarity.item()
#进行预测展示
g1,g2,similarity=predict(path1,path2)
#g1,g2,similarity=predict(path3,path4)
nx_g1=g1.to_networkx()
nx_g2=g2.to_networkx()
plt.figure()
plt.suptitle('Similarity:{:.2f}'.format(similarity))
plt.subplot(121)
nx.draw(nx_g1,pos=nx.kamada_kawai_layout(nx_g1),with_labels=True)
plt.title('节点数：{}，边数：{}'.format(nx_g1.number_of_nodes(),nx_g1.number_of_edges()))
plt.subplot(122)
nx.draw(nx_g2,pos=nx.kamada_kawai_layout(nx_g2),with_labels=True)
plt.title('节点数：{}，边数：{}'.format(nx_g2.number_of_nodes(),nx_g2.number_of_edges()))
plt.show()
dir_path='C:/Users/zx/Downloads/googlejam4/googlejam4_src'
pairs_each_class=15
pairs_between_classes=5
dataset,dataloader=Data_supplier(dir_path,pairs_each_class,pairs_between_classes)
print('数据集包括样例对数：',len(dataset))
def assessment_model():

    TP,TN,FP,FN=0,0,0,0 #true-positive,  true-negative,  false-positive,   false-negative
    model.eval()
    with torch.no_grad():
        for (bg1,bg2),label in dataloader:
            predicted=model(bg1,bg2)
            predicted=(predicted>=0.5).int().numpy()
            label=label.squeeze().numpy()

            for i in range(label.shape[0]):
                if(predicted[i]==1 and label[i]==1):
                    TP+=1
                elif(predicted[i]==0 and label[i]==0):
                    TN+=1
                elif(predicted[i]==1 and label[i]==0):
                    FP+=1
                elif(predicted[i]==0 and label[i]==1):
                    FN+=1
        Accuracy=(TP+TN)/(TP+TN+FP+FN) #准确率
        Precision=TP/(TP+FP)  #查准率
        Call=TP/(TP+FN)   #召回率
        F1_scores=2*TP/(2*TP+FP+FN)   #F1-scores
        print('Accuracy:{:.2f}%,Precision:{:.2f}%,Call:{:.2f}%,F1_scores:{:.2f}%'.format(Accuracy*100,Precision*100,Call*100,F1_scores*100))
        return Accuracy,Precision,Call,F1_scores
#模型的评估指标
assessment_model()
'''
Accuracy:82.43%,Precision:74.23%,Call:77.07%,F1_scores:75.62%
'''





































path1=r'C:\Users\zx\Downloads\googlejam4\googlejam4_src\1\googlejam1.p003.Mushroom.java'
path2=r'C:\Users\zx\Downloads\googlejam4\googlejam4_src\1\googlejam1.p006.A.java'
path3=r'C:\Users\zx\Downloads\googlejam4\googlejam4_src\4\googlejam4.p049.A.java'
path4=r'C:\Users\zx\Downloads\googlejam4\googlejam4_src\8\googlejam8.p064.SenateEvacuation.java'
'''
astdict,vocabsize,vocabdict = createast()
g1=create_edges(path1,astdict,vocabsize,vocabdict)
g2=create_edges(path2,astdict,vocabsize,vocabdict)
g3=create_edges(path3,astdict,vocabsize,vocabdict)
g4=create_edges(path4,astdict,vocabsize,vocabdict)
nx_g1=g1.to_networkx().to_directed()
nx_g2=g2.to_networkx().to_directed()
nx_g3=g3.to_networkx().to_directed()
nx_g4=g4.to_networkx().to_directed()
plt.subplot(221)
plt.title('nodes:{} | edges:{}'.format(nx_g1.number_of_nodes(),nx_g1.number_of_edges()))
nx.draw(nx_g1,pos=nx.kamada_kawai_layout(nx_g1),with_labels=True)
print(nx.degree(nx_g1))
plt.subplot(222)
plt.title('nodes:{} | edges:{}'.format(nx_g2.number_of_nodes(),nx_g2.number_of_edges()))
nx.draw(nx_g2,pos=nx.kamada_kawai_layout(nx_g2),with_labels=True)
plt.subplot(223)
plt.title('nodes:{} | edges:{}'.format(nx_g3.number_of_nodes(),nx_g3.number_of_edges()))
nx.draw(nx_g3,pos=nx.kamada_kawai_layout(nx_g3),with_labels=True)
plt.subplot(224)
plt.title('nodes:{} | edges:{}'.format(nx_g4.number_of_nodes(),nx_g4.number_of_edges()))
nx.draw(nx_g4,pos=nx.kamada_kawai_layout(nx_g4),with_labels=True)
plt.show()
'''
#print('标签为1')
#predict=model(g1,g2)
#print(predict)
