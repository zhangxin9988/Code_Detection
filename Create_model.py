from DataLoader import Data_supplier
import torch
from torch import nn
from torch.nn import functional as F
import dgl
from dgl.nn.pytorch import GraphConv,GATConv,SAGEConv,GINConv

#定义模型
#step1 将整图编码为词嵌入向量
device=torch.device("cuda:0"if torch.cuda.is_available else "cpu")
class Encode_Graph(nn.Module):
    def __init__(self,vocablen,embedding_dim):
        super(Encode_Graph,self).__init__()
        self.embed=nn.Embedding(vocablen,embedding_dim)
        self.gcn1=GraphConv(16,32)
        self.gcn2 =SAGEConv(32,64,aggregator_type='pool')
        self.linear1=nn.Linear(64,64)
    def forward(self,g):
        inputs=g.ndata['t'].squeeze()
        h=self.embed(inputs)
        #h=h.to(device)
        h=F.relu(self.gcn1(g,h))
        h=F.relu(self.gcn2(g,h))
        h=F.relu(self.linear1(h))
        g.ndata['h']=h
        hg=dgl.sum_nodes(g,'h')
        return hg
#整体模型结构
class Clone_Detection(nn.Module):
    def __init__(self):
        super(Clone_Detection,self).__init__()
        self.graph_encoidng=Encode_Graph(8018,embedding_dim=16)
        self.linear1=nn.Linear(64,128)
        #self.linear2=nn.Linear(64,16)
        #self.linear3=nn.Linear(16,1)
    def forward(self,bg1,bg2):
        bg1=self.graph_encoidng(bg1)
        bg1=F.relu(self.linear1(bg1))
        bg2=self.graph_encoidng(bg2)
        bg2=F.relu(self.linear1(bg2))
        sim=F.cosine_similarity(bg1,bg2)
        sim=torch.abs(sim)
        sim = torch.clamp(sim, 0, 1)
        return sim
'''
#加载训练集
train_path = r'C:/Users/zx/PycharmProjects/Deep_learning/graphmatch_clone/testsmall.txt'
dataset,dataloader=Data_supplier(train_path)
print('数据集包括样例对数：',len(dataset))


#对模型进行训练
lr=0.01
epochs=10
#model_path=r'C:/Users/zx/Downloads/googlejam4/copy/code_clone_detections'
#model_path=r'C:/Users/zx/Downloads/googlejam4/code_clone_detections' #在这里训练的数据来自testsamll.txt
def train(lr,epochs,model_path,model):
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
    torch.save(model,model_path)
    print('Finished!!!')


def evaluate(model):
    correct = 0
    total = 0
    model=model.eval()
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
#创建模型实例
#model=Clone_Detection()
#print(model)


#model.to(device)
#model_path=r'C:\Users\zx\Downloads\googlejam4\code_clone_detections'
#train(lr,epochs,model_path,model)
#model=torch.load(model_path)
#evaluate(model)