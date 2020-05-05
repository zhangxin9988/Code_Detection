import os
import random
import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
import torch
import dgl
from anytree import AnyNode, RenderTree
import json
#import treelib
from anytree import find
from graphmatch_clone.edge_index import edges

def get_token(node):  #获取节点的名字token
    token = ''
    #print(isinstance(node, Node))
    #print(type(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #print(node.__class__.__name__,str(node))
    #print(node.__class__.__name__, node)
    return token
def get_child(root):  #把输入节点root的所有孩子写在一个list中
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []
    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))

def get_sequence(node, sequence):       #获取从root开始整棵树的节点的token,采用的BFS
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)

def getnodes(node,nodelist):  #获取从root开始整棵树的node，采用的BFS
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child,nodelist)
def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)

def createast():
    asts=[]
    paths=[]
    alltokens=[]
    dirname =r'C:\Users\zx\Downloads\googlejam4\googlejam4_src'
    for i in range(1,13):
        for rt, dirs, files in os.walk(os.path.join(dirname,str(i))):
            for file in files:
                programfile=open(os.path.join(rt,file),encoding='utf-8')
                #print(os.path.join(rt,file))
                programtext=programfile.read()
                #programtext=programtext.replace('\r','')
                programtokens=javalang.tokenizer.tokenize(programtext)
                #print(list(programtokens))
                programast=javalang.parser.parse(programtokens)
                paths.append(os.path.join(rt,file))
                asts.append(programast)
                get_sequence(programast,alltokens)
                programfile.close()
                #print(programast)
                #print(alltokens)
    astdict=dict(zip(paths,asts)) ###
    ifcount=0
    whilecount=0
    forcount=0
    blockcount=0
    docount=0
    switchcount=0
    for token in alltokens:
        if token=='IfStatement':
            ifcount+=1
        if token=='WhileStatement':
            whilecount+=1
        if token=='ForStatement':
            forcount+=1
        if token=='BlockStatement':
            blockcount+=1
        if token=='DoStatement':
            docount+=1
        if token=='SwitchStatement':
            switchcount+=1
    print('ifcount:{}\nwhilecount:{}\nforcount:{}\nblockcount:{}\ndocount:{}\nswitchcount:{}'.format(ifcount,whilecount,forcount,blockcount,docount,switchcount))
    print('allnodes: ',len(alltokens))
    alltokens=list(set(alltokens))
    vocabsize = len(alltokens)   ###
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))  ###
    print('vocabsize:',vocabsize)
    return astdict,vocabsize,vocabdict
#astdict,vocabsize,vocabdict=createast()
#vocabdict_fp=r'C:\Users\zx\Downloads\googlejam4\vocabdict.json'
#with open(vocabdict_fp,'w') as f:
#    json.dump(vocabdict, f)
#print(vocabdict)
#vocabsize=8018

def getnodeandedge_astonly(node,nodeindexlist,vocabdict,src,tgt):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child,nodeindexlist,vocabdict,src,tgt)
def getedge_flow(node,vocabdict,src,tgt,ifedge=False,whileedge=False,foredge=False):
    token=node.token
    if whileedge==True:
        if token=='WhileStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)

            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
    if foredge==True:
        if token=='ForStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
    if ifedge==True:
        if token=='IfStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            if len(node.children)==3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
    for child in node.children:
        getedge_flow(child,vocabdict,src,tgt,ifedge,whileedge,foredge)
def getedge_nextsib(node,vocabdict,src,tgt):
    token=node.token
    for i in range(len(node.children)-1):
        src.append(node.children[i].id)
        tgt.append(node.children[i+1].id)
        src.append(node.children[i+1].id)
        tgt.append(node.children[i].id)
    for child in node.children:
        getedge_nextsib(child,vocabdict,src,tgt)
def getedge_nextstmt(node,vocabdict,src,tgt):
    token=node.token
    if token=='BlockStatement':
        for i in range(len(node.children)-1):
            src.append(node.children[i].id)
            tgt.append(node.children[i+1].id)
            src.append(node.children[i+1].id)
            tgt.append(node.children[i].id)
    for child in node.children:
        getedge_nextstmt(child,vocabdict,src,tgt)

#astdict, vocabsize, vocabdict = createast()
def create_edges(path,astdict, vocabsize, vocabdict):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    tree = javalang.parse.parse(text)
    nodeindexlist = []
    newtree = AnyNode(id=0, token=None, data=None)
    createtree(newtree, tree, nodeindexlist)
    # print(RenderTree(newtree))
    nodeindexlist = []
    src, tgt = [], []
    getnodeandedge_astonly(newtree, nodeindexlist, vocabdict, src, tgt)
    getedge_flow(newtree, vocabdict, src, tgt, ifedge=True, whileedge=True, foredge=True)
    getedge_nextsib(newtree, vocabdict, src, tgt)
    getedge_nextstmt(newtree, vocabdict, src, tgt)
    G=dgl.DGLGraph()
    G.add_nodes(len(nodeindexlist))
    G.add_edges(src, tgt)
    G.ndata['t'] = torch.tensor(nodeindexlist)
    #print('节点数{}，边数{}'.format(G.number_of_nodes(),G.number_of_edges()))
    return G
