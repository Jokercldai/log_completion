'''网络用于四通道输入
   改造网络，将上采样和聚合两个模块分隔开，分别单独调用。'''

from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from typing import Union, Tuple, Optional
from torch.autograd import Variable


import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
from torch_geometric.utils import unbatch

from torch_geometric.nn.inits import glorot, zeros


#device = torch.device('cuda')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:3')
device = torch.device('cpu')


class AggredGNN(MessagePassing):
    
    _alpha: OptTensor
    
    def __init__(
        self,
        ##############  cldai  ############  cldai  ############  cldai  ############
        gnn_in_channels: Union[int, Tuple[int, int]],
        gnn_out_channels: int,
        cnn_in_channels: int,
        cnn_out_channels: int,
        batchsize: int = 32,
        down_up: bool = True, #default True == down
        ifpool: bool = False,
        poolstride: int = 2,
        onlypoolout:bool = True,
        ##############  cldai  ############  cldai  ############  cldai  ############
        heads: int = 1,
        concat: bool = False,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = False,
        root_weight: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(AggredGNN, self).__init__(node_dim=0, **kwargs)
        ##############  cldai  ############  cldai  ############  cldai  ############
        self.gnn_in_channels = gnn_in_channels
        self.gnn_out_channels = gnn_out_channels
        self.cnn_in_channels = cnn_in_channels
        self.cnn_out_channels = cnn_out_channels
        self.batchsize = batchsize
        self.heads = cnn_out_channels
        self.down_up = down_up
        self.ifpool = ifpool
        self.poolstride = poolstride

        self.device = device
        print('！！！！！！！！！！！注意网络中device：',self.device)
        ##############  cldai  ############  cldai  ############  cldai  ############

        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(gnn_in_channels, int):
            gnn_in_channels = (gnn_in_channels, gnn_in_channels)


        ############ 改lin_key,lin_query，进而改key,query.基于提取的特征来求key和query ######################
        # self.lin_key = Linear(cnn_in_channels * gnn_in_channels[0], cnn_out_channels * gnn_out_channels)
        # self.lin_query = Linear(cnn_in_channels * gnn_in_channels[0], cnn_out_channels * gnn_out_channels)  #gnn_in_channels[0]改为gnn_in_channels[1]更合适些

        self.lin_key = Linear(cnn_out_channels * gnn_out_channels, cnn_out_channels * gnn_out_channels)
        self.lin_query = Linear(cnn_out_channels * gnn_out_channels, cnn_out_channels * gnn_out_channels)  #gnn_in_channels[0]改为gnn_in_channels[1]更合适些

        # self.lin_key = Linear(1,1)
        # self.lin_query = Linear(1,1)
        ############ 改lin_key,lin_query，进而改key,query.基于提取的特征来求key和query ######################

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, cnn_out_channels * gnn_out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(gnn_in_channels[1], cnn_out_channels * gnn_out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * cnn_out_channels * gnn_out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(gnn_in_channels[1], gnn_out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * gnn_out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        ###############  将alpha替换为矩阵w思路  #################  将alpha替换为矩阵w思路  #################
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        #self.shift = Parameter(torch.Tensor(42, gnn_out_channels, gnn_out_channels))
        self.shift = Parameter(torch.Tensor(12, gnn_out_channels, gnn_out_channels))
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        ###############  将alpha替换为矩阵w思路  #################  将alpha替换为矩阵w思路  #################

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        ###############  将alpha替换为矩阵w思路  #################  将alpha替换为矩阵w思路  #################
        glorot(self.shift)
        ###############  将alpha替换为矩阵w思路  #################  将alpha替换为矩阵w思路  #################

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_list,
                edge_attr: OptTensor = None, return_attention_weights=None):
        
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        CIN,COUT,GIN,GOUT = self.cnn_in_channels,self.cnn_out_channels,self.gnn_in_channels,self.gnn_out_channels
        #print('CIN,COUT,GIN,GOUT',CIN,COUT,GIN,GOUT)

        # print('edge_index',edge_index)
        # if isinstance(x, Tensor):
        #     x: PairTensor = (x, x)
        # #print('x[0].size()',x[0].size())

        
        value=x
        #print('value.size()',value.size())   #([16, 256, 64])

        ############ 改lin_key,lin_query，进而改key,query.基于提取的特征来求key和query ######################
        if isinstance(x, Tensor):
            x: PairTensor = (x, x) 

        query = self.lin_query(x[1].view(-1,COUT * GOUT)).view(-1, COUT, GOUT)
        key = self.lin_key(x[0].view(-1,COUT * GOUT)).view(-1, COUT, GOUT) 
        #print('key.size()',key.size())   #([16, 256, 64])   bsize=4
        #print('query.size()',query.size())  #([16, 256, 64])

        # key = torch.zeros([(x[0].shape)[0],1,1]).to(self.device)
        # query = torch.zeros([(x[0].shape)[0],1,1]).to(self.device)
        ############ 改lin_key,lin_query，进而改key,query.基于提取的特征来求key和query ######################
        
        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, edge_list=edge_list, size=None)

        return out         


    '''message中引入alpha，引入可学习矩阵w共同学习'''
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor, edge_list: Optional[int],
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.gnn_out_channels)
            key_j += edge_attr


        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.gnn_out_channels)
        '''alpha经过softmax会为0-1，为正，这样alpha*W*xj,所有效果对当前的效果几乎都一样了，错误的，
        因为有些曲线对当前节点效果应为负的，有些应该为正的,所以删去了alpha。'''
        #print('alpha.size()',alpha.size())
        alpha = softmax(alpha, index, ptr, size_i)   #Tanh 有正有负  nn.Tanh(alpha)    
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # # # #print('value_j.size()',value_j.size(),value_j)
        out = value_j
        out *= alpha.view(-1, self.cnn_out_channels, 1)


        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        #shift_w = self.shift.repeat(int(out.size()[0]/42),1,1)  #eg:repeat(2,1,1)意为0维复制2份，由原先的42变84，1维和2维保持维度不变
        shift_w = self.shift.repeat(self.batchsize,1,1)   #eg:repeat(2,1,1)意为0维复制2份，由原先的42变84，1维和2维保持维度不变
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        '''#从9.25日才开始去除这一句，感觉之前dropout的话岂不是影响后面分别取shift_w中的数据相乘？'''
        #shift_w = F.dropout(shift_w, p=self.dropout, training=self.training)  
        #print('shift_w.type',shift_w.dtype)

        #print('out.type',out.dtype)
        out_sum_batch = torch.zeros(out.size()).type_as(out)  #type_as进行数据类型转换
        d_start = 0
        d_end = 0
        
        for i in range(self.batchsize):
            d_end += len(edge_list[i])
            #print('d_end',d_end)

            #print('1111111  out[d_start:d_end].size()',out[d_start:d_end].size())
            out_sum_batch[d_start:d_end] = torch.matmul(out[d_start:d_end],shift_w[i*12:(i+1)*12][edge_list[i]])

            d_start += len(edge_list[i])
            #print('d_start',d_start)

        #print('out_sum_batch.size()',out_sum_batch.size(),out_sum_batch)
        
        return out_sum_batch


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.gnn_in_channels}, '
                f'{self.gnn_out_channels}, heads={self.heads})')



class GNN_conv_downsample(nn.Module):
    def __init__(self,in_ch,out_ch,onlypoolout,poolstride = 2):
        super(GNN_conv_downsample, self).__init__()  
        self.onlypoolout = onlypoolout
        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.ELU(),

            nn.Conv1d(in_channels=out_ch,out_channels=out_ch,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.ELU()
        )

        self.downsample= nn.Sequential(
            nn.MaxPool1d(5, stride=poolstride,padding = 2)
        )
        
        
    def forward(self,x):

        out_cat = self.Conv(x)
        out=self.downsample(out_cat)
        #print('out.size()',out.size(),'out.dtype',out.dtype,'out.type()',out.type())

        if self.onlypoolout==True:
            return out
        else:
            return out, out_cat



class GNN_conv_upsample(nn.Module):
    def __init__(self,in_ch,out_ch,poolstride = 2):
        super(GNN_conv_upsample, self).__init__()  

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels = in_ch, out_channels = 2*out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2*out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.ELU(),

            nn.Conv1d(in_channels = 2*out_ch, out_channels = 2*out_ch, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(2*out_ch),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose1d(in_channels=2*out_ch,out_channels=out_ch,kernel_size=7,stride=poolstride,padding=3,output_padding=(poolstride-1)),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.ELU(),
        )
        
    def forward(self,x):
        out = self.layers(x)

        return out
          

class upsample(nn.Module):
    '''仅进行上采样（将上采样与和聚合模块分割开）'''
    def __init__(self,gnn_out_ch, cnn_in_ch, cnn_out_ch, bsize, poolstride=2):
        super(upsample,self).__init__()
        self.bsize = bsize
        self.cout = cnn_out_ch
        self.gout = gnn_out_ch

        #############  换成直接的7个net,不放在self.names中，check是否是由self.names=locals()导致的不work，loss不降 #############
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        '''GNN_conv_upsample与upsample基于相同的基类(nn.Module)，所以在外面将upsample进行.to(device)后，这里不需要将GNN_conv_upsample再.to(device)'''
        self.conv1 = GNN_conv_upsample(cnn_in_ch,cnn_out_ch,poolstride)
        self.conv0 = GNN_conv_upsample(cnn_in_ch,cnn_out_ch,poolstride)
        self.conv2 = GNN_conv_upsample(cnn_in_ch,cnn_out_ch,poolstride)
        self.conv3 = GNN_conv_upsample(cnn_in_ch,cnn_out_ch,poolstride)

    def forward(self,x):
        y = torch.zeros([self.bsize * 4, self.cout, self.gout]).to(device)
      
        y[0 :self.bsize * 4:4][:] = self.conv0(x[0:self.bsize * 4:4][:])
        y[1 :self.bsize * 4:4][:] = self.conv1(x[1:self.bsize * 4:4][:])
        y[2 :self.bsize * 4:4][:] = self.conv2(x[2:self.bsize * 4:4][:])
        y[3 :self.bsize * 4:4][:] = self.conv3(x[3:self.bsize * 4:4][:])

        return y

class cat_fuse(nn.Module):
    '''将cat后的数据利用一层卷积层进行特征融合，使通道数由类512 to 256'''
    def __init__(self,cnn_in_ch,cnn_out_ch):
        super(cat_fuse,self).__init__()

        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=cnn_in_ch,out_channels=cnn_out_ch,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm1d(cnn_out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            #nn.ELU(),
        )
    def forward(self,x):
        y = self.Conv(x)

        return y


class mean_std_head(torch.nn.Module):
    def __init__(self):
        super(mean_std_head, self).__init__()

        self.mlp1 = MLP(4*512, 4*128, 4*32)
        self.mlp2 = MLP(4*32, 4*8, 8)
        
        #self.dropout = nn.Dropout(p=0.5)


    def forward(self, x, batch):
        
        y = torch.stack(unbatch(x, batch))
        #print('unbatch之后y.size()', y.size())  #[32, 4, 512]
        y = torch.flatten(y,1,-1)
        #print('flatten之后y.size()', y.size())   #, y[:6, 0:5]

        y = self.mlp1(y)
        #y = self.dropout(y)
        y = self.mlp2(y)
        #y = self.dropout(y)
        #print('全连接后y', y.size(), y)

        return y


class MLP(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hid_ch),
            #nn.BatchNorm1d(4*ch[0]*8),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Linear(hid_ch, hid_ch),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(hid_ch, out_ch), 
        )

    def forward(self, x):
        x = self.mlp(x)

        return x




#class Main_GCNN(torch.nn.Module):
class GraphNet(torch.nn.Module):             #这是将卷积应用到transformer中的版本
    '''GraphUnet无跳连'''
    def __init__(self,gnn_ch,batchsize,fea_litho,head,drop):
        super(GraphNet, self).__init__()


        '''正常U-Net下采样：两个conv，接一个pool;上采样：两个conv,接一个upsample'''
        #CNN_ch = [1,32,64,128,256,512]   
        CNN_ch = [4,32,64,128,256,512]
        #gnn_ch = [512,256,128,64,32]   
        result_ch=1
        self.bsize = batchsize
        self.fea_n = len(fea_litho)

        self.head = mean_std_head()
        self.gd0 = NoaggreGCNN(gnn_ch[0],gnn_ch[1],CNN_ch[0],CNN_ch[1],batchsize = batchsize,onlypoolout=False,down_up = True)
        self.gd1 = NoaggreGCNN(gnn_ch[1],gnn_ch[2],CNN_ch[1],CNN_ch[2],batchsize = batchsize,onlypoolout=False,down_up = True)
        self.gd2 = NoaggreGCNN(gnn_ch[2],gnn_ch[3],CNN_ch[2],CNN_ch[3],batchsize = batchsize,onlypoolout=False,down_up = True)
        self.gd3 = NoaggreGCNN(gnn_ch[3],gnn_ch[4],CNN_ch[3],CNN_ch[4],batchsize = batchsize,onlypoolout=False,down_up = True)

        self.u0 = upsample(gnn_ch[3],CNN_ch[4],CNN_ch[4], bsize=batchsize, poolstride=2).to(device)  #gnn_ch[3] is gout   
        self.cat0 =cat_fuse(CNN_ch[5],CNN_ch[4]).to(device) 
        self.gu0 = AggredGNN(gnn_ch[3],gnn_ch[3],CNN_ch[4],CNN_ch[4],dropout=0.1,batchsize = batchsize,down_up = False,poolstride=2)   #上采样中无关用ifpool，所以删去

        self.u1 = upsample(gnn_ch[2],CNN_ch[4],CNN_ch[3], bsize=batchsize, poolstride=2).to(device)
        self.cat1 =cat_fuse(CNN_ch[4],CNN_ch[3]).to(device)
        self.gu1 = AggredGNN(gnn_ch[2],gnn_ch[2],CNN_ch[3],CNN_ch[3],dropout=0.1,batchsize = batchsize,down_up = False,poolstride=2)   #poolstride默认等于2

        self.u2 = upsample(gnn_ch[1],CNN_ch[3],CNN_ch[2], bsize=batchsize, poolstride=2).to(device)
        self.cat2 =cat_fuse(CNN_ch[3],CNN_ch[2]).to(device)
        self.gu2 = AggredGNN(gnn_ch[1],gnn_ch[1],CNN_ch[2],CNN_ch[2],dropout=0.1,batchsize = batchsize,down_up = False,poolstride=2)
        
        self.u3 = upsample(gnn_ch[0],CNN_ch[2],CNN_ch[1], bsize=batchsize, poolstride=2).to(device)
        self.cat3 =cat_fuse(CNN_ch[2],CNN_ch[1]).to(device)
        self.gu3 = AggredGNN(gnn_ch[0],gnn_ch[0],CNN_ch[1],CNN_ch[1],dropout=0.1,batchsize = batchsize,down_up = False,poolstride=2)
        
        self.result = nn.Sequential(
            nn.Conv1d(CNN_ch[1],CNN_ch[1],kernel_size=7,stride=1,padding=3),
            nn.BatchNorm1d(CNN_ch[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv1d(CNN_ch[1],CNN_ch[1],kernel_size=7,stride=1,padding=3),
            nn.BatchNorm1d(CNN_ch[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv1d(CNN_ch[1],result_ch,kernel_size=7,stride=1,padding=3),
        )

    def forward(self,data):
        x, x_ori, edge_index_ini, edge_index, edge_ini_list, edge_list, batch = data.x, data.x_ori, data.edge_index_ini, data.edge_index, data.edge_ini_list, data.edge_list, data.batch
        #x = x.unsqueeze(1)   #由于输入是四通道的，所以这里不用unsqueeze
        #print('x.size()', x.size())  #[bsize*4, ch, length]
        #print('batch', batch)  ##[0,0,0,0,1,1,1,1...32,32,32,32]
        #print('x_ori.size()', x_ori.size())  #[bsize*4, length]


        ##输入原始单通道的数据预测其mean std等
        mean_std = self.head(x_ori, batch)
        #print('mean_std.size()', mean_std.size(), mean_std)  # torch.Size([bsize, 8])

        out, cat0 = self.gd0(x,edge_index_ini)     #NoaggreGCNN输入edge_index_ini，其实没用到
        out, cat1 = self.gd1(out,edge_index_ini)
        out, cat2 = self.gd2(out,edge_index_ini)
        out, cat3 = self.gd3(out,edge_index_ini)
        #print('out.shape', out.shape)

        out = self.u0(out)
        #print('cat3.device',cat3.device)   #查看cat3数据在GPU哪块卡或cpu上
        out = torch.cat((cat3,out),dim=1)
        out = self.cat0(out)
        out = self.gu0(out,edge_index_ini, edge_ini_list)
        
        out = self.u1(out)
        out = torch.cat((cat2,out),dim=1)
        out = self.cat1(out)
        out = self.gu1(out,edge_index, edge_list)
        
        out = self.u2(out)
        out = torch.cat((cat1,out),dim=1)
        out = self.cat2(out)
        out =self.gu2(out,edge_index, edge_list)
        
        out = self.u3(out)
        out = torch.cat((cat0,out),dim=1)
        out = self.cat3(out)
        out=self.gu3(out,edge_index, edge_list)
        
        #print('out.size()',out.size())
        out = self.result(out)

        out = out.squeeze(1)
        #print('out.size()',out.size())

        bound = int(mean_std.size()[1]/2)
        mean = mean_std[:, :bound]
        #print('mean.size()', mean.size(), mean)  # torch.Size([bsize, 4])
        std = mean_std[:, bound:]

        return out, mean, std 


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



class NoaggreGCNN(MessagePassing):
    
    _alpha: OptTensor
    
    # EXAMPLE:
    # INPUT 7 * 4 * 4096 
    # OUTPUT 7 * 8 * 2048
    # gnn_in_channels:  4096
    # gnn_out_channels: 2048
    # cnn_in_channels:  4
    # cnn_out_channels: 8
    # heads == cnn_out_channels = 8
    
    def __init__(
        self,
        ##############  cldai  ############  cldai  ############  cldai  ############
        gnn_in_channels: Union[int, Tuple[int, int]],
        gnn_out_channels: int,
        cnn_in_channels: int,
        cnn_out_channels: int,
        batchsize: int = 3,
        down_up: bool = True, #default True == down
        ifpool: bool = False,
        poolstride: int = 2,
        onlypoolout:bool = True,
        ##############  cldai  ############  cldai  ############  cldai  ############
        heads: int = 1,
        concat: bool = False,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = False,
        root_weight: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(NoaggreGCNN, self).__init__(node_dim=0, **kwargs)
        ##############  cldai  ############  cldai  ############  cldai  ############
        self.gnn_in_channels = gnn_in_channels
        self.gnn_out_channels = gnn_out_channels
        self.cnn_in_channels = cnn_in_channels
        self.cnn_out_channels = cnn_out_channels
        self.batchsize = batchsize
        self.heads = cnn_out_channels
        self.down_up = down_up
        self.ifpool = ifpool
        self.poolstride = poolstride
        self.onlypoolout = onlypoolout
        ##############  cldai  ############  cldai  ############  cldai  ############

        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(gnn_in_channels, int):
            gnn_in_channels = (gnn_in_channels, gnn_in_channels)

               
        
        ######
        '''Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
        所以torch.device('cuda')还是不行的'''
        #self.device = torch.device('cuda:0') 
        # self.device = torch.device('cpu') 
        self.device = device
        print('！！！！！！！！！！！注意 不聚合的网络中device：',self.device)
        #self.names = locals()

 
        #############  换成直接的7个net,不放在self.names中，check是否是由self.names=locals()导致的不work，loss不降 #############
        if self.down_up == True:
            '''师哥这种前面加整体加self.device = torch.device('cuda')的方法很好'''
            self.conv0 =  GNN_conv_downsample(cnn_in_channels,cnn_out_channels,onlypoolout,poolstride).to(self.device)  
            self.conv1 =  GNN_conv_downsample(cnn_in_channels,cnn_out_channels,onlypoolout,poolstride).to(self.device)
            self.conv2 =  GNN_conv_downsample(cnn_in_channels,cnn_out_channels,onlypoolout,poolstride).to(self.device)
            self.conv3 =  GNN_conv_downsample(cnn_in_channels,cnn_out_channels,onlypoolout,poolstride).to(self.device)
            ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
            #self.conv4 =  GNN_conv_downsample(cnn_in_channels,cnn_out_channels,ifpool,poolstride).to(self.device)
            #self.conv5 =  GNN_conv_downsample(cnn_in_channels,cnn_out_channels,ifpool,poolstride).to(self.device)
            #self.conv6 =  GNN_conv_downsample(cnn_in_channels,cnn_out_channels,ifpool,poolstride).to(self.device)
            ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        else:
            self.conv0 = GNN_conv_upsample(cnn_in_channels,cnn_out_channels,poolstride).to(self.device)
            self.conv1 = GNN_conv_upsample(cnn_in_channels,cnn_out_channels,poolstride).to(self.device)
            self.conv2 = GNN_conv_upsample(cnn_in_channels,cnn_out_channels,poolstride).to(self.device)
            self.conv3 = GNN_conv_upsample(cnn_in_channels,cnn_out_channels,poolstride).to(self.device)
            ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
            #self.conv4 = GNN_conv_upsample(cnn_in_channels,cnn_out_channels,poolstride).to(self.device)
            #self.conv5 = GNN_conv_upsample(cnn_in_channels,cnn_out_channels,poolstride).to(self.device)
            #self.conv6 = GNN_conv_upsample(cnn_in_channels,cnn_out_channels,poolstride).to(self.device)
            ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        #############  换成直接的7个net,不放在self.names中，check是否是由self.names=locals()导致的不work，loss不降 #############


        self.lin_key = Linear(cnn_in_channels * gnn_in_channels[0], cnn_out_channels * gnn_out_channels)
        self.lin_query = Linear(cnn_in_channels * gnn_in_channels[0], cnn_out_channels * gnn_out_channels)  #gnn_in_channels[0]改为gnn_in_channels[1]更合适些
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, cnn_out_channels * gnn_out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(gnn_in_channels[1], cnn_out_channels * gnn_out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * cnn_out_channels * gnn_out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(gnn_in_channels[1], gnn_out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * gnn_out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        CIN,COUT,GIN,GOUT = self.cnn_in_channels,self.cnn_out_channels,self.gnn_in_channels,self.gnn_out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        # query = self.lin_query(x[1].view(-1,CIN * GIN)).view(-1, COUT, GOUT)
        # key = self.lin_key(x[0].view(-1,CIN * GIN)).view(-1, COUT, GOUT)
        # '''query 和 self.lin_query这里我还是要好好理解一下'''        

 
        #############  换成直接的7个net,不放在self.names中，check是否是由self.names=locals()导致的不work，loss不降 #############
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        y = torch.zeros([self.batchsize * 4,COUT,GOUT]).to(self.device)
        #添加out_cat用于跳连
        out_cat = torch.zeros([self.batchsize * 4,COUT,GIN]).to(self.device)
        # for i in range(7):
        #     y[0 +i:self.batchsize * 6+1+i:7][:] = self.names['convingnn_' + str(i)](x[0][0+i:self.batchsize * 6+1+i:7][:])
        # #print('y.size()',y.size())
        #print('x[0][0:self.batchsize * 4:4][:].size()',x[0][0:self.batchsize * 4:4][:].size())
        #print('y.dtype',y.dtype, 'y.type()',y.type())
        #print('y[0 :self.batchsize * 4:4][:].type()', y[0 :self.batchsize * 4:4][:].type(),y[0 :self.batchsize * 4:4][:].size())

        if self.onlypoolout == True:
            y[0 :self.batchsize * 4:4][:] = self.conv0(x[0][0:self.batchsize * 4:4][:])
            y[1 :self.batchsize * 4:4][:] = self.conv1(x[0][1:self.batchsize * 4:4][:])
            y[2 :self.batchsize * 4:4][:] = self.conv2(x[0][2:self.batchsize * 4:4][:])
            y[3 :self.batchsize * 4:4][:] = self.conv3(x[0][3:self.batchsize * 4:4][:])
        else:
            y[0 :self.batchsize * 4:4][:], out_cat[0 :self.batchsize * 4:4][:] = self.conv0(x[0][0:self.batchsize * 4:4][:])
            y[1 :self.batchsize * 4:4][:], out_cat[1 :self.batchsize * 4:4][:] = self.conv1(x[0][1:self.batchsize * 4:4][:])
            y[2 :self.batchsize * 4:4][:], out_cat[2 :self.batchsize * 4:4][:] = self.conv2(x[0][2:self.batchsize * 4:4][:])
            y[3 :self.batchsize * 4:4][:], out_cat[3 :self.batchsize * 4:4][:] = self.conv3(x[0][3:self.batchsize * 4:4][:])
            #y[4 :self.batchsize * 6:6][:] = self.conv4(x[0][4:self.batchsize * 6:6][:])
            #y[5 :self.batchsize * 6:6][:] = self.conv5(x[0][5:self.batchsize * 6:6][:])
            #y[6 :self.batchsize * 7:7][:] = self.conv6(x[0][6:self.batchsize * 7:7][:])
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################

        #############  换成直接的7个net,不放在self.names中，check是否是由self.names=locals()导致的不work，loss不降 #############
        
        ############   去掉transformer聚合看下仅使用U-Net的效果   ############
        # # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        # out = self.propagate(edge_index, query=query, key=key, value=y,
        #                      edge_attr=edge_attr, size=None)
        # # out = out.unsqueeze(1)
        # '''这里out +=y什么意思,是融合自身原本信息（相当于又多加了一层自身信息）'''
        # #out += 3*y
        # #print('out.size()',out.size())
        # return out 
        return y, out_cat
        ############   去掉transformer聚合看下仅使用U-Net的效果   ############


