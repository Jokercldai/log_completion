import time
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.optim import lr_scheduler
import os
from utils.utils import *
from utils.load_data import *    #only use MyGNNDataset  #cldai
import torch_geometric.loader as loader
from utils.model import *
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'   

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")    #32
    parser.add_argument('--sam_num',default=2560, type=int, help='the number of sample in every set') #2560
    parser.add_argument('--epochs', default=1501, type=int, help='Number of epochs to train.')
    parser.add_argument("--save_model_path", default='./model/ctra_exp/', help="Save_model_path")  #'./model/
    parser.add_argument("--lossfigpath", default='/home/cldai/sciresearch/logcomplete/figure/ctra_exp/', help="loss图存放地址")#

    parser.add_argument('--sam_len',default=512, type=int, help='length of every sample')  #cldai add
    parser.add_argument("--graph_channels", default = [512,256,128,64,32], type=list, help="Channels of graphunet in each layers")  #[64,32,16,8,4,1]，[128,64,32,8],[512,128,32,8]
    #parser.add_argument("--graph_channels", default = [256,128,64,32,16,4], type=list, help="Channels of graphunet in each layers")  
    parser.add_argument('--data_norm_type',default='',type=str ,
                        help="数据归一化的类型,按样本'sample'(sample)每个分别归一化或者按每口井(well)归一化")    ############             
    parser.add_argument('--LR', type=float, default=0.001, help='Learning rate.')    #original is 0.0002
    
    ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
    ###fea 的顺序还是重要的，比如utils.py 中算不同曲线的loss时就使用到曲线的顺序
    parser.add_argument('--fea_litho',default=['GR','RHOB', 'NPHI', 'DTC'],type=list,help="feature and lithology that we need")  #'CALI',
    ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
    parser.add_argument('--need_maskd',default=True, help='现在去 自身节点聚合，本质上不需要maskd,所以False，')

    parser.add_argument('--filePath',default='/home/cldai/sciresearch/logcomplete/data/Teamdata/standard_wells/data2/', help='file path of train and test')  #cldai add
    parser.add_argument("--loss_name", default='./loss/loss_init.npz', help="Loss_name")
    parser.add_argument('--filt_size',default=19,type=int,help='中值滤波的窗口大小，只可为奇数')  ##7.14日用filt_size由7改为13
    parser.add_argument("--drop", default=0, type=float, help="网络的dropout")
    parser.add_argument('--head', default=1, type=int, help="head")
    args = parser.parse_args()

    return args
    #

def visualize_maskc(maskc,well,name):
    #可视化maskc或其一部分

    fig,axes = plt.subplots(1,2,sharey=True,figsize=(5,10),dpi=300)   #(wigth,height)
    samp = 80000  #samp是为查看多少个采样点
    msec = maskc[0:samp].flatten()    #ndarray.flatten()展平数组，C 按行展平（默认）；F 按列展平

    wellsec = pd.DataFrame(well.iloc[0:samp].copy(),columns=['WELL'])  #对于datafraame，columns要传入集合形式，如['WELL']可以，而‘WELL’则不可
    #print('wellsec.columns',wellsec.columns,wellsec.shape)
    base = 11111
    well_name = wellsec['WELL'].unique()
    for it in range(len(well_name)):
        wellsec[wellsec['WELL']==well_name[it]] = base*(it+1)

    #print('wellsec',wellsec)
    y = [i for i in range(len(msec))]
    axes[0].imshow(np.array(wellsec,dtype=np.float64),interpolation='nearest',aspect='auto')  #imshow中的数据为数组时，数据需为浮点型
    axes[1].plot(msec,y,color='k')

    plt.tight_layout()
    plt.savefig('/home/cldai/sciresearch/logcomplete/figure/check_maskc/'+str(name)+'maskc.png')


def main(args):
    
    #print(args.loss_name)
    print('读取数据路径：',args.filePath)
    print('模型保存在：',args.save_model_path)
    
    dfTrain=pd.read_csv(args.filePath+'train_standard.csv',engine='python')
    dfTest=pd.read_csv(args.filePath+'val_standard.csv',engine='python')
    #mean_log=pd.read_csv(args.filePath+'train_logs_mean.csv')
    #print('dfTrain.columns',dfTrain.columns)

    ################  查看没有去除null值的数据(即原始数据)最浅深度和最深深度分布  ################
    df_fea_md = dfTrain[['WELL','DEPTH_MD','GR', 'RHOB', 'NPHI', 'DTC']].copy()
    dep_min = df_fea_md['DEPTH_MD'].min()
    dep_max = df_fea_md['DEPTH_MD'].max()
    print('dep_min',dep_min)    #dep_min=136.086   #311.624(去除null值则dep_min=311.624)
    print('dep_max',dep_max)    #dep_max=5436.632
    print('井按深度划分格子数：',int((dep_max-dep_min)/0.152)+1)  #34873
    ################  查看最浅深度和最深深度分布  ################
    
    ############################  按所有井来设计一个关于拼接处的mask，目的是避免选择到拼接附近的样本  #############################a = time.time()
    train_well = dfTrain['WELL'].copy()  #注意只取一列则现在well是Series而不是DataFrame
    train_maskc = np.ones((dfTrain.shape[0],1))   #maskc定义为mask_cat_loc

    train_wells_name = dfTrain['WELL'].unique()
    train_wells_num = len(train_wells_name)
    train_loc_li = [-1]  #loc_li存储完井位置,加上-1为表示起始处，0不可，因为起始点可能选择到0
    for i in range(train_wells_num):
        loc = dfTrain[dfTrain['WELL']==train_wells_name[i]].index[-1].tolist()  #index取[-1]即为该井完井位置
        #print('loc',loc)
        train_loc_li.append(loc)
        train_maskc[loc-args.sam_len+1 : loc+1]=0

    #print('train_loc_li',train_loc_li)
    test_well = dfTest['WELL'].copy()  #注意只取一列则现在well是Series而不是DataFrame
    test_maskc = np.ones((dfTest.shape[0],1))   #maskc定义为mask_cat_loc

    test_wells_name = dfTest['WELL'].unique()
    #print(test_wells_name[0].type)  #int64
    test_wells_num = len(test_wells_name)
    test_loc_li = [-1]  #loc_li存储完井位置
    for i in range(test_wells_num):
        loc = dfTest[dfTest['WELL']==test_wells_name[i]].index[-1].tolist()  #index取[-1]即为该井完井位置
        #print('loc',loc)
        test_loc_li.append(loc)
        test_maskc[loc-args.sam_len+1 : loc+1]=0
    
    #visualize_maskc(train_maskc,train_well,name='train_')
    #visualize_maskc(test_maskc,test_well,name='test_')
    ############################  按所有井来设计一个关于拼接处的mask，目的是避免选择到拼接附近的样本  #############################


    # pos_index = np.load('./data/preprocessdata1/pos_index.npz')['arr_0']
    # new_pos_index = pos_index.copy()   
    #print('new_pos_index',new_pos_index.shape,new_pos_index)   #(2, 42)

    print('数据已读取！！！')
    train_dataset = MyGNNDataset(dfTrain, args.sam_len, args.sam_num, args.fea_litho, 
                                args.data_norm_type, args.filt_size,
                                args.need_maskd, train_maskc, train_loc_li)
    test_dataset = MyGNNDataset(dfTest, args.sam_len, args.sam_num, args.fea_litho, 
                                args.data_norm_type, args.filt_size,
                                args.need_maskd, test_maskc, test_loc_li)
    test_loader = loader.DataLoader(test_dataset, batch_size=args.batch_size,num_workers=12)  #num_workers=12相当于12个线程读数据
    train_loader = loader.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=12)
    '''train_loader将数据按iteration*[batch_size,self.sam_len]存放。iteration=len(train_loader)， self.sam_num=iteration * batch_size'''
    '''现在要求一个epoch理论上要覆盖所有数据。 即iteration * batch_size * self.sam_len>=训练集数据'''
    print('len(train_loader)',len(train_loader))  #len(train_loader)=iteration
    '''num_workers遇到的影响之一：在MyGNNDataset中因为多线程，print显示看起来未按逻辑顺序。'''
    '''这里gdata在前面没直接发现导入是因为他在from utils.load_data import *的load_data文件中有import torch_geometric.data as gdata'''

    print('Finish read data, begin to train')
    device = torch.device('cuda:3')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphNet(args.graph_channels,args.batch_size,args.fea_litho,args.head,args.drop).to(device) 
    #model = torch.nn.DataParallel(model,device_ids=[3])    #cldai notes: [5] is cuda:5   #当device = torch.device('cuda:2')时，device_ids=[0],会报错在两个cuda上
    print('device',device)
    
    
    train_network(model,device,args.LR,args.epochs,args.save_model_path,train_loader,
                    test_loader,args.loss_name,args.lossfigpath,args.batch_size,args.fea_litho)
    
    
    
if __name__ == '__main__':
    args = read_args()
    main(args)

