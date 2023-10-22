import os
from re import T
import numpy as np
import pandas as pd
import random
import scipy.io as sio
import scipy.signal as signal
from skimage.measure import block_reduce
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import time


import torch
import torch_geometric
import torch_geometric.data as gdata
from torch_geometric.data import DataLoader
from utils import *

#########  ##################  #############  #############
gdata.InMemoryDataset
class MyGNNDataset(gdata.Dataset):
    def __init__(self, data, sam_len, sam_num, fea_litho, norm_type, filt_size, need_maskd, 
                maskc, loc_li):
        gdata.Dataset.__init__(self)
        self.signal = data  #这里data和self.signal是dataframe
        self.sam_len = sam_len
        self.sam_num = sam_num
        self.fea_litho = fea_litho
        self.norm_type = norm_type
        self.filt_size = filt_size
        self.need_maskd = need_maskd
        self.maskc = maskc
        self.loc_li = loc_li
        

    def __getitem__(self,index):
        '''完成标签集、数据集、mask集的制作'''
         
        label_ori, label, mask0, sam_stap, sam_endp, well_order,well_stap, well_endp,np_mean, np_std=self.label_mask0(self.signal)

        if self.need_maskd == True:
            maskd=self.mask_d(label,mask0)
        else:
            maskd = np.ones((label.shape[0],label.shape[1]))
        
        #制作对应的sample
        input_old=label*maskd  #label*maskd作为输入input
        input_ori = label_ori*maskd

        #将input转置为后面符合numpy的按行进行下采样
        input = input_old.T
        input_d4 = self.resample(input,4)
        input_d8 = self.resample(input,8)
        input_d16 = self.resample(input,16)
        # input_d4 = self.resample(input,8)  
        # input_d8 = self.resample(input,16)
        # input_d16 = self.resample(input,32)

        #将二维拓展为三维并在通道维度上cat。拓维方法有[:, np.newaxis/None,:],  reshape(4,1,512); np.expand_dims(arr,axis=1(1意为拓展后在意为上))
        input_3dim = input[:,np.newaxis,:]
        d4_3dim = input_d4[:,np.newaxis,:]   #注意这种改法input_d4与d4_3dim内存是不共享的，即input_d4 shape =(4,512)不变
        d8_3dim = input_d8[:,np.newaxis,:]
        d16_3dim = input_d16[:,np.newaxis,:]

        #np.concatenate拼接成四通道数据
        input_4ch = np.concatenate((input_3dim,d4_3dim,d8_3dim,d16_3dim),axis=1)

        #画input_4ch图
        #self.plot_input_ch(input_4ch)

        edge_index_ini, edge_ini_list, edge_index, edge_list = self.construct_edge(input_old)

        x_ori = torch.tensor(input_ori.T, dtype=torch.float)  #index为了索引对应的batch
        x = torch.tensor(input_4ch, dtype=torch.float)   #input_4ch已经被转置，这里无需转置
        label = torch.tensor(label.T, dtype=torch.float)
        label_ori = torch.tensor(label_ori.T, dtype=torch.float)
        mask0_1 = torch.tensor(mask0.T, dtype=torch.float)
        maskd = torch.tensor(maskd.T, dtype=torch.float)
        edge_index_ini = torch.tensor(edge_index_ini, dtype = torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        norm_mean = torch.tensor(np_mean.T, dtype=torch.float)
        norm_std = torch.tensor(np_std.T, dtype=torch.float)

        return gdata.Data(x=x, edge_index_ini=edge_index_ini, edge_index=edge_index, y=label, mask=mask0_1, maskd=maskd,
                            edge_ini_list=edge_ini_list, edge_list=edge_list,
                            sam_stap=sam_stap, sam_endp=sam_endp, well_order=well_order,
                            well_stap=well_stap, well_endp=well_endp, 
                            norm_mean = norm_mean, norm_std=norm_std,
                            label_ori = label_ori, x_ori = x_ori
                            )  #应该这里写y,mask,后面eg:mydata调用出来就mydata.y,mydata.mask
                            
                            

    def __len__(self):
        return self.sam_num  #index根据__len__(self)的返回值得到

    '''在类中定义函数,第一参数永远是类的本身实例变量self，并且调用时，不用传递该参数
       在类中调用类中定义的函数，需要在前加self(类的本身实例变量)'''

    def plot_input_ch(self,input_4ch):
        '''check input_4ch中的各类输入'''
        print('检验四通道输入')
        fig = plt.figure(figsize=(20,16),dpi=300)
        x = np.linspace(1,self.sam_len,self.sam_len)
        ch_lis = ['ori','d4','d8','d16']
        for iter in range(len(self.fea_litho)):
            ############################ 去掉SP，下面图个数固定了，改变fea个数时需要注意整改  #############################  
            ax0 = fig.add_subplot(2,2,iter+1)
            ############################ 去掉SP，下面图个数固定了，改变fea个数时需要注意整改  #############################  
            input_fea = input_4ch[iter,:,:]
            for i in range(input_fea.shape[0]):
                plt.plot(x,input_fea[i,:],label=ch_lis[i])
            ax0.set_title(self.fea_litho[iter],fontsize=15)
            plt.legend(loc='upper right',fontsize=15)
        plt.tight_layout()
        plt.savefig('/home/cldai/sciresearch/logcomplete/figure/loda_data_check/input_4ch.jpg')


    def resample(self,input,down):
        '''将input进行下采样，再样条插值回去'''
        '''para: down:下采样倍数'''

        #用block_reduce下采样
        input_d = block_reduce(input,block_size=(1,down), func=np.median)  #input_d即input_down   #cval=np.median(input) 是当非整除时，以cval来填充
        #print('input_d.shape',input_d.shape)
        
        input_ip = np.zeros((input_d.shape[0], self.sam_len))  #input_ip is input_interp
        for i in range(input_d.shape[0]):
            input_ip[i,:] = self.interp(input_d[i,:],i,down)

        #print('np.isnan(input_ip).any()  True说明有nan值: ',np.isnan(input_ip).any())
        return input_ip


    def interp(self,seq,i,down):
        #print('len(seq)',len(seq))
        x = np.linspace(1,len(seq),len(seq))
        y = seq
        #f是一个关于x和y的样条函数，这样再利用f来插值。'zero', 'slinear', 'quadratic', 'cubic'分别对应0，1，2，3次样条函数
        f = interp1d(x,y,kind='quadratic')   #二次样条函数
        xnew = np.linspace(1,len(seq),self.sam_len)
        ynew = f(xnew)

        # #
        # fig = plt.figure(figsize=(10,6),dpi=300)
        # plt.plot(x,y,label=self.fea_litho[i]+'-d'+str(down)+'-before interp')
        # plt.plot(xnew,ynew,'r',label=self.fea_litho[i]+'-d'+str(down)+'-after interp')
        # plt.legend(loc='upper right',fontsize=10)
        # plt.tight_layout()
        # plt.savefig('/home/cldai/sciresearch/logcomplete/figure/loda_data_check/'+self.fea_litho[i]+'-d'+str(down)+'.png')

        return ynew

    def construct_edge(self,input):
        '''
        构造边
        '''
        fea_n = len(self.fea_litho)
        #fea_n : n of node
        pos = np.zeros([fea_n*(fea_n-1), 2])
        k = 0
        for i in range(fea_n**2):
            if int(i/fea_n) != int(i % fea_n):
                pos[k, 1] = int(i/fea_n)
                pos[k, 0] = int(i % fea_n)
                k += 1
        pos_all = pos.copy()
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        pos_all_df = pd.DataFrame(pos_all,columns=['source','des'])
        edge_index_list = pos_all_df.index.values
        edge_index_list = np.array(edge_index_list)
        edge_index_list = edge_index_list.T
        #print('edge_index_list',edge_index_list)
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        pos_all = pos_all.T
        #print('pos_all',pos_all)
        #print('pos42',pos)

        a= input.sum(axis=0)
        #print('a',a)
        loc=np.argwhere(a==0)
        #print('loc',loc) 
        loc = loc.flatten()  
        #print('loc',loc)  

        posdf = pd.DataFrame(pos,columns=['source','des'])
        #print(posdf)
        for iter in range(len(loc)):
            posdf = posdf[posdf['source'] != loc[iter]].copy()
            #print('delete 后的pos ',posdf)

        ###########  分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        #获取posdf的index，存入作为edge_index_ini_list，再变换为numpy,并转置，
        edge_index_ini_list = posdf.index.values
        edge_index_ini_list = np.array(edge_index_ini_list)
        edge_index_ini_list = edge_index_ini_list.T
        #print('edge_index_ini_list',edge_index_ini_list)
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########

        pos = np.array(posdf)
        pos = pos.T
        #print('pos',pos)
        return pos, edge_index_ini_list, pos_all, edge_index_list

    def precond(self,well_data):
        '''去除一些异常数据'''
        '''由于现在没有SP和RMED RDEP数据，所以暂无需precond'''
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        #well_data['SP'][well_data['SP']<-350] = np.nan
        #well_data['RMED'][well_data['RMED']<0] = 0  #岩性分类GIR队伍中是将此变为0
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        return well_data

    def nullNum_by_col(self,df):
        '''用于检验是否有null值,同时也查看了columns是否按照fea_litho那样排列的（yes）'''
        nullNum=df.isnull().sum()
        return nullNum
    
    def depth_loc(self,a):
        b = int((a-136.086)/0.152)  #元坝起始点深度3618.0，0.125；原norway136.086，0.152
        return b


    def label_mask0(self,df):
        '''
        选取随机的起始点，
        制作单个label和对应的初始mask0，该mask0参与loss，不需要恢复
        '''
        awmd = df['DEPTH_MD'].copy()   #awmd is all_wells_md
        awfea = df[self.fea_litho].copy()   #awfea is all_wells_fea
        #print('awfea.shape[0]',awfea.shape[0])
        #awfea = self.precond(awfea)
        
        #选取随机起始点
        stap=np.random.randint(0,len(awfea)-self.sam_len)  #stap表示start_point
        #不选拼接处的数据
        while(self.maskc[stap]==0):
            stap=np.random.randint(0,len(awfea)-self.sam_len)
        endp=stap+self.sam_len  #endp表示end_point
        

        i=0
        while(self.loc_li[i]<stap):   # and stap<loc_li[i+1]
            #print('self.loc_li[i]',self.loc_li[i])
            if stap<self.loc_li[i+1]:
                #下面这个按照深度顺序
                sam_stap = self.depth_loc(awmd.iloc[stap]) 
                sam_endp = self.depth_loc(awmd.iloc[stap + self.sam_len])    # - self.loc_li[i]
                well_order = i
                well_stap = self.depth_loc(awmd.iloc[self.loc_li[i]+1])  
                well_endp = self.depth_loc(awmd.iloc[self.loc_li[i+1]])  
                #print('111111')
            i+=1
        #print('Time:{:.3f}'.format(b-a))

        #制作单个label, label*mask0=sample
        label_ori=awfea.iloc[stap:endp,:].copy()
        #self.plot_label(label,'label_before_norm')

        ##将对应位置的参考样本选取出来，进而用参考样本的相关数值作均方归一化
        label, np_mean, np_std = self.mean_var_norm(label_ori)
        #self.plot_label(label,'label_after_norm')

        #制作初始的mask0
        mask0=np.where(pd.isnull(label), 0, 1)  #label中null值赋0，nonnulll赋值1  #mask0为numpy
        #print('mask0.shape,mask0',mask0.shape,mask0) 
        label_np=np.where(pd.isnull(label), 0, label)
        label_ori = np.where(pd.isnull(label_ori), 0, label_ori)

        return label_ori, label_np, mask0, sam_stap, sam_endp, well_order, well_stap, well_endp, np_mean, np_std

    def plot_label(self, label, name):
        print('绘制label')
        figwidth = len(self.fea_litho)
        fig, axes = plt.subplots(1,label.shape[1], figsize=(3*4,16), sharex=False,
                            sharey=True,dpi=300)
        x = np.linspace(1,self.sam_len,self.sam_len)
        for i in range(figwidth):
            axes[i].plot(label.iloc[:,i],x,color='k')
            axes[i].set_title(self.fea_litho[i],fontsize=25)
        axes[0].set_ylim(self.sam_len,1)
        plt.tight_layout()
        plt.savefig('/home/cldai/sciresearch/logcomplete/figure/loda_data_check/'+str(name)+'.jpg',
                    dpi=300,bbox_inches='tight')


    def mean_var_norm(self, label):
        df1_mean=label.mean()
        df1_mean=np.where(pd.isnull(df1_mean), 0, df1_mean)
        #print('df1_mean',df1_mean.shape,df1_mean)
        df1_std=label.std(ddof=0)  #避免std使用index=0的影响
        df1_std=np.where(pd.isnull(df1_std), 0, df1_std)
        #print('df1_std.shape',df1_std.shape, df1_std)

        label_mean_var_norm=(label-df1_mean)/df1_std
        
        #由于进行gardner比较时，需要用到有实际物理意义的数据，所以这里返回numpy格式的均值和方差
        #直接np.array(df1_mean)时，其shape=(4,) 是一维数组
        np_mean = np.array(df1_mean).reshape((1,len(self.fea_litho)))   #np_mean.shape=(1,4)
        np_std = np.array(df1_std).reshape((1,len(self.fea_litho)))
        return label_mean_var_norm, np_mean, np_std

    def mask_d(self,label,mask0):
        '''制作随机的maskd（需要恢复的），起始点随机'''
        maskd=np.ones((label.shape[0],label.shape[1]))
        a=maskd.shape[0]  #a样本长度，64
        b=maskd.shape[1]  #b特征曲线数目，7

        '''check mask0的哪些列为0，避免让后续的maskd与mask0重合'''
        m0_loc= mask0.sum(axis=0)
        #print('m0_loc',m0_loc)
        loc=np.argwhere(m0_loc==0)
        #print('loc',loc) 
        loc = loc.flatten()  
        #print('loc',loc)  

        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        '''用于六条曲线时'''
        # if mask0.sum()<a*(b/2):  #意为mask0多于一半
        #     maskd=maskd
        # elif mask0.sum()<=a*(4*b/5):  #当mask0少于4/5时，给1/3的条mask
        #     num_maskd=int(b/3)              
        #     maskd=self.mask_d_loc(a,b,num_maskd,maskd)
        # else:                                             
        #     num_maskd=int(b/2)                            
        #     maskd=self.mask_d_loc(a,b,num_maskd,maskd)

        ''''用于4条曲线时'''
        if mask0.sum() <= a*(b/2):  #意为mask0多于一半
            maskd=maskd
        
        else:                                             
            num_maskd=1  #3条或者4条时只给1条maskd                            
            maskd=self.mask_d_loc(a,b,num_maskd,maskd,loc)
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################

        return maskd


    def mask_d_loc(self,a,b,num_maskd,maskd,loc): 
        '''注意是训练还是测试，num选择性注释'''
        num=np.random.randint(b,size=num_maskd)  #生成num_maskd个上限为b-1的随机整数，这里b=7，num：list
        # num=[3]   #这样写只是暂时测试指定缺少谁，训练时要记住改回来
        '''因为nu列表只有一个参数，所以直接写了num[0]'''
        '''利用set函数判断是否为子集,<=子集;<真子集'''
        while(set(num) <= set(loc) or self.fea_litho.index('GR') in num):
            num=np.random.randint(b,size=num_maskd)
        #print('while后num',num)
        for i in range(num_maskd):  
            maskd[:,num[i]]=0
        return maskd
#########  ##################  #############  #############







######### MyGNNDataset for test ################## MyGNNDataset for test #############  #############
class MyGNNDataset_test(gdata.Dataset):
    #print('############ MyGNNDataset_test')
    def __init__(self, data, sam_len, sam_num, fea_litho, norm_type, mask_logs,
                null_sequence, sam_fixed, filt_size,fix_well,fix_stap,):
        gdata.Dataset.__init__(self)
        self.signal = data  #这里data和self.signal是dataframe
        self.sam_len = sam_len
        self.sam_num = sam_num
        self.fea_litho = fea_litho
        self.norm_type = norm_type
        self.mask_logs = mask_logs
        self.null_sequence = null_sequence
        self.sam_fixed = sam_fixed
        self.filt_size = filt_size
        self.fix_well = fix_well
        self.fix_stap = fix_stap
        #self.mean_log = mean_log

    def __getitem__(self,index):
        '''完成标签集、数据集、mask集的制作'''
        print('运行的是MyGNNDataset_test,测试集的')

        well_data, well_md=self.random_single_well(self.signal, self.fea_litho) #random_single_well输入df返回DataFrame数据类型
        label_intact_ori, label_intact_df, np_mean, np_std = self.label_for_intact(well_data,well_md)  #输入df,返回df
        #print('label_intact_df',label_intact_df.isnull().sum())
        label_null = self.label_mask_log(label_intact_df)  #mask_log输入dataframe返回df数据类型
        label_ori_null = self.label_mask_log(label_intact_ori)
        label_ori, label, mask0=self.label_mask0(label_null, label_ori_null)  #label_mask0输入dataframe返回numpy数据类型
        #print('label',label)
        #print('mask0',mask0)
        maskd=np.ones((label.shape[0],label.shape[1]))   #作测试集时实际上就不需要maskd去掩盖了
        
        #制作对应的sample
        input_old=label*maskd  #label*maskd作为输入input
        input_ori = label_ori*maskd

        #将input转置为后面符合numpy的按行进行下采样
        input = input_old.T
        input_d4 = self.resample(input,4)
        input_d8 = self.resample(input,8)
        input_d16 = self.resample(input,16)
        # input_d4 = self.resample(input,8)  
        # input_d8 = self.resample(input,16)
        # input_d16 = self.resample(input,32)

        #将二维拓展为三维并在通道维度上cat。拓维方法有[:, np.newaxis/None,:],  reshape(4,1,512); np.expand_dims(arr,axis=1(1意为拓展后在意为上))
        input_3dim = input[:,np.newaxis,:]
        d4_3dim = input_d4[:,np.newaxis,:]   #注意这种改法input_d4与d4_3dim内存是不共享的，即input_d4 shape =(4,512)不变
        d8_3dim = input_d8[:,np.newaxis,:]
        d16_3dim = input_d16[:,np.newaxis,:]

        #np.concatenate拼接成四通道数据
        input_4ch = np.concatenate((input_3dim,d4_3dim,d8_3dim,d16_3dim),axis=1)
        
        #画input_4ch图
        #self.plot_input_ch(input_4ch)

        label_intact=np.array(label_intact_df)
        #print('label_intact_df',label_intact)

        edge_index_ini, edge_ini_list, edge_index, edge_list = self.construct_edge(input_old)

        label_intact_ori = np.array(label_intact_ori)
        label_ori_com = torch.tensor(label_intact_ori.T, dtype=torch.float)
        x = torch.tensor(input_4ch, dtype=torch.float)  #index为了索引对应的batch
        x_ori = torch.tensor(input_ori.T, dtype=torch.float)
        label = torch.tensor(label.T, dtype=torch.float)
        mask0_1 = torch.tensor(mask0.T, dtype=torch.float)
        maskd = torch.tensor(maskd.T, dtype=torch.float)
        edge_index_ini = torch.tensor(edge_index_ini, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        label_intact = torch.tensor(label_intact.T,dtype=torch.float)
        norm_mean = torch.tensor(np_mean.T, dtype=torch.float)
        norm_std = torch.tensor(np_std.T, dtype=torch.float)

        return gdata.Data(x=x, edge_index_ini=edge_index_ini, edge_index=edge_index, y=label, mask=mask0_1, maskd=maskd, 
                            label_intact=label_intact,
                            edge_ini_list=edge_ini_list, edge_list=edge_list,
                            norm_mean=norm_mean, norm_std=norm_std,
                            label_ori = label_ori_com, x_ori=x_ori)  #应该这里写y,mask,后面eg:mydata调用出来就mydata.y,mydata.mask

    def __len__(self):
        return self.sam_num  #index根据__len__(self)的返回值得到。 __len__返回值用于决定生成多少个样本

    '''在类中定义函数,第一参数永远是类的本身实例变量self，并且调用时，不用传递该参数
       在类中调用类中定义的函数，需要在前加self(类的本身实例变量)'''

    def plot_input_ch(self,input_4ch):
        '''check input_4ch中的各类输入'''
        print('检验四通道输入')
        fig = plt.figure(figsize=(20,16),dpi=300)
        x = np.linspace(1,self.sam_len,self.sam_len)
        ch_lis = ['ori','d4','d8','d16']
        for iter in range(len(self.fea_litho)):
            ############################ 去掉SP，下面图个数固定了，改变fea个数时需要注意整改  #############################  
            ax0 = fig.add_subplot(2,2,iter+1)
            ############################ 去掉SP，下面图个数固定了，改变fea个数时需要注意整改  #############################  
            input_fea = input_4ch[iter,:,:]
            for i in range(input_fea.shape[0]):
                plt.plot(x,input_fea[i,:],label=ch_lis[i])
            ax0.set_title(self.fea_litho[iter],fontsize=15)
            plt.legend(loc='upper right',fontsize=15)
        plt.tight_layout()
        plt.savefig('/home/cldai/sciresearch/logcomplete/figure/loda_data_check/test/input_4ch.png')

    def resample(self,input,down):
        '''将input进行下采样，再样条插值回去'''
        '''para: down:下采样倍数'''

        #用block_reduce下采样
        input_d = block_reduce(input,block_size=(1,down), func=np.median)  #input_d即input_down   #cval=np.median(input) 是当非整除时，以cval来填充
        #print('input_d.shape',input_d.shape)
        
        input_ip = np.zeros((input_d.shape[0], self.sam_len))  #input_ip is input_interp
        for i in range(input_d.shape[0]):
            input_ip[i,:] = self.interp(input_d[i,:],i,down)

        #print('np.isnan(input_ip).any()  True说明有nan值: ',np.isnan(input_ip).any())
        return input_ip
    
    def interp(self,seq,i,down):
        #print('len(seq)',len(seq))
        x = np.linspace(1,len(seq),len(seq))
        y = seq
        #f是一个关于x和y的样条函数，这样再利用f来插值。'zero', 'slinear', 'quadratic', 'cubic'分别对应0，1，2，3次样条函数
        f = interp1d(x,y,kind='quadratic')   #二次样条函数
        xnew = np.linspace(1,len(seq),self.sam_len)
        ynew = f(xnew)

        # #
        # fig = plt.figure(figsize=(10,6),dpi=300)
        # plt.plot(x,y,label=self.fea_litho[i]+'-d'+str(down)+'-before interp')
        # plt.plot(xnew,ynew,'r',label=self.fea_litho[i]+'-d'+str(down)+'-after interp')
        # plt.legend(loc='upper right',fontsize=10)
        # plt.tight_layout()
        # plt.savefig('/home/cldai/sciresearch/logcomplete/figure/loda_data_check/test/'+self.fea_litho[i]+'-d'+str(down)+'.png')

        return ynew

    def construct_edge(self,input):
        '''
        构造边
        '''
        fea_n = len(self.fea_litho)
        #fea_n : n of node
        pos = np.zeros([fea_n*(fea_n-1), 2])
        k = 0
        for i in range(fea_n**2):
            if int(i/fea_n) != int(i % fea_n):
                pos[k, 1] = int(i/fea_n)
                pos[k, 0] = int(i % fea_n)
                k += 1
        pos_all = pos.copy()
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        pos_all_df = pd.DataFrame(pos_all,columns=['source','des'])
        edge_index_list = pos_all_df.index.values
        edge_index_list = np.array(edge_index_list)
        edge_index_list = edge_index_list.T
        #print('edge_index_list',edge_index_list)
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        pos_all = pos_all.T
        #print('pos_all',pos_all)
        #print('pos42',pos)

        a= input.sum(axis=0)
        #print('a',a)
        loc=np.argwhere(a==0)
        #print('loc',loc) 
        loc = loc.flatten()  
        #print('loc',loc)  


        posdf = pd.DataFrame(pos,columns=['source','des'])
        #print(posdf)
        for iter in range(len(loc)):
            posdf = posdf[posdf['source'] != loc[iter]].copy()
            #print('delete 后的pos ',posdf)

        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        #获取posdf的index，存入作为edge_index_ini_list，再变换为numpy,并转置，
        edge_index_ini_list = posdf.index.values
        edge_index_ini_list = np.array(edge_index_ini_list)
        edge_index_ini_list = edge_index_ini_list.T
        #print('edge_index_ini_list',edge_index_ini_list)
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########

        pos = np.array(posdf)
        pos = pos.T
        #print('pos',pos)
        return pos, edge_index_ini_list, pos_all, edge_index_list

    def random_single_well(self, df,fea_litho):
        '''取随机单口井，取需要的feature'''
        '''如果不想要随机井，可以把order替换乘固定值'''
        '''三个文件中井的采样间隔为0.152m'''
        
        all_well_name=df['WELL'].unique()  #df.WELL.unique()也是正确的  #wellname是numpy.ndarry
        #print('!!!!!!!!!!!!! all_well_name',all_well_name)  #(1170511, 30) 
        order = np.random.randint(0,len(all_well_name))

        if self.sam_fixed==True:
            ################ 固定order固定了使用某口井，取了'15/9-14'井 ################### 固定order固定了使用某口井 ###################
            well_data = df[df['WELL'] == self.fix_well]  
            print('$$ 固定 人工 选取的井为：',well_data['WELL'].unique(),   '该井长度为：',len(well_data))
            ################ 固定order固定了使用某口井 ################### 固定order固定了使用某口井 ################### 
        else:
            well_data = df[df['WELL'] == all_well_name[order]]
            print('随机选择 井：',all_well_name[order])

        well_md = well_data[['DEPTH_MD']].copy()
        well_data=well_data[fea_litho].copy()
        #print('well_data.shape',well_data.shape)

        #well_data = self.precond(well_data)

        fea_null=self.nullNum_by_col(well_data)
        print('fea_null',fea_null)
            
        return well_data, well_md

    def precond(self,well_data):
        '''去除一些异常数据'''
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        #well_data['SP'][well_data['SP']<-350] = np.nan
        #well_data['RMED'][well_data['RMED']<0] = 0  #岩性分类GIR队伍中是将此变为0
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        return well_data

    def nullNum_by_col(self,df):
        '''用于检验是否有null值,同时也查看了columns是否按照fea_litho那样排列的（yes）'''
        nullNum=df.isnull().sum()
        return nullNum

    def label_for_intact(self, well_data,well_md):

        #寻找含有null值的行和列
        for columname in well_data.columns:    #在numpy和pandas中,null和nan是算在长度里的，也算在shape里的
            if well_data[columname].count() != len(well_data):
                loc = well_data[columname][well_data[columname].isnull().values==True].index.tolist()
                print('列名："{}", 第{}行位置有缺失值'.format(columname,loc))
        #print('loc',loc)

        #选取随机起始点
        if self.sam_fixed==True:
            stap=self.fix_stap
            endp=stap+self.sam_len
            print('选取的起始点：', stap)
        else:
            stap=np.random.randint(0,len(well_data)-self.sam_len)  #stap表示start_point
            endp=stap+self.sam_len  #endp表示end_point

        #制作单个label, label*mask0=sample
        label_intact_ori=well_data.iloc[stap:endp,:]

        print('测试样本起始深度', well_md.iloc[stap])
        ##将对应位置的参考样本选取出来，进而用参考样本的相关数值作均方归一化
        #mean_log_fea = self.mean_log[self.fea_litho].copy()
        #sam_stap = self.depth_loc(well_md.iloc[stap]) 
        #refer_sam = mean_log_fea.iloc[sam_stap:(sam_stap+self.sam_len),:].copy()
        label_intact, np_mean, np_std = self.mean_var_norm(label_intact_ori)

        return label_intact_ori, label_intact, np_mean, np_std

    def depth_loc(self,a):
        b = int((a-136.086)/0.152)  #元坝起始点深度3618.0，0.125；原norway136.086，0.152
        return b

    def mean_var_norm(self, label):
        df1_mean=label.mean()
        print('df1_mean',df1_mean.shape,df1_mean)
        df1_std=label.std(ddof=0)  #避免std使用index=0的影响
        print('df1_std.shape',df1_std.shape, df1_std)

        label_mean_var_norm=(label-df1_mean)/df1_std

        #由于进行gardner比较时，需要用到有实际物理意义的数据，所以这里返回numpy格式的均值和方差
        #直接np.array(df1_mean)时，其shape=(4,) 是一维数组
        np_mean = np.array(df1_mean).reshape((1,len(self.fea_litho)))   #np_mean.shape=(1,4)
        np_std = np.array(df1_std).reshape((1,len(self.fea_litho)))
        return label_mean_var_norm, np_mean, np_std


    def label_mask_log(self,label_data):
        '''传入列表来自定义要mask某条或者某些log全序列，或者随机mask掉log上的部分片段
        并且返回七个log完整的label'''
        label_null=label_data.copy()
        if self.null_sequence==True:
            label_null[self.mask_logs]=np.nan
        else:
            mask_frag=self.mask_log_fragement(label_data)  #mask_log_fragement输入df,返回label的df
            label_null=np.where(mask_frag==0,np.nan,label_data)
            label_null=pd.DataFrame(label_null)
        return label_null

    # def mask_log_fragement(self,label_data):
        
    #     mlf=np.ones((label_data.shape[0],label_data.shape[1]))  #mlf表示mask_log_fragements
    #     a=mlf.shape[0]  #a样本长度，64
    #     b=mlf.shape[1]  #b样本宽度，7
    #     num_mlf=int(b/3)  #int是向下取整   
    #     mask_frag=self.mask_d_loc(a,b,num_mlf,mlf) 

    #     return mask_frag

    # def mask_d_loc(self,a,b,num_maskd,maskd): 
    #     num=np.random.randint(b,size=num_maskd)  #生成b个上限为b-1的随机整数，这里b=7，num：array
    #     #print('num',num)
    #     #print("self.fea_litho.index('GR')",self.fea_litho.index('GR'))
    #     while(self.fea_litho.index('GR') in num):  #保证num中无GR对应的index，进而使GR不被mask
    #         num=np.random.randint(b,size=num_maskd)
    #         #print('while后num',num)
    #     for i in range(num_maskd):  
    #         #保证缺失段长度至少为全长的1/4
    #         stap=np.random.randint(0,2*a/3)
    #         endp=np.random.randint(stap+a/3,a)
    #         maskd[stap:endp,num[i]]=0
    #     return maskd

    def label_mask0(self, label_null, label_ori_null):
            
        #制作初始的mask0
        mask0=np.where(pd.isnull(label_null), 0, 1)  #label中null值赋0，nonnulll赋值1  #mask0为numpy
        #print('mask0.shape,mask0',mask0.shape,mask0) 
        label_np=np.where(pd.isnull(label_null), 0, label_null)
        label_ori=np.where(pd.isnull(label_ori_null), 0, label_ori_null)

        return label_ori, label_np, mask0

    # def mask_d(self,label):
    #     '''制作随机的maskd（需要恢复的），起始点随机'''
    #     maskd=np.ones((label.shape[0],label.shape[1]))
    #     a=maskd.shape[0]  #a样本长度，64
    #     b=maskd.shape[1]  #b样本宽度，7
    #     num_maskd=int(b/6)   
    #     mask_frag=self.mask_d_loc(self,a,b,num_maskd,maskd)
       
    #     return mask_frag
######### MyGNNDataset for test ################## MyGNNDataset for test #############  #############










############# MyGNNDataset for cat sample  ############### MyGNNDataset for cat sample  ##################
class MyGNNDataset_cat_sam(gdata.Dataset):
    #print('############ MyGNNDataset_test')
    def __init__(self, data, sam_len, sam_num, fea_litho, norm_type, mask_logs,
                null_sequence, sam_fixed, filt_size,wo,wep,fix_stap):
        gdata.Dataset.__init__(self)
        self.signal = data  #这里data和self.signal是dataframe
        self.sam_len = sam_len
        self.sam_num = sam_num
        self.fea_litho = fea_litho
        self.norm_type = norm_type
        self.mask_logs = mask_logs
        self.null_sequence = null_sequence
        self.sam_fixed = sam_fixed
        self.filt_size = filt_size
        self.wo = wo
        self.wep = wep
        self.fix_stap = fix_stap
        #self.mean_log = mean_log

    def __getitem__(self,index):
        '''完成标签集、数据集、mask集的制作'''
        #print('运行的是MyGNNDataset_cat_sam,测试集的')

        well_data,md_stap, md_endp, md=self.random_single_well(self.signal) #random_single_well输入df返回DataFrame数据类型
        label_intact_ori, label_intact_df, np_mean, np_std = self.label_for_intact(well_data, md)  #输入df,返回df
        #print('label_intact_df',label_intact_df.isnull().sum())
        label_null = self.label_mask_log(label_intact_df)  #mask_log输入dataframe返回df数据类型
        label_ori_null = self.label_mask_log(label_intact_ori)
        label_ori, label, mask0=self.label_mask0(label_null, label_ori_null)  #label_mask0输入dataframe返回numpy数据类型
        maskd=np.ones((label.shape[0],label.shape[1]))   #作测试集时实际上就不需要maskd去掩盖了
        
        #制作对应的sample
        input_old=label*maskd  #label*maskd作为输入input
        input_ori = label_ori*maskd

        #将input转置为后面符合numpy的按行进行下采样
        input = input_old.T
        input_d4 = self.resample(input,4)
        input_d8 = self.resample(input,8)
        input_d16 = self.resample(input,16)
        # input_d4 = self.resample(input,8)  
        # input_d8 = self.resample(input,16)
        # input_d16 = self.resample(input,32)

        #将二维拓展为三维并在通道维度上cat。拓维方法有[:, np.newaxis/None,:],  reshape(4,1,512); np.expand_dims(arr,axis=1(1意为拓展后在意为上))
        input_3dim = input[:,np.newaxis,:]
        d4_3dim = input_d4[:,np.newaxis,:]   #注意这种改法input_d4与d4_3dim内存是不共享的，即input_d4 shape =(4,512)不变
        d8_3dim = input_d8[:,np.newaxis,:]
        d16_3dim = input_d16[:,np.newaxis,:]

        #np.concatenate拼接成四通道数据
        input_4ch = np.concatenate((input_3dim,d4_3dim,d8_3dim,d16_3dim),axis=1)
        
        #画input_4ch图
        #self.plot_input_ch(input_4ch)

        edge_index_ini, edge_ini_list, edge_index, edge_list = self.construct_edge(input_old)


        # label_intact=np.array(label_intact_df)
        # #print('label_intact_df',label_intact)
        # label_intact_ori = np.array(label_intact_ori)
        '''将label_intact_ori就赋给label_intact，减少再反归一化这一步骤'''
        label_intact=np.array(label_intact_ori)
        
        x = torch.tensor(input_4ch, dtype=torch.float)  #index为了索引对应的batch
        x_ori = torch.tensor(input_ori.T, dtype=torch.float)

        label = torch.tensor(label.T, dtype=torch.float)
        mask0_1 = torch.tensor(mask0.T, dtype=torch.float)
        maskd = torch.tensor(maskd.T, dtype=torch.float)
        edge_index_ini = torch.tensor(edge_index_ini, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        label_intact = torch.tensor(label_intact.T,dtype=torch.float)
        md_stap = torch.tensor(md_stap, dtype=torch.float)
        md_endp = torch.tensor(md_endp, dtype=torch.float)
        norm_mean = torch.tensor(np_mean.T, dtype=torch.float)
        norm_std = torch.tensor(np_std.T, dtype=torch.float)
        #print('np_mean.shape',np_mean.shape)

        return gdata.Data(x=x, edge_index_ini=edge_index_ini, edge_index=edge_index, y=label, mask=mask0_1, maskd=maskd, 
                            label_intact=label_intact, md_stap=md_stap, 
                            md_endp=md_endp,
                            edge_ini_list=edge_ini_list, edge_list=edge_list,
                            norm_mean = norm_mean, norm_std=norm_std,
                            label_ori = label_ori, 
                            x_ori=x_ori
                            )  #应该这里写y,mask,后面eg:mydata调用出来就mydata.y,mydata.mask

    def __len__(self):
        return self.sam_num  #index根据__len__(self)的返回值得到。 __len__返回值用于决定生成多少个样本

    '''在类中定义函数,第一参数永远是类的本身实例变量self，并且调用时，不用传递该参数
       在类中调用类中定义的函数，需要在前加self(类的本身实例变量)'''

    def plot_input_ch(self,input_4ch):
        '''check input_4ch中的各类输入'''
        print('检验四通道输入')
        fig = plt.figure(figsize=(20,16),dpi=300)
        x = np.linspace(1,self.sam_len,self.sam_len)
        ch_lis = ['ori','d4','d8','d16']
        for iter in range(len(self.fea_litho)):
            ############################ 去掉SP，下面图个数固定了，改变fea个数时需要注意整改  #############################  
            ax0 = fig.add_subplot(2,2,iter+1)
            ############################ 去掉SP，下面图个数固定了，改变fea个数时需要注意整改  #############################  
            input_fea = input_4ch[iter,:,:]
            for i in range(input_fea.shape[0]):
                plt.plot(x,input_fea[i,:],label=ch_lis[i])
            ax0.set_title(self.fea_litho[iter],fontsize=15)
            plt.legend(loc='upper right',fontsize=15)
        plt.tight_layout()
        plt.savefig('/home/cldai/sciresearch/logcomplete/figure/loda_data_check/test/input_4ch.png')

    def resample(self,input,down):
        '''将input进行下采样，再样条插值回去'''
        '''para: down:下采样倍数'''

        #用block_reduce下采样
        input_d = block_reduce(input,block_size=(1,down), func=np.median)  #input_d即input_down   #cval=np.median(input) 是当非整除时，以cval来填充
        #print('input_d.shape',input_d.shape)
        
        input_ip = np.zeros((input_d.shape[0], self.sam_len))  #input_ip is input_interp
        for i in range(input_d.shape[0]):
            input_ip[i,:] = self.interp(input_d[i,:],i,down)

        #print('np.isnan(input_ip).any()  True说明有nan值: ',np.isnan(input_ip).any())
        return input_ip
    
    def interp(self,seq,i,down):
        #print('len(seq)',len(seq))
        x = np.linspace(1,len(seq),len(seq))
        y = seq
        #f是一个关于x和y的样条函数，这样再利用f来插值。'zero', 'slinear', 'quadratic', 'cubic'分别对应0，1，2，3次样条函数
        f = interp1d(x,y,kind='quadratic')   #二次样条函数
        xnew = np.linspace(1,len(seq),self.sam_len)
        ynew = f(xnew)

        # #
        # fig = plt.figure(figsize=(10,6),dpi=300)
        # plt.plot(x,y,label=self.fea_litho[i]+'-d'+str(down)+'-before interp')
        # plt.plot(xnew,ynew,'r',label=self.fea_litho[i]+'-d'+str(down)+'-after interp')
        # plt.legend(loc='upper right',fontsize=10)
        # plt.tight_layout()
        # plt.savefig('/home/cldai/sciresearch/logcomplete/figure/loda_data_check/test/'+self.fea_litho[i]+'-d'+str(down)+'.png')

        return ynew

    def construct_edge(self,input):
        '''
        构造边
        '''
        fea_n = len(self.fea_litho)
        #fea_n : n of node
        pos = np.zeros([fea_n*(fea_n-1), 2])
        k = 0
        for i in range(fea_n**2):
            if int(i/fea_n) != int(i % fea_n):
                pos[k, 1] = int(i/fea_n)
                pos[k, 0] = int(i % fea_n)
                k += 1
        pos_all = pos.copy()
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        pos_all_df = pd.DataFrame(pos_all,columns=['source','des'])
        edge_index_list = pos_all_df.index.values
        edge_index_list = np.array(edge_index_list)
        edge_index_list = edge_index_list.T
        #print('edge_index_list',edge_index_list)
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        pos_all = pos_all.T
        #print('pos_all',pos_all)
        #print('pos42',pos)

        a= input.sum(axis=0)
        #print('a',a)
        loc=np.argwhere(a==0)
        #print('loc',loc) 
        loc = loc.flatten()  
        #print('loc',loc)  


        posdf = pd.DataFrame(pos,columns=['source','des'])
        #print(posdf)
        for iter in range(len(loc)):
            posdf = posdf[posdf['source'] != loc[iter]].copy()
            #print('delete 后的pos ',posdf)

        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        #获取posdf的index，存入作为edge_index_ini_list，再变换为numpy,并转置，
        edge_index_ini_list = posdf.index.values
        edge_index_ini_list = np.array(edge_index_ini_list)
        edge_index_ini_list = edge_index_ini_list.T
        #print('edge_index_ini_list',edge_index_ini_list)
        ########### 分层设置边按需连接或全连接  ########### 分层设置边按需连接或全连接  ###########
        pos = np.array(posdf)
        pos = pos.T
        #print('pos',pos)
        return pos, edge_index_ini_list, pos_all, edge_index_list

    def random_single_well(self, df):
        '''取随机单口井，取需要的feature'''
        '''如果不想要随机井，可以把order替换乘固定值'''
        '''三个文件中井的采样间隔为0.152m'''
        all_well_name=df['WELL'].unique()  #df.WELL.unique()也是正确的  #wellname是numpy.ndarry
        #print('all_well_name',all_well_name)
        well_data = df[df['WELL'] == all_well_name[self.wo]]
        #well_name = all_well_name[self.wo]
        
        md = well_data[['DEPTH_MD']].copy()
        md_stap = md.iloc[0]
        #print('md_stap',md_stap)
        md_endp = md.iloc[self.wep]
        #print('md_endp',md_endp)

        well_data=well_data[self.fea_litho].copy()
        #well_data = self.precond(well_data)
            
        return well_data, md_stap, md_endp, md

    def precond(self,well_data):
        '''去除一些异常数据'''
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        #well_data['SP'][well_data['SP']<-350] = np.nan
        #well_data['RMED'][well_data['RMED']<0] = 0  #岩性分类GIR队伍中是将此变为0
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        return well_data

    def nullNum_by_col(self,df):
        '''用于检验是否有null值,同时也查看了columns是否按照fea_litho那样排列的（yes）'''
        nullNum=df.isnull().sum()
        return nullNum

    def label_for_intact(self, well_data, well_md):

        #选取随机起始点
        if self.sam_fixed==True:
            stap=self.fix_stap
            endp=stap+self.sam_len
            print('选取的起始点：', stap)
        else:
            stap=np.random.randint(0,len(well_data)-self.sam_len)  #stap表示start_point
            endp=stap+self.sam_len  #endp表示end_point

        #制作单个label, label*mask0=sample
        label_intact_ori=well_data.iloc[stap:endp,:]

        ##将对应位置的参考样本选取出来，进而用参考样本的相关数值作均方归一化
        #mean_log_fea = self.mean_log[self.fea_litho].copy()
        #sam_stap = self.depth_loc(well_md.iloc[stap]) 
        #refer_sam = mean_log_fea.iloc[sam_stap:(sam_stap+self.sam_len),:].copy()
        label_intact, np_mean, np_std = self.mean_var_norm(label_intact_ori)
            
        return label_intact_ori, label_intact, np_mean, np_std

    def depth_loc(self,a):
        b = int((a-136.086)/0.152)  #元坝起始点深度3618.0，0.125；原norway136.086，0.152
        return b

    def mean_var_norm(self, label):
        df1_mean=label.mean()
        #print('df1_mean',df1_mean.shape,df1_mean)
        df1_std=label.std(ddof=0)  #避免std使用index=0的影响
        #print('df1_std.shape',df1_std.shape, df1_std)

        label_mean_var_norm=(label-df1_mean)/df1_std

        #由于进行gardner比较时，需要用到有实际物理意义的数据，所以这里返回numpy格式的均值和方差
        #直接np.array(df1_mean)时，其shape=(4,) 是一维数组
        np_mean = np.array(df1_mean).reshape((1,len(self.fea_litho)))   #np_mean.shape=(1,4)
        np_std = np.array(df1_std).reshape((1,len(self.fea_litho)))
        return label_mean_var_norm, np_mean, np_std


    def label_mask_log(self,label_data):
        '''传入列表来自定义要mask某条或者某些log全序列，或者随机mask掉log上的部分片段
        并且返回七个log完整的label'''
        label_null=label_data.copy()
        if self.null_sequence==True:
            label_null[self.mask_logs]=np.nan
            ##这里要将一些数据换成null
            label_null[['GR', 'NPHI']] = np.nan
        else:
            mask_frag=self.mask_log_fragement(label_data)  #mask_log_fragement输入df,返回label的df
            label_null=np.where(mask_frag==0,np.nan,label_data)
            label_null=pd.DataFrame(label_null)
        return label_null

    # def mask_log_fragement(self,label_data):
        
    #     mlf=np.ones((label_data.shape[0],label_data.shape[1]))  #mlf表示mask_log_fragements
    #     a=mlf.shape[0]  #a样本长度，64
    #     b=mlf.shape[1]  #b样本宽度，7
    #     num_mlf=int(b/3)  #int是向下取整   
    #     mask_frag=self.mask_d_loc(a,b,num_mlf,mlf) 

    #     return mask_frag

    # def mask_d_loc(self,a,b,num_maskd,maskd): 
    #     num=np.random.randint(b,size=num_maskd)  #生成b个上限为b-1的随机整数，这里b=7，num：array
    #     #print('num',num)
    #     #print("self.fea_litho.index('GR')",self.fea_litho.index('GR'))
    #     while(self.fea_litho.index('GR') in num):  #保证num中无GR对应的index，进而使GR不被mask
    #         num=np.random.randint(b,size=num_maskd)
    #         #print('while后num',num)
    #     for i in range(num_maskd):  
    #         #保证缺失段长度至少为全长的1/4
    #         stap=np.random.randint(0,2*a/3)
    #         endp=np.random.randint(stap+a/3,a)
    #         maskd[stap:endp,num[i]]=0
    #     return maskd

    def label_mask0(self, label_null, label_ori_null):
            
        #制作初始的mask0
        mask0=np.where(pd.isnull(label_null), 0, 1)  #label中null值赋0，nonnulll赋值1  #mask0为numpy
        #print('mask0.shape,mask0',mask0.shape,mask0) 
        label_np=np.where(pd.isnull(label_null), 0, label_null)
        label_ori=np.where(pd.isnull(label_ori_null), 0, label_ori_null)

        return label_ori, label_np, mask0

    # def mask_d(self,label):
    #     '''制作随机的maskd（需要恢复的），起始点随机'''
    #     maskd=np.ones((label.shape[0],label.shape[1]))
    #     a=maskd.shape[0]  #a样本长度，64
    #     b=maskd.shape[1]  #b样本宽度，7
    #     num_maskd=int(b/6)   
    #     mask_frag=self.mask_d_loc(self,a,b,num_maskd,maskd)
       
    #     return mask_frag
############# MyGNNDataset for cat sample  ############### MyGNNDataset for cat sample  ##################

