import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse   #用于解析命令行参数和选项的标准模块
import torch
import torch.nn as nn
import scipy.signal as signal
import seaborn as sn
from matplotlib import gridspec



def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sam',default=10000,help='需要计算相关度的样本数目')

    ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
    parser.add_argument('--fea_litho',default=['GR','RHOB', 'NPHI', 'DTC'],
                        type=list,help="feature and lithology that we need")
    ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################

    parser.add_argument('--filePath',default='/home/cldai/sciresearch/logcomplete/data/Teamdata/', 
                        help='file path of train and test')  #cldai add
    parser.add_argument('--sam_len',default=512, type=int, help='length of every sample')
    parser.add_argument('--norm_type',default='sample',type=str ,
                        help="数据归一化的类型,按样本(sample)每个分别归一化或者按每口井(well)归一化") 
    parser.add_argument('--filt_size',default=7,type=int,help='中值滤波的窗口大小，只可为奇数')
    parser.add_argument('--sam_fixed',default=False, 
                        help='是否固定sample,固定则样本对应井同起始点、终止点')
    parser.add_argument("--cross_corr_figpath", default='/home/cldai/sciresearch/logcomplete/figure/cross_corr/', help="cross_corr图存放地址")#

    #parser.add_argument('--savePath',default='/home/cldai/sciresearch/logcomplete/data/Teamdata/', help='matr的保存地址')  #cldai add
    #parser.add_argument("--figpath", default='/home/cldai/sciresearch/logcomplete/figure/matr/', help="数据matr图存放地址")#

    args = parser.parse_args()
    return args


class Cross_corelation(nn.Module):
    #print('############ MyGNNDataset_test')
    def __init__(self, data, sam_len,  fea_litho, norm_type, sam_fixed, filt_size):
        super(Cross_corelation,self).__init__()
        self.signal = data  
        self.sam_len = sam_len
        self.fea_litho = fea_litho
        self.norm_type = norm_type
        self.sam_fixed = sam_fixed
        self.filt_size = filt_size

        #self.sam_num = sam_num
        # self.mask_logs = mask_logs
        # self.null_sequence = null_sequence


    def random_single_well(self, df):
        '''取随机单口井，取需要的feature'''
        '''如果不想要随机井，可以把order替换乘固定值'''
        '''df 是文件内所有井数据'''
        
        all_well_name=df['WELL'].unique()  #df.WELL.unique()也是正确的  #wellname是numpy.ndarry
        
        order=np.random.randint(0,len(all_well_name))  #返回一个随机整数，范围[, )
        #print('all_well_name[order]',all_well_name[order]) #打印出是哪口井
        well_data = df[df['WELL'] == all_well_name[order]]
        ################ 固定order固定了使用某口井，取了'15/9-14'井 ################### 固定order固定了使用某口井 ###################
        #well_data = df[df['WELL'] == '15/9-14']  #经full_log检验'15/9-14'无null值   29/3-1
        print('$$$$$$$$$$$$$ 选取的井为：  ',well_data['WELL'].unique())
        #print('well_data',well_data)
        ################ 固定order固定了使用某口井 ################### 固定order固定了使用某口井 ################### 

        ''' #原来这种加上两个[]索引出来的还是dataframe,e而不是series,所以下面仍然可以直接pd.concat([md,well_data],axis=1)'''
        md = well_data[['DEPTH_MD']].copy()  
        #print('md',md)

        well_data=well_data[self.fea_litho].copy()
        #print('well_data.shape',well_data.shape)

        well_data = self.precond(well_data)

        #fea_null=self.nullNum_by_col(well_data)
        #print('fea_null',fea_null)

        #按照每口井进行归一化
        if self.norm_type=='well':
            print('按井归一化！')
            well_data=self.mean_var_norm(well_data)

        #在井操作最后drop null值
        well = pd.concat([md,well_data],axis=1)
        #print('well',well.columns,'len(well)',len(well))

        #将md中null值部分也删去了
        wellna=well.dropna(axis=0,how='any')  
        #print('len(wellna)',len(wellna))
        #print('wellna',wellna)


        return wellna


    def precond(self,well_data):
        '''去除一些异常数据'''
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        #well_data['SP'][well_data['SP']<-350] = np.nan  统计原始数据的相关度，这里先不换成np.nan了
        ##well_data['RMED'][well_data['RMED']<0] = 0  #岩性分类GIR队伍中是将此变为0
        ################ 去掉SP 去掉SP ################## 去掉SP 去掉SP ################## 去掉SP 去掉SP ##################
        return well_data

    def nullNum_by_col(self,df):
        '''用于检验是否有null值,同时也查看了columns是否按照fea_litho那样排列的（yes）'''
        nullNum=df.isnull().sum()
        return nullNum

    def stap_endp_dif(self,well_data):
        '''计算样本起始点和终止点的差，便于后续判断是否含有过多null值拼接而成'''
        if self.sam_fixed==True:
            stap=8000
            endp=stap+self.sam_len
            print('样本起始点固定为：{}'.format(stap))
        else:
            stap=np.random.randint(0,len(well_data)-self.sam_len)  #stap表示start_point
            endp=stap+self.sam_len  #endp表示end_point

        md_fea = well_data.iloc[stap:endp,:]
        md = md_fea['DEPTH_MD'].copy()
        dif = md.iloc[-1]-md.iloc[0]
        #print('dif',dif)

        return dif,stap,endp


    def label_for_intact(self, df):

        #寻找含有null值的行和列
        # for columname in well_data.columns:
        #     if well_data[columname].count() != len(well_data):
        #         loc = well_data[columname][well_data[columname].isnull().values==True].index.tolist()
        #         print('列名："{}", 第{}行位置有缺失值'.format(columname,loc))

        well_data = self.random_single_well(df)
        while(len(well_data)<self.sam_len):
            print('while while 111111')
            well_data = self.random_single_well(df)

        dif,stap,endp = self.stap_endp_dif(well_data)

        count = 0
        while((dif>self.sam_len*0.152+2) and (count<10)):   #放宽限度2.   count给到10应该能避免还是没有合适样本的情况了
            print('##################### while  dif')
            
            if count <4:   #给4次机会寻找没有drop null 而拼接的样本，否则直接换其他井再选样本
                dif,stap,endp = self.stap_endp_dif(well_data)
            else:
                well_data = self.random_single_well(df)
                while(len(well_data)<self.sam_len):
                    print('while 2222222')
                    well_data = self.random_single_well(df)

                dif,stap,endp = self.stap_endp_dif(well_data)
            
            count+=1


        well_data = well_data[self.fea_litho].copy()
        #print('well_data.columns',well_data.columns)

        #制作单个label, label*mask0=sample
        label_intact=well_data.iloc[stap:endp,:]

        #作中值滤波
        label_intact=self.medfilt(label_intact)

        #按样本将label归一化
        if self.norm_type == 'sample':
            # label_df=self.mean_var_norm(label_intact)
            label_intact=self.mean_var_norm(label_intact)
            print('按样本归一化')

        if self.norm_type != 'sample' and self.norm_type !='well':
            print('数据没有归一化!!!')
            
        #return label_df
        return label_intact

    def medfilt(self,label):
        b=label.shape[1]
        lab_cop=label.copy()
        for i in range(b):
            lab_cop.iloc[:,i]=signal.medfilt(lab_cop.iloc[:,i],self.filt_size)
        
        return lab_cop

    def mean_var_norm(self,label):
        #print('label.shape',label.shape)
        df1_mean=label.mean()
        #print('df1_mean',df1_mean.shape,df1_mean)
        df1_std=label.std(ddof=0)  #避免std使用index=0的影响
        #print('df1_std.shape',df1_std.shape)
        label_mean_var_norm=(label-df1_mean)/df1_std

        return label_mean_var_norm 

    def plot_mean_std_corr(self,corr_all,save_p):
        corr_mean = corr_all.mean(axis=0)
        corr_std = corr_all.std(axis=0)

        corr_mean = pd.DataFrame(corr_mean,columns=self.fea_litho,index=self.fea_litho)
        corr_std = pd.DataFrame(corr_std,columns=self.fea_litho,index=self.fea_litho)
        
        fig = plt.figure(figsize=(14,7))
        ax1 = fig.add_subplot(1,2,1)
        sn.heatmap(corr_mean, annot=True, cmap="BuPu")
        ax1.set_title('mean correlation')

        ax2 = fig.add_subplot(1,2,2)
        sn.heatmap(corr_std,annot=True,cmap='BuPu')
        ax2.set_title('std correlation')
        plt.savefig(args.cross_corr_figpath+'corr_mean_std.png',dpi=300,bbox_inches='tight')

        np.save(save_p+'corr_mean',corr_mean)
        np.save(save_p+'corr_std',corr_std)

    def plot_corr_fea(self,i,corr_matr,label_intact_df):
        fig = plt.figure(figsize = (14,7))
        spec = gridspec.GridSpec(ncols=8, nrows=1, width_ratios=[9,1,1,1,1,1,1,1],wspace=0.7)
        #ax = fig.add_subplot(1,8,1,spec[0])
        ax = fig.add_subplot(spec[0])
        sn.heatmap(corr_matr, annot=True, cmap="BuPu")
        ax.set_title('correlation confusion matrix',fontsize=15)

        y = y=np.linspace(1, label_intact_df.shape[0], label_intact_df.shape[0])
        for p in range(label_intact_df.shape[1]):  
            ax=fig.add_subplot(spec[p+1])  
            ax.plot(label_intact_df.iloc[:,p],y)
            plt.tick_params(labelsize=6)   #设置坐标轴刻度字体大小，没有set_tick_params,直接plt.tick_params
            ax.set_title(self.fea_litho[p])
            if p==0:
                ax.set_ylabel('sample_len',fontsize=10)

            ax.set_ylim(1,self.sam_len)

        if i%1000==0:
            plt.savefig(args.cross_corr_figpath+str(i)+'.png',dpi=300,bbox_inches='tight')

    def forward(self,num_sam,save_p):
        corr_all=np.zeros((num_sam,len(self.fea_litho),len(self.fea_litho)))
        
        for i in range(num_sam):
            ##wellna=self.random_single_well(self.signal)
            '''注意：随机采样的样本后它的index并不是重置为0-511(即label_intact_df的index不为0-511)，
            (pandas严格按照index和columns运算)
            所以若columns或index对应不上，后续四则运算等会为null值，'''

            label_intact_df = self.label_for_intact(self.signal)  
            '''corr()函数默认Pearson法，还有Kendall，spearman法。corr不能忽略nan影响，有nan/null则涉及到的均为nan/null.'''
            corr_matr = label_intact_df.corr()  
            np_cm = corr_matr.copy()
            corr_all[i,:,:] = np.array(np_cm)

            self.plot_corr_fea(i,corr_matr,label_intact_df)
            
        self.plot_mean_std_corr(corr_all,save_p)




def main(args):

    dfTrain=pd.read_csv(args.filePath+'train1.csv',engine='python')
    #dfTest=pd.read_csv(args.filePath+'test.csv',engine='python')
    # df_board_test=pd.read_csv(args.filePath+'board_test.csv',engine='python')

    '''#用cross_corr实例化Cross_corelation类，传入__init__中的参数。然后再通过给实例化的cross_corr传入其forward中的参数进行运算'''
    cross_corr = Cross_corelation(data=dfTrain, sam_len=args.sam_len, fea_litho=args.fea_litho, 
                                norm_type=args.norm_type, sam_fixed=args.sam_fixed, filt_size=args.filt_size)
    cross_corr(args.num_sam,args.filePath)


if __name__ == '__main__':
    args = read_args()
    main(args)








