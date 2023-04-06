'''该module为使用gardner's equation相互预测DTC与RHOB并绘图
    计算gardner结果与label的相关度，并绘制相关矩阵'''

import numpy as np  #看看像numpy，pandas这种官方的缩写都是歌两个字母来缩写，可借鉴其写法（  它是按发音缩写的）
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
import scipy.stats as stats




def gardner_plot(args,rhob,dtc,dp_rhob,rp_dtc):
    '''绘制Gardner预测效果与true的对比'''
    a = rhob.shape[0]  #rhob经过从label中抽取，其shape=(512,)，一维的
    #print('a',a)  #512
    fig = plt.figure(figsize=(4,8))
    y = np.linspace(1,a,a)

    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(rhob,y)
    ax1.set_title('RHOB (g/cm^3)')

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(dtc,y,label='label_dtc')
    ax2.plot(rp_dtc,y,'r',label='rp_dtc')
    ax2.set_title('DTC (us/ft)')
    plt.legend(loc='upper right')

    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(dtc,y)
    ax3.set_title('DTC (us/ft)')

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(rhob,y,label='label_rhob')
    ax4.plot(dp_rhob,y,'r',label='dp_rhob')
    ax4.set_title('RHOB (g/cm^3)')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(args.fig_path+'gardner/result/'+'gardner_result.png')

def plot_gardner_corr(args,corr_matr,figname):
    fig = plt.figure(figsize = (7,7))
    sn.heatmap(corr_matr, annot=True, cmap="BuPu",annot_kws={'fontsize':15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("coefficient of correlation(Gardner)",fontsize=20)
    plt.savefig(args.fig_path+'gardner/corrcoef/'+figname+'.png',dpi=300,bbox_inches='tight')

def gardner_corr(args,rhob,dtc,dp_rhob,rp_dtc):
    n_rb = np.vstack((rhob,dp_rhob))  #rhob,dp_rhob均是一维的，也可用np.vstack这样垂直堆叠
    n_rb = n_rb.T
    #print('n_rb.shape',n_rb.shape)  #n_rb.shape=(512,2)

    n_dtc = np.vstack((dtc,rp_dtc))
    n_dtc = n_dtc.T

    df_rb = pd.DataFrame(n_rb, columns=['label_rhob','dp_rhob'])
    df_dtc = pd.DataFrame(n_dtc, columns=['label_dtc','rp_dtc'])

    rb_corr = df_rb.corr()
    dtc_corr = df_dtc.corr()

    plot_gardner_corr(args,rb_corr,figname='RHOB')
    plot_gardner_corr(args,dtc_corr,figname='DTC')


def gardner_equ(args,label,mean,std):  
    '''函数旨在使用与我方法一样经过相同预处理的样本(label).以此来做RHOB和DTC间的预测
    RHOB预测DTC，DTC也预测RHOB'''
    '''调用计算相关度的函数，计算使用加德纳的预测结果与真实数值的相关度，
    而我方法的预测也与真实数值作相关度，二者进行比较'''

    #print('label.shape',label.shape)   #(4,512)
    #print('std.shape',std.shape)  #得是1行4列二维的数组才行，是(4,)这种一维的数据则错误  #std.shape=(1,4)

    #反归一化，将原始数据还原成未经过归一化的表示实际物理意义的数据
    label_phy = label*(std.T)+mean.T   #乘以转置是因为label是(4,512),mean是(1,4)顺序,然后再广播机制

    #从这label（单个）中取出RHOB和DTC序列
    rhob_ord = args.fea_litho.index('RHOB')   #rhob_ord is the order of RHOB
    dtc_ord = args.fea_litho.index('DTC')
    #print('dtc_ord',dtc_ord)  #3
    rhob = label_phy[rhob_ord,:]
    #print('rhob.shape',rhob.shape)  #shape=(512,)一维
    dtc = label_phy[dtc_ord,:]

    #基于Gardner's equation用dtc来推导rhob
    vp=1/(dtc*(1e-6))  #vp是纵波速度
    dp_rhob = 0.23*np.power(vp,0.25)   #dp_rhob is dtc_pred_rhob

    #基于Gardner's equation用rhob来推导dtc
    rp_vp = rhob/0.23  #rp_vp is rhob_pred_vp,利用rhob来推导vp
    rp_dtc = (1/np.power(rp_vp,4))*(1e6)

    gardner_plot(args,rhob,dtc,dp_rhob,rp_dtc)

    gardner_corr(args,rhob,dtc,dp_rhob,rp_dtc)
    





# ################ Gardner计算整口井的相关系数 ################ Gardner计算整口井的相关系数 ################ 
# def whole_gardner(label,fea_litho,md_stap,md_endp,figpath,well_name):  
#     '''函数旨在使用与我方法一样经过相同预处理的样本(label).以此来做RHOB和DTC间的预测
#     RHOB预测DTC，DTC也预测RHOB'''
#     '''调用计算相关度的函数，计算使用加德纳的预测结果与真实数值的相关度，
#     而我方法的预测也与真实数值作相关度，二者进行比较'''

#     #从这label（单个）中取出RHOB和DTC序列
#     rhob_ord = fea_litho.index('RHOB')   #rhob_ord is the order of RHOB
#     dtc_ord = fea_litho.index('DTC')
#     #print('dtc_ord',dtc_ord)  #3
#     rhob = label[rhob_ord,:]
#     #print('rhob.shape',rhob.shape)  #shape=(512,)一维
#     dtc = label[dtc_ord,:]

#     #基于Gardner's equation用dtc来推导rhob
#     vp=1/(dtc*(1e-6))  #vp是纵波速度
#     dp_rhob = 0.23*np.power(vp,0.25)   #dp_rhob is dtc_pred_rhob

#     #基于Gardner's equation用rhob来推导dtc
#     rp_vp = rhob/0.23  #rp_vp is rhob_pred_vp,利用rhob来推导vp
#     rp_dtc = (1/np.power(rp_vp,4))*(1e6)

#     #绘制Gardner方程的，rhob预测dtc，dtc预测rhob
#     plot_whole_gardner(md_stap,md_endp,figpath,well_name,input=rhob,true=dtc,pred=rp_dtc,predwho='DTC')
#     plot_whole_gardner(md_stap,md_endp,figpath,well_name,input=dtc,true=rhob,pred=dp_rhob,predwho='RHOB')

#     #计算相关系数
#     whole_gardner_corr(figpath,well_name,rhob,dtc,dp_rhob,rp_dtc)


# def plot_whole_gardner(md_stap,md_endp,figpath,well_name,input,true,pred,predwho):
#     '''绘制Gardner预测效果与true的对比'''
#     a = input.shape[0]  
#     print('input.shape',input.shape)
#     print('输入的长度',a)
#     y = np.linspace(a,1,a)
#     fig,axes = plt.subplots(1,2,figsize=(2*2.5,16),sharex = False,sharey = True,dpi=300) 
#     fig.suptitle(str(well_name),fontsize=15)
#     for i in range(2):
#         if i == 0:
#             axes[i].plot(input,y,label='Input')
#             if predwho == 'RHOB':
#                 axes[i].set_title('DTC',fontsize=15)
#                 axes[i].set_xlabel('g/cm\u00B3',fontsize=15)
#             else:
#                 axes[i].set_title('RHOB',fontsize=15)
#                 axes[i].set_xlabel('us/ft',fontsize=15)
#             axes[i].legend(loc='upper right',fontsize=13)
#         else:
#             axes[i].plot(true,y,label='True')
#             axes[i].plot(pred,y,'r',label='Pred')     
#             if predwho == 'RHOB':
#                 axes[i].set_title('RHOB',fontsize=15)
#                 axes[i].set_xlabel('us/ft',fontsize=15)
#             else:
#                 axes[i].set_title('DTC',fontsize=15)
#                 axes[i].set_xlabel('g/cm\u00B3',fontsize=15)
#             axes[i].legend(loc='upper right',fontsize=13)

#     axes[0].set_ylabel('Depth(m)',fontsize=15)
#     axes[0].set_ylim(1,a)
#     ticks = np.linspace(1,a,10)
#     labels = np.linspace(md_endp, md_stap, 10)
#     int_labels = list(map(int, labels[:]))
#     plt.yticks(ticks=ticks,labels=int_labels,fontsize=13)

#     plt.tight_layout()
#     plt.savefig(figpath+'gardner/whole_result/'+str(well_name)+'_'+predwho+'.png',dpi=300,bbox_inches='tight')


# def whole_gardner_corr(figpath,well_name,rhob,dtc,dp_rhob,rp_dtc):
#     '''计算Gardner方程预测效果的相关系数'''
#     n_rb = np.vstack((rhob,dp_rhob))  #rhob,dp_rhob均是一维的，也可用np.vstack这样垂直堆叠
#     n_rb = n_rb.T

#     n_dtc = np.vstack((dtc,rp_dtc))
#     n_dtc = n_dtc.T

#     df_rb = pd.DataFrame(n_rb, columns=['label(RHOB)','pred(RHOB)'])
#     df_dtc = pd.DataFrame(n_dtc, columns=['label(DTC)','pred(DTC)'])

#     #下面为去除nan值后计算相关系数
#     rb_dna = df_rb.dropna(axis=0,how='any')  #dropna后，df_rb是未变的
#     dtc_dna = df_dtc.dropna(axis=0,how='any')

#     rb_corr = rb_dna.corr()
#     dtc_corr = dtc_dna.corr()

#     plot_whole_gardner_corr(figpath,well_name,rb_corr,figname='RHOB')
#     plot_whole_gardner_corr(figpath,well_name,dtc_corr,figname='DTC')

# def plot_whole_gardner_corr(figpath,well_name,corr_matr,figname):
#     fig = plt.figure(figsize = (7,7))
#     sn.heatmap(corr_matr, annot=True, cmap="BuPu",fmt='.3f',annot_kws={'fontsize':15})   #fmt为数据格式，fmt='.3f'保留3位小数
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=15)
#     plt.title("coefficient of correlation(Gardner)",fontsize=20)
#     plt.savefig(figpath+'gardner/whole_corrcoef/'+str(well_name)+'_'+figname+'.png',dpi=300,bbox_inches='tight')
# ################ Gardner计算整口井的相关系数 ################ Gardner计算整口井的相关系数 ################ 


################ Gardner计算整口井的相关系数,不画输入 ################ Gardner计算整口井的相关系数，不画输入 ################ 
def whole_gardner(label,fea_litho,md_stap,md_endp,figpath,well_name):  
    '''函数旨在使用与我方法一样经过相同预处理的样本(label).以此来做RHOB和DTC间的预测
    RHOB预测DTC，DTC也预测RHOB'''
    '''调用计算相关度的函数，计算使用加德纳的预测结果与真实数值的相关度，
    而我方法的预测也与真实数值作相关度，二者进行比较'''

    #从这label（单个）中取出RHOB和DTC序列
    rhob_ord = fea_litho.index('RHOB')   #rhob_ord is the order of RHOB
    dtc_ord = fea_litho.index('DTC')
    #print('dtc_ord',dtc_ord)  #3
    rhob = label[rhob_ord,:]
    #print('rhob.shape',rhob.shape)  #shape=(512,)一维
    dtc = label[dtc_ord,:]

    #基于Gardner's equation用dtc来推导rhob
    vp=1/(dtc*(1e-6))  #vp是纵波速度
    dp_rhob = 0.23*np.power(vp,0.25)   #dp_rhob is dtc_pred_rhob

    #基于Gardner's equation用rhob来推导dtc
    rp_vp = rhob/0.23  #rp_vp is rhob_pred_vp,利用rhob来推导vp
    rp_dtc = (1/np.power(rp_vp,4))*(1e6)

    #绘制Gardner方程的，rhob预测dtc，dtc预测rhob
    plot_whole_gardner(md_stap,md_endp,figpath,well_name,input=rhob,true=dtc,pred=rp_dtc,predwho='DTC')
    plot_whole_gardner(md_stap,md_endp,figpath,well_name,input=dtc,true=rhob,pred=dp_rhob,predwho='RHOB')

    #计算相关系数
    whole_gardner_corr(figpath,well_name,rhob,dtc,dp_rhob,rp_dtc)



def plot_whole_gardner(md_stap,md_endp,figpath,well_name,input,true,pred,predwho):
    '''绘制Gardner预测效果与true的对比'''
    a = input.shape[0]  
    print('input.shape',input.shape)
    print('输入的长度',a)
    y = np.linspace(a,1,a)
    fig = plt.figure(figsize=(3.5,16))
    #fig,axes = plt.subplots(1,2,figsize=(2*2.5,16),sharex = False,sharey = True,dpi=300) 
    #fig.suptitle(str(well_name),fontsize=15)
    
    plt.plot(true,y,color='k',linestyle='dashed')
    plt.plot(pred,y,'r')     
    if predwho == 'RHOB':
        plt.title('RHOB',fontsize=15)
        plt.xlabel('g/cm\u00B3',fontsize=15)
    else:
        plt.title('DTC',fontsize=15)
        plt.xlabel('us/ft',fontsize=15)
            #axes[i].legend(loc='upper right',fontsize=13)

    plt.ylabel('Depth(m)',fontsize=15)
    plt.ylim(1,a)
    ticks = np.linspace(1,a,10)
    labels = np.linspace(md_endp, md_stap, 10)
    int_labels = list(map(int, labels[:]))
    plt.yticks(ticks=ticks,labels=int_labels,fontsize=13)

    plt.tight_layout()
    plt.savefig(figpath+'gardner/whole_result/'+str(well_name)+'_'+predwho+'.png',dpi=300,bbox_inches='tight')


def whole_gardner_corr(figpath,well_name,rhob,dtc,dp_rhob,rp_dtc):
    '''计算Gardner方程预测效果的相关系数'''
    n_rb = np.vstack((rhob,dp_rhob))  #rhob,dp_rhob均是一维的，也可用np.vstack这样垂直堆叠
    n_rb = n_rb.T

    n_dtc = np.vstack((dtc,rp_dtc))
    n_dtc = n_dtc.T

    df_rb = pd.DataFrame(n_rb, columns=['label(RHOB)','pred(RHOB)'])
    df_dtc = pd.DataFrame(n_dtc, columns=['label(DTC)','pred(DTC)'])

    #下面为去除nan值后计算相关系数
    rb_dna = df_rb.dropna(axis=0,how='any')  #dropna后，df_rb是未变的
    dtc_dna = df_dtc.dropna(axis=0,how='any')

    ##绘制Gardner预测的交汇图
    plot_gardner_whole_scatter(rb_dna,figpath,well_name,figname='RHOB')
    plot_gardner_whole_scatter(dtc_dna,figpath,well_name,figname='DTC')

    ##绘制相关系数的热力图
    rb_corr = rb_dna.corr()
    dtc_corr = dtc_dna.corr()
    plot_whole_gardner_corr(figpath,well_name,rb_corr,figname='RHOB')
    plot_whole_gardner_corr(figpath,well_name,dtc_corr,figname='DTC')

def plot_gardner_whole_scatter(data,figpath,well_name,figname): #figpath2
    fig = plt.figure(figsize=(8.8,7))  #(width,height)
    df = data.copy()
    df.columns = ["label", "pred"]

    _ = df['label'].copy()
    ax = plt.scatter(x=df['label'], y=df['pred'], c=_, cmap='jet')  #colormap=viridis
    xylim_min, xylim_max = gardner_whole_scatter_xylim(x=df['label'], y=df['pred'])
    plt.xlim(xylim_min, xylim_max)
    plt.ylim(xylim_min, xylim_max)
    #plt.title("{} crossplot".format(figname),fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    cbar = plt.colorbar(ax)  #plt.colorbar(ax, label=u'g/cm\u00B3')正确  #plt.colorbar()没有labelsize=15参数
    cbar.ax.tick_params(labelsize=15)  #设置色标刻度字体大小。

    if figname == 'RHOB':
        plt.ylabel('Predicted {} ({})'.format(figname,u'g/cm\u00B3'),fontsize=15)  #u'g/cm\00B3'为打印角标  注意其他方法
        plt.xlabel('True {} ({})'.format(figname,u'g/cm\u00B3'),fontsize=15)
        cbar.set_label(label=u'g/cm\u00B3',fontsize=15)   #cbar.set_label必须要有label
    elif figname == 'NPHI':
        plt.ylabel('Predicted {} (Pu)'.format(figname),fontsize=15)  #u'g/cm\00B3'为打印角标  注意其他方法
        plt.xlabel('True {} (Pu)'.format(figname),fontsize=15)
        cbar.set_label(label='Pu',fontsize=15)
    else:
        plt.ylabel('Predicted {} (us/ft)'.format(figname),fontsize=15)  
        plt.xlabel('True {} (us/ft)'.format(figname),fontsize=15)
        cbar.set_label(label='us/ft',fontsize=15)
    plt.tight_layout()
    plt.savefig(figpath+'gardner/cross_plot/'+ well_name+'_'+figname+'_cp.png',dpi=300,bbox_inches='tight')  #cp is cross_plot

def gardner_whole_scatter_xylim(x,y):
    x_min = pd.DataFrame(x).min()
    x_max = pd.DataFrame(x).max()
    y_min = pd.DataFrame(y).min()
    y_max = pd.DataFrame(y).max()

    xylim_min = pd.DataFrame(np.concatenate((x_min.values, y_min.values),axis=0))
    xylim_max = pd.DataFrame(np.concatenate((x_max.values, y_max.values),axis=0))
    return xylim_min.min().values*0.9, xylim_max.min().values*1.05  #乘以一个数是为了放大坐标让看起来好看点



def plot_whole_gardner_corr(figpath,well_name,corr_matr,figname):
    fig = plt.figure(figsize = (7,7))
    sn.heatmap(corr_matr, annot=True, cmap="BuPu",fmt='.3f',annot_kws={'fontsize':15})   #fmt为数据格式，fmt='.3f'保留3位小数
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("coefficient of correlation(Gardner)",fontsize=20)
    plt.savefig(figpath+'gardner/whole_corrcoef/'+str(well_name)+'_'+figname+'.png',dpi=300,bbox_inches='tight')
################ Gardner计算整口井的相关系数,不画输入 ################ Gardner计算整口井的相关系数，不画输入 ################ 
