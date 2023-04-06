'''该module计算我的方法预测结果与完整标签的相关度，以此作为对比评价标准，并绘制相关矩阵'''

import numpy as np  #看看像numpy，pandas这种官方的缩写都是歌两个字母来缩写，可借鉴其写法（  它是按发音缩写的）
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
import scipy.stats as stats

def plot_corr(args,corr_matr,figname):
    fig = plt.figure(figsize = (7,7))
    sn.heatmap(corr_matr, annot=True, cmap="BuPu",fmt='.3f', annot_kws={"fontsize":15})
    plt.title("coefficient of correlation(my method)",fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(args.fig_path+'my_qt_eva/'+figname+'.png',dpi=300,bbox_inches='tight')

def my_qt_eva(args,pred,label_itc,mean,std):  #label_itc=label_intact

    #反归一化，将原始数据还原成未经过归一化的表示实际物理意义的数据
    label_itc_phy = label_itc*(std.T)+mean.T   #乘以转置是因为label是(4,512),mean是(1,4)顺序,然后再广播机制
    pred_phy = pred*(std.T)+mean.T

    rhob_od = args.fea_litho.index('RHOB')   #rhob_od is the order of RHOB
    nphi_od = args.fea_litho.index('NPHI')
    dtc_od = args.fea_litho.index('DTC')
    #print('dtc_ord',dtc_od)  #3

    rhob_l = label_itc_phy[rhob_od,:]  #rhob_l is rhob_label
    #print('rhob_l.shape',rhob_l.shape)  #shape=(512,)一维
    nphi_l = label_itc_phy[nphi_od,:]
    dtc_l = label_itc_phy[dtc_od,:]

    rhob_p = pred_phy[rhob_od,:]  #rhob_p is rhob_pred
    #print('rhob_p.shape',rhob_p.shape)  #shape=(512,)一维
    nphi_p = pred_phy[nphi_od,:]
    dtc_p = pred_phy[dtc_od,:]


    n_rb = np.vstack((rhob_l,rhob_p))  #rhob_l,rhob_p均是一维的，也可用np.vstack这样垂直堆叠
    n_rb = n_rb.T
    #print('n_rb.shape',n_rb.shape)  #n_rb.shape=(512,2)

    n_nphi = np.vstack((nphi_l,nphi_p))
    n_nphi = n_nphi.T

    n_dtc = np.vstack((dtc_l,dtc_p))
    n_dtc = n_dtc.T

    df_rb = pd.DataFrame(n_rb, columns=['label_RHOB','pred_RHOB'])
    df_nphi = pd.DataFrame(n_nphi, columns=['label_NPHI','pred_NPHI'])
    df_dtc = pd.DataFrame(n_dtc, columns=['label_DTC','pred_DTC'])

    rb_corr = df_rb.corr()
    nphi_corr = df_nphi.corr()
    dtc_corr = df_dtc.corr()

    plot_corr(args,rb_corr,figname='RHOB')
    plot_corr(args,nphi_corr,figname='NPHI')
    plot_corr(args,dtc_corr,figname='DTC')


################ 计算绘制整口井的相关系数,绘制散点图 ################ 计算绘制整口井的相关系数,绘制散点图 ################ 
def whole_well_corr(pred,label,fea_litho,figpath,well_name):
    
    #print('RHOB的order', fea_litho.index('RHOB'))
    rhob_od = fea_litho.index('RHOB')   #rhob_od is the order of RHOB
    nphi_od = fea_litho.index('NPHI')
    dtc_od = fea_litho.index('DTC')
    #print('dtc_ord',dtc_od)  #3

    rhob_l = label[rhob_od,:]  #rhob_l is rhob_label
    #print('rhob_l.shape',rhob_l.shape)  #shape=(512,)一维
    nphi_l = label[nphi_od,:]
    dtc_l = label[dtc_od,:]

    rhob_p = pred[rhob_od,:]  #rhob_p is rhob_pred
    #print('rhob_p.shape',rhob_p.shape)  #shape=(512,)一维
    nphi_p = pred[nphi_od,:]
    dtc_p = pred[dtc_od,:]

    n_rb = np.vstack((rhob_l,rhob_p))  #rhob_l,rhob_p均是一维的，也可用np.vstack这样垂直堆叠
    n_rb = n_rb.T
    #print('n_rb.shape',n_rb.shape)  #n_rb.shape=(512,2)

    n_nphi = np.vstack((nphi_l,nphi_p))
    n_nphi = n_nphi.T

    n_dtc = np.vstack((dtc_l,dtc_p))
    n_dtc = n_dtc.T

    df_rb = pd.DataFrame(n_rb, columns=['label','pred'])
    df_nphi = pd.DataFrame(n_nphi, columns=['label','pred'])
    df_dtc = pd.DataFrame(n_dtc, columns=['label','pred'])
    print('df_dtc.shape', df_dtc.shape)

    #绘制交汇图
    plot_whole_scatter(df_rb,figpath,'withna/',well_name,figname='RHOB')
    plot_whole_scatter(df_nphi,figpath,'withna/',well_name,figname='NPHI')
    plot_whole_scatter(df_dtc,figpath,'withna/',well_name,figname='DTC')

    rb_corr = df_rb.corr()
    nphi_corr = df_nphi.corr()
    dtc_corr = df_dtc.corr()
    #print('dtc_corr',dtc_corr)

    plot_whole_corr(figpath,'cat_sam(withna)/',well_name,rb_corr,figname='RHOB')
    plot_whole_corr(figpath,'cat_sam(withna)/',well_name,nphi_corr,figname='NPHI')
    plot_whole_corr(figpath,'cat_sam(withna)/',well_name,dtc_corr,figname='DTC')

    print('下面看去除nan值后的相关系数和交汇图')
    #下面为去除nan值后计算相关系数
    rb_dna = df_rb.dropna(axis=0,how='any')  #dropna后，df_rb是未变的
    nphi_dna = df_nphi.dropna(axis=0,how='any')
    dtc_dna = df_dtc.dropna(axis=0,how='any')
    print("dtc_dna.shape", dtc_dna.shape)
    #绘制交汇图
    plot_whole_scatter(rb_dna,figpath,'dropna/',well_name,figname='RHOB')
    plot_whole_scatter(nphi_dna,figpath,'dropna/',well_name,figname='NPHI')
    plot_whole_scatter(dtc_dna,figpath,'dropna/',well_name,figname='DTC')
    #计算相关系数
    rb_dna_c = rb_dna.corr()
    nphi_dna_c = nphi_dna.corr()
    dtc_dna_c = dtc_dna.corr()
    #print('dtc_dna_c',dtc_dna_c)
    plot_whole_corr(figpath,'cat_sam(dropna)/',well_name,rb_dna_c,figname='RHOB')
    plot_whole_corr(figpath,'cat_sam(dropna)/',well_name,nphi_dna_c,figname='NPHI')
    plot_whole_corr(figpath,'cat_sam(dropna)/',well_name,dtc_dna_c,figname='DTC')


def plot_whole_corr(figpath,figpath2,well_name,corr_matr,figname):
    fig = plt.figure(figsize = (7,7))
    sn.heatmap(corr_matr, annot=True, cmap="BuPu",fmt='.3f',annot_kws={"fontsize":15})
    plt.title("{} correlation coefficient".format(figname),fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(figpath+'my_qt_eva/'+figpath2+ well_name+'_'+figname+'.png',dpi=300,bbox_inches='tight')


def plot_whole_scatter(df,figpath,figpath2,well_name,figname):
    fig = plt.figure(figsize=(8.8,7))  #(width,height)
    _ = df['label'].copy()
    ax = plt.scatter(x=df['label'], y=df['pred'], c=_, cmap='jet')  #colormap=viridis
    xylim_min, xylim_max = whole_scatter_xylim(x=df['label'], y=df['pred'])
    plt.xlim(xylim_min, xylim_max)
    plt.ylim(xylim_min, xylim_max)
    #plt.title("{} crossplot".format(figname),fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar = plt.colorbar(ax)  #plt.colorbar(ax, label=u'g/cm\u00B3')正确  #plt.colorbar()没有labelsize=15参数
    cbar.ax.tick_params(labelsize=20)  #设置色标刻度字体大小。

    if figname == 'RHOB':
        plt.ylabel('Predicted {} ({})'.format(figname,u'g/cm\u00B3'),fontsize=25)  #u'g/cm\00B3'为打印角标  注意其他方法
        plt.xlabel('True {} ({})'.format(figname,u'g/cm\u00B3'),fontsize=25)
        cbar.set_label(label=u'g/cm\u00B3',fontsize=25)   #cbar.set_label必须要有label
    elif figname == 'NPHI':
        plt.ylabel('Predicted {} (Pu)'.format(figname),fontsize=25)  #u'g/cm\00B3'为打印角标  注意其他方法
        plt.xlabel('True {} (Pu)'.format(figname),fontsize=25)
        cbar.set_label(label='Pu',fontsize=25)
    else:
        plt.ylabel('Predicted {} (us/ft)'.format(figname),fontsize=25)  
        plt.xlabel('True {} (us/ft)'.format(figname),fontsize=25)
        cbar.set_label(label='us/ft',fontsize=25)
    plt.tight_layout()
    plt.savefig(figpath+'my_qt_eva/cross_plot/'+figpath2+ well_name+'_'+figname+'_cp.png',dpi=300,bbox_inches='tight')  #cp is cross_plot

def whole_scatter_xylim(x,y):
    x_min = pd.DataFrame(x).min()
    x_max = pd.DataFrame(x).max()
    y_min = pd.DataFrame(y).min()
    y_max = pd.DataFrame(y).max()

    xylim_min = pd.DataFrame(np.concatenate((x_min.values, y_min.values),axis=0))
    xylim_max = pd.DataFrame(np.concatenate((x_max.values, y_max.values),axis=0))
    return xylim_min.min().values*0.9, xylim_max.min().values*1.05  #乘以一个数是为了放大坐标让看起来好看点


################ 计算绘制整口井的相关系数,绘制散点图 ################ 计算绘制整口井的相关系数,绘制散点图 ################ 




