import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.my_corr_eva import *

################# 用于陈述工作内容 ################## 用于陈述工作内容 ################## 用于陈述工作内容 #####################
def plt_state_work(label,figpath):  

    figwidth = label.shape[0]
    y=np.linspace(label.shape[1], 1, label.shape[1])  #注意画图时坐标轴及曲线的正反
    
    rhob_stap = 1000
    rhob_endp = 3000
    nphi_stap = 6000
    nphi_endp = 8500
    dtc_stap = 3500
    dtc_endp = 7000

    rhob_ip = label[1].copy()
    rhob_ip[rhob_stap:rhob_endp] = np.nan

    nphi_ip = label[2].copy()
    nphi_ip[nphi_stap:nphi_endp] = np.nan

    dtc_ip = label[3].copy()
    dtc_ip[dtc_stap:dtc_endp] = np.nan

    plot_input(figwidth,y,label,rhob_ip,nphi_ip,dtc_ip,figpath)
    plot_output(figwidth,y,label,rhob_ip,nphi_ip,dtc_ip,figpath)


def plot_input(figwidth,y,label,rhob_ip,nphi_ip,dtc_ip,figpath):
    fig,axes = plt.subplots(1,label.shape[0], figsize=(3*figwidth,14), sharex=False,sharey=True,dpi=300)  #sharey=True,y轴刻度值共享  #2.5*figwidth,16

    print('label.shape',label.shape)
    axes[0].plot(label[0,:],y,color='k')
    #画掩盖后的log的输入曲线
    #RHOB
    axes[1].plot(rhob_ip,y,color='k')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
    #NPHI
    axes[2].plot(nphi_ip,y,color='k')
    #DTC
    axes[3].plot(dtc_ip,y,color='k')

    '''#axes.xaxis.set_visible(False)设置坐标轴标签，轴刻度，轴刻度值是否可见  
    # #x/yaxis.set_ticks([]) 设置坐标轴刻度及刻度值不可见，坐标轴标签不受影响
    #x/yaxis.set_ticklabels([])  设置坐标轴刻度值不可见，坐标轴刻度线及坐标轴标签不受影响'''
    axes[0].set_ylim(1,label.shape[1])
    axes[0].yaxis.set_visible(False)
    for i in range(label.shape[0]):
        axes[i].xaxis.set_visible(False)  #设置坐标轴标签，轴刻度，轴刻度值是否可见
    plt.tight_layout()     #在我看来plt.tight_layout()是对整张图进行包括主标题，而bbox_inches感觉更像是对子图之间进行，而对与主图就不作用，如仅靠bbox_inches时主标题与子图之间位置不紧凑
    plt.savefig(figpath+'paper_fig/state_work/'+'input.png',dpi=300,bbox_inches='tight')


def plot_output(figwidth,y,label,rhob_ip,nphi_ip,dtc_ip,figpath):
    fig,axes = plt.subplots(1,label.shape[0], figsize=(3*figwidth,14), sharex=False,sharey=True,dpi=300)  #sharey=True,y轴刻度值共享
    
    axes[0].plot(label[0,:],y,color='k')
    #先用红色画完整的label，再用蓝色线条画输入，覆盖掉很多红色，只有输入中缺失部分的红色不覆盖，反向意味着这没覆盖的是预测的结果
    #RHOB
    axes[1].plot(label[1,:],y,color='r')  #
    axes[1].plot(rhob_ip,y,color='k')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
    #NPHI
    axes[2].plot(label[2,:],y,color='r')
    axes[2].plot(nphi_ip,y,color='k')
    #DTC
    axes[3].plot(label[3,:],y,color='r')
    axes[3].plot(dtc_ip,y,color='k')

    axes[0].set_ylim(1,label.shape[1])
    for i in range(label.shape[0]):
        axes[i].xaxis.set_visible(False)  #设置坐标轴标签，轴刻度，轴刻度值是否可见
    axes[0].yaxis.set_visible(False)
    plt.tight_layout() 
    plt.savefig(figpath+'paper_fig/state_work/'+'output.png',dpi=300,bbox_inches='tight')
################# 用于陈述工作内容 ################## 用于陈述工作内容 ################## 用于陈述工作内容 #####################





######@@@@@@@@@@@@ 用于绘制样本经历关键流程图 @@@@@@@@@@@@@@ 用于绘制样本经历关键流程图 @@@@@@@@@@@@@ 用于绘制样本经历关键流程图 @@@@@@@@@@@@######
def plot_sample_flow(ip_muti_ch,pred,label,mask0,maskd,fea_litho,figpath):
    sample_ori = label.copy()
    input = ip_muti_ch[:,0,:].copy()   #取出没有降采样插值即近原始的0通道输入
    subfig_num = len(fea_litho)
    #fig=plt.figure(figsize=(2*figwidth,4),dpi=300)  #三行七列时（12，12）  #figsize=(width,height)
    y=np.linspace(input.shape[1], 1, input.shape[1])  #注意画图时坐标轴及曲线的正反
    
    input_afm0=np.where(mask0==0, np.nan, input)  #input_afm0=input_after_mask0_for_fig
    input_afmd=np.where(maskd==0, np.nan, input_afm0)  #input_afmd = input_after_maskd_for_fig
    #label只涉及到null值，即考虑不画null值时，用mask0即可，不涉及到maskd
    #print('input_afmd',input_afmd)

    #将RHOB列设置为代替mask0
    sample_ori[1,:] = np.nan
    label[1,:] = np.nan
    input_afmd[1,:] = np.nan
    ip_muti_ch[1,:,:] = np.nan

    #设置多通道输入的DTC列为nan
    ip_muti_ch[3,:,:] = np.nan

    lis_xlim=[input_afmd, pred, label]  

    
    #plot_sam_flow_sample(sample_ori,y,lis_xlim,subfig_num,fea_litho,figpath,name='1_sample_ori',muti_ch=False)
    plot_sam_flow_sample(label,y,lis_xlim,subfig_num,fea_litho,figpath,name='2_label',muti_ch=False)
    plot_sam_flow_sample(input_afmd,y,lis_xlim,subfig_num,fea_litho,figpath,name='3_input_afmd',muti_ch=False)
    plot_sam_flow_sample(ip_muti_ch,y,lis_xlim,subfig_num,fea_litho,figpath,name='4_input_muti_ch',muti_ch=True)
    plot_sam_flow_sample(pred,y,lis_xlim,subfig_num,fea_litho,figpath,name='5_pred',muti_ch=False)
    
    plot_sam_flow_pred_and_label(pred,label,y,lis_xlim,subfig_num,fea_litho,figpath,name='6_pred_and_label')

    #以原始缺失RHOB列为例画mask0
    plot_mask(y,subfig_num,fea_litho,figpath,mask_order=1)
    #以缺失DTC为例画maskd
    plot_mask(y,subfig_num,fea_litho,figpath,mask_order=3)


def plot_sam_flow_sample(sample,y,lis_xlim,subfig_num,fea_litho,figpath,name,muti_ch=False):
    fig,axes = plt.subplots(1,subfig_num,figsize=(2*subfig_num,5),sharex=False,sharey=True,dpi=300)
    for i in range(subfig_num):
        if muti_ch == True:
            '''多通道样本时，分别画每个通道的数据'''
            "plot中默认的十种颜色default: cycler('color', ['#1f77b4'蓝, '#ff7f0e'黄, '#2ca02c'绿, '#d62728'红"
            "'#9467bd'紫, '#8c564b'棕, '#e377c2'深粉, '#7f7f7f'深灰, '#bcbd22'黄绿, '#17becf'蓝绿]))"
            axes[i].plot(sample[i,0,:],y,color='k')  
            axes[i].plot(sample[i,1,:],y,color='y')  #green 
            axes[i].plot(sample[i,2,:],y,color='g')  #c:cyan  #r:red  #m:magenta洋红  #brown棕色  
            axes[i].plot(sample[i,3,:],y,color='brown')  #y:yellow  b:blue
            #plt.legend(loc='lower right',fontsize=15,shadow=False)
            
        else: 
            if name == '5_pred':
                axes[i].plot(sample[i],y,color='r')  
            else:
                axes[i].plot(sample[i],y,color='k')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用

        #axes[i].set_title(fea_litho[i],fontsize=20)

        #按需求决定要不要使lim保持一致
        # xlim_min, xlim_max = sam_flow_xlim(lis_xlim,i)
        # axes[i].set_xlim(xlim_min, xlim_max)
    
    axes[0].set_ylim(1,len(y))
    axes[0].yaxis.set_visible(False)
    for i in range(subfig_num):
        axes[i].xaxis.set_visible(False)  #设置坐标轴标签，轴刻度，轴刻度值是否可见
 
    plt.tight_layout()     #在我看来plt.tight_layout()是对整张图进行包括主标题，而bbox_inches感觉更像是对子图之间进行，而对与主图就不作用，如仅靠bbox_inches时主标题与子图之间位置不紧凑
    plt.savefig(figpath+'paper_fig/sample_flow/'+str(name)+'.png',dpi=300,bbox_inches='tight')


def plot_sam_flow_pred_and_label(pred,label,y,lis_xlim,subfig_num,fea_litho,figpath,name):
    fig,axes = plt.subplots(1,subfig_num,figsize=(2*subfig_num,5),sharex=False,sharey=True,dpi=300)
    for i in range(subfig_num):
        axes[i].plot(label[i],y,color='k')
        axes[i].plot(pred[i],y,color='r')  
        #axes[i].set_title('y$_{}$ & y$_{}$'.format(i,i),fontsize=15)
        xlim_min, xlim_max = sam_flow_xlim(lis_xlim,i)
        axes[i].set_xlim(xlim_min, xlim_max)
        #plt.legend(loc='upper right',shadow=False)  #frameon表示图例边框  #shadow控制图例背后有无阴影
    
    axes[0].set_ylim(1,len(y))
    axes[0].yaxis.set_visible(False)
    for i in range(subfig_num):
        axes[i].xaxis.set_visible(False)  #设置坐标轴标签，轴刻度，轴刻度值是否可见
    
    plt.tight_layout()     #在我看来plt.tight_layout()是对整张图进行包括主标题，而bbox_inches感觉更像是对子图之间进行，而对与主图就不作用，如仅靠bbox_inches时主标题与子图之间位置不紧凑
    plt.savefig(figpath+'paper_fig/sample_flow/'+str(name)+'.png',dpi=300,bbox_inches='tight')

def sam_flow_xlim(lis_xlim,i):  #lis[0],[1],[2]的shape：7*batch_size, 64
    '''选择合适的横坐标范围'''
    input_min=pd.DataFrame(lis_xlim[0][i,:]).min()
    input_max=pd.DataFrame(lis_xlim[0][i,:]).max()
    pred_min=pd.DataFrame(lis_xlim[1][i,:]).min()
    pred_max=pd.DataFrame(lis_xlim[1][i,:]).max()
    label_min=pd.DataFrame(lis_xlim[2][i,:]).min()
    label_max=pd.DataFrame(lis_xlim[2][i,:]).max()
    
    xlim_left=pd.DataFrame(np.concatenate((input_min.values, pred_min.values, label_min.values),axis=0))
    xlim_right=pd.DataFrame(np.concatenate((input_max.values, pred_max.values, label_max.values),axis=0))
    #print(xlim_left,xlim_right)
    #print("xlim_left.min(), xlim_right.max()",xlim_left.min(), xlim_right.max())

    return xlim_left.min().values, xlim_right.max().values


def plot_mask(y,subfig_num,fea_litho,figpath,mask_order):
    '''以原始缺失RHOB列为例画mask0,以DTC缺失为例画maskd'''
    fig,axes = plt.subplots(1,subfig_num,figsize=(2*subfig_num,5),sharex=False,sharey=True)
    null = np.full((y.shape),np.nan)
    a = np.ones((y.shape[0],1))
    for i in range(subfig_num):
        if i == mask_order:
            axes[i].imshow(a,interpolation='nearest',aspect='auto',cmap=plt.cm.gray) 
        else:
            axes[i].plot(null,y)
        axes[i].xaxis.set_visible(False)
    axes[0].set_ylim(1,len(y))
    axes[0].yaxis.set_visible(False)

    plt.tight_layout()
    plt.savefig(figpath+'paper_fig/sample_flow/'+'mask_'+str(fea_litho[mask_order])+'.png',
                dpi=300,bbox_inches='tight')
        


######@@@@@@@@@@@@ 用于绘制样本经历关键流程图 @@@@@@@@@@@@@@ 用于绘制样本经历关键流程图 @@@@@@@@@@@@@ 用于绘制样本经历关键流程图 @@@@@@@@@@@@######




################ 绘制全井效果 ###################### 绘制全井效果 ################
def paper_plt_cat_sam(input,pred,label,mask0,maskd,fea_litho,
                md_stap,md_endp,figpath,well_name,label_intact=False,pnl=False,
                mask_logs='DTC',plot_Gardner=True):  

    figwidth = input.shape[0]
    fig=plt.figure(figsize=(2*figwidth,16),dpi=300)  #三行七列时（12，12）  #figsize=(width,height)
    y=np.linspace(input.shape[1], 1, input.shape[1])  #注意画图时坐标轴及曲线的正反
    
    input_afm0=np.where(mask0==0, np.nan, input)  #input_afm0=input_after_mask0_for_fig
    input_afmd=np.where(maskd==0, np.nan, input_afm0)  #input_afmd = input_after_maskd_for_fig
    #label只涉及到null值，即考虑不画null值时，用mask0即可，不涉及到maskd
    #print('input_afmd',input_afmd)

    if label_intact==True:
        label_1=label
    else:
        label_1=np.where(mask0==0, np.nan, label)  

    pred1=np.where(mask0==0, pred, label)
    pred_need_label=np.where(maskd==0, pred, pred1)

    if pnl==True:
        lis=[input_afmd, pred, pred_need_label, label_1]
    else:
        lis=[input_afmd, pred, pred, label_1]
    print('len(lis)',len(lis))  #len=4,列表的长度是按照最外围计算的
    
    #print('label',label)
    #print('pred',pred)

    lis_copy=['input_afmd', 'pred', 'pred_need_label', 'label_1']

    #lis_xlim不需要加pred_need_label，因为pred_need_label的值由input和pred得来，
    #其所以xlim值由input和pred即可得
    #lis_xlim=[input, pred, label]   #检查发现lis_xlim不应该直接用input,label作比较，应该用最原始的含null值的
    lis_xlim=[input_afmd, pred, label_1]  
    lis1=['input', 'pred', 'pred & label']

    #计算Gardner预测数据
    rp_dtc, dtc, dp_rhob, rhob = return_gardner_data(label,fea_litho)
    
    ##因为考虑到绘图预测的异常值过大会影响图的趋势显示，所以以真值的(50%~150%为界)
    rp_dtc, dp_rhob = gardner_pred_bound(rp_dtc, dtc, dp_rhob, rhob)

    #paper_plot_input(figwidth,y,lis,lis1,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name)
    paper_plot_input_pred_label(figwidth,y,lis,lis1,lis_copy,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name,mask_logs,
                                rp_dtc, dtc, dp_rhob, rhob, plot_Gardner)


def gardner_pred_bound(rp_dtc, dtc, dp_rhob, rhob):
    '''因为考虑到绘图预测的异常值过大会影响图的趋势显示，所以以真值的(50%~150%为界)'''
    lb = 0.6  #lb is lower bound
    up = 1.4  #up is upper bound
    dtc_lb = dtc*lb   ##dtc_lb = dtc_lower_bound
    dtc_up = dtc*up
    rhob_lb = rhob*lb
    rhob_up = rhob*up

    pred_dtc = np.where(rp_dtc<dtc_lb,  np.nan, rp_dtc)
    pred_dtc = np.where(pred_dtc>dtc_up, np.nan, pred_dtc)

    pred_rhob = np.where(dp_rhob<rhob_lb, np.nan, dp_rhob)
    pred_rhob = np.where(pred_rhob>rhob_up, np.nan, pred_rhob)

    return pred_dtc, pred_rhob




def paper_plot_input_pred_label(figwidth,y,lis,lis1,lis_copy,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name,mask_logs,
                                rp_dtc, dtc, dp_rhob, rhob,plot_Gardner):
    unit = ['API','g/cm\u00B3','100Pu','us/ft']
    if plot_Gardner == True and (mask_logs == 'DTC' or mask_logs == 'RHOB') :
        '''plot_Gardner==True且mask_logs==RHOB或DTC时Gardner才会被使用，被绘制'''
        #下面的+1即是为绘制Gardner的效果
        width = lis[0].shape[0]+1
        fig,axes = plt.subplots(1,width, figsize=(4.5*(figwidth+1),16), sharex=False,sharey=True,dpi=300)
        if mask_logs == 'DTC':
            axes[-1].plot(rp_dtc,y,color='r')
            axes[-1].plot(dtc,y,color='k',linestyle='dashed')
            axes[-1].set_title('DTC (Gardner)',fontsize=25)
            axes[-1].set_xlabel('us/ft',fontsize=20)
            axes[-1].tick_params(labelsize=15) #刻度字体大小20
        else:
            axes[-1].plot(dp_rhob,y,color='r')
            axes[-1].plot(rhob,y,color='k',linestyle='dashed')
            axes[-1].set_title('RHOB (Gardner)',fontsize=25)
            axes[-1].set_xlabel('g/cm$^3$',fontsize=20)
            axes[-1].tick_params(labelsize=15) #刻度字体大小20
    else:
        fig,axes = plt.subplots(1,lis[0].shape[0], figsize=(4.5*figwidth,16), sharex=False,sharey=True,dpi=300)  #sharey=True,y轴刻度值共享
    
    #fig.suptitle(str(well_name),fontsize=15)  #添加主标题
    
    for i in range(lis[0].shape[0]): 
        #print('fea_litho.index(mask_logs)', fea_litho.index(mask_logs))
        if i == fea_litho.index(mask_logs):
            axes[3].plot(lis[lis1.index('pred & label')][i,:],y,'r-')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
            axes[3].plot(lis[lis_copy.index('label_1')][i,:],y,color='k',linestyle='dashed')  #同linestyle='--'
            axes[3].set_title(fea_litho[i]+' (our method)',fontsize=25)
            axes[3].set_xlabel(unit[i],fontsize=20)
            axes[3].tick_params(labelsize=15) #刻度字体大小20
        else:
            if fea_litho.index(mask_logs) == 3:
                axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
                axes[i].set_title(fea_litho[i],fontsize=25)
                # xlim_min, xlim_max = xlim(lis_xlim,i)
                # axes[i].set_xlim(xlim_min, xlim_max)
                axes[i].set_xlabel(unit[i],fontsize=20)
                axes[i].tick_params(labelsize=15) #刻度字体大小20
            elif fea_litho.index(mask_logs) == 2:
                if i == 0 or i == 1:
                    axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
                    axes[i].set_title(fea_litho[i],fontsize=25)
                    # xlim_min, xlim_max = xlim(lis_xlim,i)
                    # axes[i].set_xlim(xlim_min, xlim_max)
                    axes[i].set_xlabel(unit[i],fontsize=20)
                    axes[i].tick_params(labelsize=15) #刻度字体大小20
                else:
                    axes[i-1].plot(lis[lis1.index('input')][i,:],y,color='k')
                    axes[i-1].set_title(fea_litho[i],fontsize=25)
                    # xlim_min, xlim_max = xlim(lis_xlim,i)
                    # axes[i-1].set_xlim(xlim_min, xlim_max)
                    axes[i-1].set_xlabel(unit[i],fontsize=20)
                    axes[i-1].tick_params(labelsize=15) #刻度字体大小20
            elif fea_litho.index(mask_logs) == 1:
                if i == 0 :
                    axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
                    axes[i].set_title(fea_litho[i],fontsize=25)
                    # xlim_min, xlim_max = xlim(lis_xlim,i)
                    # axes[i].set_xlim(xlim_min, xlim_max)
                    axes[i].set_xlabel(unit[i],fontsize=20)
                    axes[i].tick_params(labelsize=15) #刻度字体大小20
                else:
                    axes[i-1].plot(lis[lis1.index('input')][i,:],y,color='k')
                    axes[i-1].set_title(fea_litho[i],fontsize=25)
                    # xlim_min, xlim_max = xlim(lis_xlim,i)
                    # axes[i-1].set_xlim(xlim_min, xlim_max)
                    axes[i-1].set_xlabel(unit[i],fontsize=20)
                    axes[i-1].tick_params(labelsize=15) #刻度字体大小20
    

    axes[0].set_ylabel('Depth(m)',fontsize=20)
    axes[0].set_ylim(1,lis[0].shape[1])
    ticks = np.linspace(1,lis[0].shape[1],10)
    labels = np.linspace(md_endp, md_stap, 10)
    int_labels = list(map(int, labels[:]))
    plt.yticks(ticks=ticks,labels=int_labels)  #,fontsize=13  这里其实只是最后一个子图尺寸为13
    plt.tight_layout()
    plt.savefig(figpath+'paper_fig/cat_sam/'+str(well_name)+'input_pred_label.jpg',dpi=300,bbox_inches='tight')










def xlim(lis_xlim,i):  #lis[0],[1],[2]的shape：7*batch_size, 64
    '''选择合适的横坐标范围'''
    input_min=pd.DataFrame(lis_xlim[0][i,:]).min()
    input_max=pd.DataFrame(lis_xlim[0][i,:]).max()
    pred_min=pd.DataFrame(lis_xlim[1][i,:]).min()
    pred_max=pd.DataFrame(lis_xlim[1][i,:]).max()
    label_min=pd.DataFrame(lis_xlim[2][i,:]).min()
    label_max=pd.DataFrame(lis_xlim[2][i,:]).max()
    
    xlim_left=pd.DataFrame(np.concatenate((input_min.values, pred_min.values, label_min.values),axis=0))
    xlim_right=pd.DataFrame(np.concatenate((input_max.values, pred_max.values, label_max.values),axis=0))
    #print(xlim_left,xlim_right)
    #print("xlim_left.min(), xlim_right.max()",xlim_left.min(), xlim_right.max())

    return xlim_left.min().values, xlim_right.max().values

def return_gardner_data(label,fea_litho):  
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

    return rp_dtc, dtc, dp_rhob, rhob
################ 绘制全井效果 ###################### 绘制全井效果 ################




#########################  输入三个不完整的log的预测图  ######################
def paper_input_3_incom_log(input,pred,label,mask0,maskd,fea_litho,
                md_stap,md_endp,figpath,well_name,label_intact=False,
                pnl=False,
                mask_logs='DTC', incom_log=False):  

    figwidth = input.shape[0]
    fig=plt.figure(figsize=(2*figwidth,16),dpi=300)  #三行七列时（12，12）  #figsize=(width,height)
    y=np.linspace(input.shape[1], 1, input.shape[1])  #注意画图时坐标轴及曲线的正反
    
    input_afm0=np.where(mask0==0, np.nan, input)  #input_afm0=input_after_mask0_for_fig
    input_afmd=np.where(maskd==0, np.nan, input_afm0)  #input_afmd = input_after_maskd_for_fig
    #label只涉及到null值，即考虑不画null值时，用mask0即可，不涉及到maskd
    #print('input_afmd',input_afmd)

    if label_intact==True:
        label_1=label
    else:
        label_1=np.where(mask0==0, np.nan, label)  

    pred1=np.where(mask0==0, pred, label)
    pred_need_label=np.where(maskd==0, pred, pred1)

    if incom_log == True:
        ##下面只保留要预测的部分的label_intact、pred,其他的弄成nan，让其不显示
        label_1 = np.where(mask0==0, label_1, np.nan)
        pred = np.where(mask0==0, pred, np.nan)

    if pnl==True:
        lis=[input_afmd, pred, pred_need_label, label_1]
    else:
        lis=[input_afmd, pred, pred, label_1]
    print('len(lis)',len(lis))  #len=4,列表的长度是按照最外围计算的
    
    #print('label',label)
    #print('pred',pred)

    lis_copy=['input_afmd', 'pred', 'pred_need_label', 'label_1']

    #lis_xlim不需要加pred_need_label，因为pred_need_label的值由input和pred得来，
    #其所以xlim值由input和pred即可得
    #lis_xlim=[input, pred, label]   #检查发现lis_xlim不应该直接用input,label作比较，应该用最原始的含null值的
    lis_xlim=[input_afmd, pred, label_1]  
    lis1=['input', 'pred', 'pred & label']


    #paper_plot_input(figwidth,y,lis,lis1,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name)
    paper_3_incom_log_input_pred_label(figwidth,y,lis,lis1,lis_copy,lis_xlim,fea_litho,
                                md_stap,md_endp,figpath,well_name,mask_logs,
                                )
    #计算相关系数,并绘制散点图
    whole_well_corr(pred,label_1,fea_litho,figpath,well_name)
    print('pred.shape, label_1.shape', pred.shape, label_1.shape)

def paper_3_incom_log_input_pred_label(figwidth,y,lis,lis1,lis_copy,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name,mask_logs,
                                ):
    unit = ['API','g/cm\u00B3','100Pu','us/ft']
    fig,axes = plt.subplots(1,lis[0].shape[0], figsize=(4.5*figwidth,16), sharex=False,sharey=True,dpi=300)  #sharey=True,y轴刻度值共享
    
    for i in range(lis[0].shape[0]): 

        if i == 0:
            axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
            axes[i].set_title(fea_litho[i],fontsize=25)
            axes[i].set_xlabel(unit[i],fontsize=20)
            axes[i].tick_params(labelsize=15) #刻度字体大小20
        
        else:
            axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
            axes[i].plot(lis[lis1.index('pred & label')][i,:],y,'r-',linewidth=3)  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
            axes[i].plot(lis[lis_copy.index('label_1')][i,:],y,color='g',linestyle='dashed', linewidth=3)  #同linestyle='--'
            axes[i].set_title(fea_litho[i],fontsize=25)
            axes[i].set_xlabel(unit[i],fontsize=20)
            axes[i].tick_params(labelsize=15) #刻度字体大小20

    axes[0].set_ylabel('Depth(m)',fontsize=20)
    axes[0].set_ylim(1,lis[0].shape[1])
    ticks = np.linspace(1,lis[0].shape[1],10)
    labels = np.linspace(md_endp, md_stap, 10)
    int_labels = list(map(int, labels[:]))
    plt.yticks(ticks=ticks,labels=int_labels)  #,fontsize=13  这里其实只是最后一个子图尺寸为13
    plt.tight_layout()
    plt.savefig(figpath+'paper_fig/cat_sam/'+str(well_name)+'_3_incom_log.jpg',dpi=300,bbox_inches='tight')
#########################  输入三个不完整的log的预测图  ######################



def gardner_pred_bound(rp_dtc, dtc, dp_rhob, rhob):
    '''因为考虑到绘图预测的异常值过大会影响图的趋势显示，所以以真值的(50%~150%为界)'''
    lb = 0.6  #lb is lower bound
    up = 1.4  #up is upper bound
    dtc_lb = dtc*lb   ##dtc_lb = dtc_lower_bound
    dtc_up = dtc*up
    rhob_lb = rhob*lb
    rhob_up = rhob*up

    pred_dtc = np.where(rp_dtc<dtc_lb,  np.nan, rp_dtc)
    pred_dtc = np.where(pred_dtc>dtc_up, np.nan, pred_dtc)

    pred_rhob = np.where(dp_rhob<rhob_lb, np.nan, dp_rhob)
    pred_rhob = np.where(pred_rhob>rhob_up, np.nan, pred_rhob)

    return pred_dtc, pred_rhob











# def paper_plot_input_pred_label(figwidth,y,lis,lis1,lis_copy,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name,mask_logs):
#     unit = ['API','g/cm\u00B3','100Pu','us/ft']
#     if plot_Gardner == True and (mask_logs == 'DTC' or mask_logs == 'RHOB'):
#         #下面的+1即是为绘制Gardner的效果
#         width = lis[0].shape[0]+1
#         fig,axes = plt.subplots(1,width, figsize=(4.5*(figwidth+1),16), sharex=False,sharey=True,dpi=300)
#     else:
#         fig,axes = plt.subplots(1,lis[0].shape[0], figsize=(4.5*figwidth,16), sharex=False,sharey=True,dpi=300)  #sharey=True,y轴刻度值共享
    
#     #fig.suptitle(str(well_name),fontsize=15)  #添加主标题
    
#     if fea_litho.index(mask_logs) == 3:
#         for i in range(lis[0].shape[0]): 
#             if i == fea_litho.index(mask_logs):
#                 axes[3].plot(lis[lis1.index('pred & label')][i,:],y,'r-')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
#                 axes[3].plot(lis[lis_copy.index('label_1')][i,:],y,color='k',linestyle='dashed')  #同linestyle='--'
#                 axes[3].set_title(fea_litho[i],fontsize=15)
#                 # xlim_min, xlim_max = xlim(lis_xlim,3)
#                 # axes[3].set_xlim(xlim_min, xlim_max)
#                 axes[3].set_xlabel(unit[i],fontsize=15)
#             else:
#                 axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
#                 axes[i].set_title(fea_litho[i],fontsize=15)
#                 # xlim_min, xlim_max = xlim(lis_xlim,i)
#                 # axes[i].set_xlim(xlim_min, xlim_max)
#                 axes[i].set_xlabel(unit[i],fontsize=15)

#     if fea_litho.index(mask_logs) == 2:
#         for i in range(lis[0].shape[0]): 
#             if i == fea_litho.index(mask_logs):
#                 axes[3].plot(lis[lis1.index('pred & label')][i,:],y,'r-')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
#                 axes[3].plot(lis[lis_copy.index('label_1')][i,:],y,color='k',linestyle='dashed')  #同linestyle='--'
#                 axes[3].set_title(fea_litho[i],fontsize=15)
#                 # xlim_min, xlim_max = xlim(lis_xlim,3)
#                 # axes[3].set_xlim(xlim_min, xlim_max)
#                 axes[3].set_xlabel(unit[i],fontsize=15)
#             elif i == 0 or i == 1:
#                 axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
#                 axes[i].set_title(fea_litho[i],fontsize=15)
#                 # xlim_min, xlim_max = xlim(lis_xlim,i)
#                 # axes[i].set_xlim(xlim_min, xlim_max)
#                 axes[i].set_xlabel(unit[i],fontsize=15)
#             else:
#                 axes[i-1].plot(lis[lis1.index('input')][i,:],y,color='k')
#                 axes[i-1].set_title(fea_litho[i],fontsize=15)
#                 # xlim_min, xlim_max = xlim(lis_xlim,i)
#                 # axes[i-1].set_xlim(xlim_min, xlim_max)
#                 axes[i-1].set_xlabel(unit[i],fontsize=15)
    
#     if fea_litho.index(mask_logs) == 1:
#         for i in range(lis[0].shape[0]): 
#             if i == fea_litho.index(mask_logs):
#                 axes[3].plot(lis[lis1.index('pred & label')][i,:],y,'r-')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
#                 axes[3].plot(lis[lis_copy.index('label_1')][i,:],y,color='k',linestyle='dashed')  #同linestyle='--'
#                 axes[3].set_title(fea_litho[i],fontsize=15)
#                 # xlim_min, xlim_max = xlim(lis_xlim,3)
#                 # axes[3].set_xlim(xlim_min, xlim_max)
#                 axes[3].set_xlabel(unit[i],fontsize=15)
#             elif i == 0 :
#                 axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
#                 axes[i].set_title(fea_litho[i],fontsize=15)
#                 # xlim_min, xlim_max = xlim(lis_xlim,i)
#                 # axes[i].set_xlim(xlim_min, xlim_max)
#                 axes[i].set_xlabel(unit[i],fontsize=15)
#             else:
#                 axes[i-1].plot(lis[lis1.index('input')][i,:],y,color='k')
#                 axes[i-1].set_title(fea_litho[i],fontsize=15)
#                 # xlim_min, xlim_max = xlim(lis_xlim,i)
#                 # axes[i-1].set_xlim(xlim_min, xlim_max)
#                 axes[i-1].set_xlabel(unit[i],fontsize=15)
        
#     axes[0].set_ylabel('Depth(m) [Pred(red)&Label(blue)]',fontsize=15)
#     axes[0].set_ylim(1,lis[0].shape[1])
#     ticks = np.linspace(1,lis[0].shape[1],10)
#     labels = np.linspace(md_endp, md_stap, 10)
#     int_labels = list(map(int, labels[:]))
#     plt.yticks(ticks=ticks,labels=int_labels,fontsize=13)
#     plt.tight_layout()
#     plt.savefig(figpath+'cat_sam/'+str(well_name)+'input_pred_label.jpg',dpi=300,bbox_inches='tight')










#### 下列code为了将所有的预测都画出来
# def paper_plot_input_pred_label(figwidth,y,lis,lis1,lis_copy,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name,mask_logs,
#                                 rp_dtc, dtc, dp_rhob, rhob,plot_Gardner):
#     unit = ['API','g/cm\u00B3','100Pu','us/ft']
#     if plot_Gardner == True and (mask_logs == 'DTC' or mask_logs == 'RHOB') :
#         '''plot_Gardner==True且mask_logs==RHOB或DTC时Gardner才会被使用，被绘制'''
#         #下面的+1即是为绘制Gardner的效果
#         width = lis[0].shape[0]+1
#         fig,axes = plt.subplots(1,width, figsize=(4.5*(figwidth+1),16), sharex=False,sharey=True,dpi=300)
#         if mask_logs == 'DTC':
#             axes[-1].plot(rp_dtc,y,color='r')
#             axes[-1].plot(dtc,y,color='k',linestyle='dashed')
#             axes[-1].set_title('DTC',fontsize=25)
#             axes[-1].set_xlabel('us/ft',fontsize=20)
#         else:
#             axes[-1].plot(dp_rhob,y,color='r')
#             axes[-1].plot(rhob,y,color='k',linestyle='dashed')
#             axes[-1].set_title('RHOB',fontsize=25)
#             axes[-1].set_xlabel('g/cm$^3$',fontsize=20)
#     else:
#         fig,axes = plt.subplots(1,lis[0].shape[0], figsize=(4.5*figwidth,16), sharex=False,sharey=True,dpi=300)  #sharey=True,y轴刻度值共享
    
#     #fig.suptitle(str(well_name),fontsize=15)  #添加主标题
    
#     for i in range(lis[0].shape[0]): 
#         if i == fea_litho.index(mask_logs):
#             axes[3].plot(lis[lis1.index('pred & label')][i,:],y,'r-')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
#             axes[3].plot(lis[lis_copy.index('label_1')][i,:],y,color='k',linestyle='dashed')  #同linestyle='--'
#             axes[3].set_title(fea_litho[i],fontsize=25)
#             axes[3].set_xlabel(unit[i],fontsize=20)
#         else:
#             if fea_litho.index(mask_logs) == 3:
#                 axes[i].plot(lis[lis1.index('pred & label')][i,:],y,'r-') 

#                 axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
#                 axes[i].set_title(fea_litho[i],fontsize=25)
#                 # xlim_min, xlim_max = xlim(lis_xlim,i)
#                 # axes[i].set_xlim(xlim_min, xlim_max)
#                 axes[i].set_xlabel(unit[i],fontsize=20)
#             elif fea_litho.index(mask_logs) == 2:
#                 if i == 0 or i == 1:
#                     axes[i].plot(lis[lis1.index('pred & label')][i,:],y,'r-') 

#                     axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
#                     axes[i].set_title(fea_litho[i],fontsize=25)
#                     # xlim_min, xlim_max = xlim(lis_xlim,i)
#                     # axes[i].set_xlim(xlim_min, xlim_max)
#                     axes[i].set_xlabel(unit[i],fontsize=20)
#                 else:
#                     axes[i-1].plot(lis[lis1.index('pred & label')][i,:],y,'r-') 

#                     axes[i-1].plot(lis[lis1.index('input')][i,:],y,color='k')
#                     axes[i-1].set_title(fea_litho[i],fontsize=25)
#                     # xlim_min, xlim_max = xlim(lis_xlim,i)
#                     # axes[i-1].set_xlim(xlim_min, xlim_max)
#                     axes[i-1].set_xlabel(unit[i],fontsize=20)
#             elif fea_litho.index(mask_logs) == 1:
#                 if i == 0 :
#                     axes[i].plot(lis[lis1.index('pred & label')][i,:],y,'r-') 

#                     axes[i].plot(lis[lis1.index('input')][i,:],y,color='k')
#                     axes[i].set_title(fea_litho[i],fontsize=25)
#                     # xlim_min, xlim_max = xlim(lis_xlim,i)
#                     # axes[i].set_xlim(xlim_min, xlim_max)
#                     axes[i].set_xlabel(unit[i],fontsize=20)
#                 else:
#                     axes[i-1].plot(lis[lis1.index('pred & label')][i,:],y,'r-') 

#                     axes[i-1].plot(lis[lis1.index('input')][i,:],y,color='k')
#                     axes[i-1].set_title(fea_litho[i],fontsize=25)
#                     # xlim_min, xlim_max = xlim(lis_xlim,i)
#                     # axes[i-1].set_xlim(xlim_min, xlim_max)
#                     axes[i-1].set_xlabel(unit[i],fontsize=20)
    

#     axes[0].set_ylabel('Depth(m)',fontsize=20)
#     axes[0].set_ylim(1,lis[0].shape[1])
#     ticks = np.linspace(1,lis[0].shape[1],10)
#     labels = np.linspace(md_endp, md_stap, 10)
#     int_labels = list(map(int, labels[:]))
#     plt.yticks(ticks=ticks,labels=int_labels,fontsize=13)
#     plt.tight_layout()
#     plt.savefig(figpath+'paper_fig/cat_sam/'+str(well_name)+'input_pred_label.jpg',dpi=300,bbox_inches='tight')
