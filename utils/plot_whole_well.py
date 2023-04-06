import numpy as np
import matplotlib.pyplot as plt

###############################################################################################################
def plt_whole_well(input,pred,label,mask0,maskd,fea_litho,
                md_stap,md_endp,figpath,well_name,label_intact=False,pnl=False):  

    figwidth = input.shape[0]
    fig=plt.figure(figsize=(2*figwidth,16),dpi=300)  #三行七列时（12，12）  #figsize=(width,height)
    y=np.linspace(input.shape[1], 1, input.shape[1])  #注意画图时坐标轴及曲线的正反

    input_afm0=np.where(mask0==0, np.nan, input)  #input_afm0=input_after_mask0_for_fig
    input_afmd=np.where(maskd==0, np.nan, input_afm0)  #input_afmd = input_after_maskd_for_fig
    #label只涉及到null值，即考虑不画null值时，用mask0即可，不涉及到maskd
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
    
    lis_copy=['input_afmd', 'pred', 'pred_need_label', 'label_1']

    #lis_xlim不需要加pred_need_label，因为pred_need_label的值由input和pred得来，
    #其所以xlim值由input和pred即可得
    lis_xlim=[input, pred, label]

    lis1=['input', 'pred', 'pred & label']

    plot_input(figwidth,y,lis,lis1,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name)
    plot_pred_label(figwidth,y,lis,lis1,lis_copy,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name)


def plot_input(figwidth,y,lis,lis1,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name):
    fig,axes = plt.subplots(1,lis[0].shape[0], figsize=(2*figwidth,16), sharex=False,sharey=True,dpi=300)  #sharey=True,y轴刻度值共享
    for i in range(lis[0].shape[0]):
        axes[i].plot(lis[lis1.index('input')][i,:],y)  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
        axes[i].set_title(fea_litho[i],fontsize=15)
        xlim_min, xlim_max = xlim(lis_xlim,i)
        axes[i].set_xlim(xlim_min, xlim_max)
    axes[0].set_ylabel('Input',fontsize=15)
    axes.set_ylim(1,lis[0].shape[1])
    ticks = np.linspace(1,lis[0].shape[1],10)
    labels = np.linspace(md_endp, md_stap, 10)
    int_labels = list(map(int, labels[:]))
    plt.yticks(ticks=ticks,labels=int_labels,fontsize=13)
    plt.savefig(figpath+'whole_well'+str(well_name)+'_input.png',dpi=300,bbox_inches='tight')


def plot_pred_label(figwidth,y,lis,lis1,lis_copy,lis_xlim,fea_litho,md_stap,md_endp,figpath,well_name):
    fig,axes = plt.subplots(1,lis[0].shape[0], figsize=(2*figwidth,16), sharex=False,sharey=True,dpi=300)  #sharey=True,y轴刻度值共享
    for i in range(lis[0].shape[0]): 
        axes[i].plot(lis[lis1.index('pred & label')][i,:],y,'r-')  #因为是1行，所以用axes[i]一维来调用，而不是axes[1,i]来调用
        axes[i].plot(lis[lis_copy.index('label_1')][i,:],y)
        axes[i].set_title(fea_litho[i],fontsize=15)
        xlim_min, xlim_max = xlim(lis_xlim,i)
        axes[i].set_xlim(xlim_min, xlim_max)
    axes[0].set_ylabel('Pred(red)&Label(blue)',fontsize=15)
    axes.set_ylim(1,lis[0].shape[1])
    ticks = np.linspace(1,lis[0].shape[1],10)
    labels = np.linspace(md_endp, md_stap, 10)
    int_labels = list(map(int, labels[:]))
    plt.yticks(ticks=ticks,labels=int_labels,fontsize=13)
    plt.savefig(figpath+'whole_well'+str(well_name)+'_pred_label.png',dpi=300,bbox_inches='tight')


def xlim(lis_xlim,i):  #lis[0],[1],[2]的shape：7*batch_size, 64
    '''选择合适的横坐标范围'''
    input_min=lis_xlim[0][i,:].min()
    input_max=lis_xlim[0][i,:].max()
    pred_min=lis_xlim[1][i,:].min()
    pred_max=lis_xlim[1][i,:].max()
    label_min=lis_xlim[2][i,:].min()
    label_max=lis_xlim[2][i,:].max()
    
    xlim_left=[input_min, pred_min, label_min]
    xlim_right=[input_max, pred_max, label_max]
    #print(xlim_left,xlim_right)

    return min(xlim_left), max(xlim_right)