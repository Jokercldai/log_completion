import numpy as np
import matplotlib.pyplot as plt

###############################################################################################################
# def plot_figure2(input_sam_order,prediction_sam_order,label_sam_order,
#                 mask0_sam_order,maskd_sam_order,fea_litho,label_intact=False,pnl=False,order=1,muti_sam=False):  
               
#     """pnl:缩写pred(need)_label。true：第三张图画pred_need_label，即需要预测的地方和label，False:第三张图画整条预测和label"""

#     figwidth = input_sam_order.shape[0]
#     fig=plt.figure(figsize=(2*figwidth,16),dpi=300)  #三行七列时（12，12）  #figsize=(width,height)
#     y=np.linspace(1, input_sam_order.shape[1], input_sam_order.shape[1])

#     input_sam_order_1=np.where(mask0_sam_order==0, np.nan, input_sam_order)
#     input_sam_order_2=np.where(maskd_sam_order==0, np.nan, input_sam_order_1)
#     #label只涉及到null值，即考虑不画null值时，用mask0即可，不涉及到maskd
#     if label_intact==True:
#         label_sam_order_1=label_sam_order
#     else:
#         label_sam_order_1=np.where(mask0_sam_order==0, np.nan, label_sam_order)  

#     prediction_sam_order_1=np.where(mask0_sam_order==0, prediction_sam_order, label_sam_order)
#     pred_need_label=np.where(maskd_sam_order==0, prediction_sam_order, prediction_sam_order_1)

#     if label_intact==True:
#         error_lab_pred =label_sam_order-prediction_sam_order
#         error_lab_pred_for_xlim =label_sam_order-prediction_sam_order
#         #print(label_sam_order,prediction_sam_order)
#     else:    
#         #在mask0==0的地方，即label处为nan值，则error设为np.nan,在mask0！=0地方，error=
#         error_lab_pred =np.where(mask0_sam_order==0, np.nan, label_sam_order-prediction_sam_order) 
#         #与上面不同之处在于np.nan 变 0,这样在画error图的xlim时不报错
#         error_lab_pred_for_xlim =np.where(mask0_sam_order==0, 0, label_sam_order-prediction_sam_order)
    
#     if pnl==True:
#         lis=[input_sam_order_2, prediction_sam_order, pred_need_label, label_sam_order_1, error_lab_pred]
#     else:
#         lis=[input_sam_order_2, prediction_sam_order, prediction_sam_order, label_sam_order_1, error_lab_pred]
#     print('len(lis)',len(lis))  #len=3,列表的长度是按照最外围计算的
    
#     lis_copy=['input_sam_order_2', 'prediction_sam_order', 'pred_need_label', 'label_sam_order_1', 'error_lab_pred']

#     #lis_xlim不需要加pred_need_label，因为pred_need_label的值由input和pred得来，
#     #其所以xlim值由input和pred即可得
#     lis_xlim=[input_sam_order, prediction_sam_order, label_sam_order]

#     lis1=['input', 'pred', 'pred_label', 'error']
#     count=1
#     for item in range(len(lis1)):
        
#         for p in range(lis[item].shape[0]):  #p
#             ax=fig.add_subplot(len(lis1), lis[0].shape[0], count)  #3行7列，从左到右第count个
            
#             if item==lis1.index('input'):
                
                
#                 ax.plot(lis[item][p,:],y)

#             elif item==lis1.index('pred'):
#                 ax.plot(lis[item][p,:],y, 'r-')
            
#             elif item==lis1.index('pred_label'):
#                 ax.plot(lis[lis1.index('pred_label')][p,:],y, 'r-')  #list不能直接按里面的value取，变相通过index取
#                 ax.plot(lis[lis_copy.index('label_sam_order_1')][p,:],y) 

#             elif item==lis1.index('error'):
#                 ax.plot(lis[lis_copy.index('error_lab_pred')][p,:],y)

#             if item==0:
#                 ax.set_title(fea_litho[p],fontsize=15)
#             if p==0:
#                 ax.set_ylabel(lis1[item],fontsize=15)

#             ax.set_ylim(1,lis[0].shape[1])


#             '''注意这里我现在改了下，将input横坐标与预测、label不保持一致'''
#             if item != lis1.index('error') and item != lis1.index('input'):  #由于error_lab_pred包含很多值，所以lis.index(error_lab_pred)报错
#                 xlim_min, xlim_max = xlim(lis_xlim,p)
#                 ax.set_xlim(xlim_min, xlim_max)
#             elif item == lis1.index('error'):
#                 ax.set_xlim(error_lab_pred_for_xlim[p,:].min(), error_lab_pred_for_xlim[p,:].max())
#             # if item != lis1.index('error'):  #由于error_lab_pred包含很多值，所以lis.index(error_lab_pred)报错
#             #     xlim_min, xlim_max = xlim(lis_xlim,p)
#             #     ax.set_xlim(xlim_min, xlim_max)
#             # elif item == lis1.index('error'):
#             #     ax.set_xlim(error_lab_pred_for_xlim[p,:].min(), error_lab_pred_for_xlim[p,:].max())



#             if count%lis[item].shape[0]!=1:
#                 ax.axes.yaxis.set_ticklabels([])  #去除y轴刻度值，有刻度线
                
#             count+=1
    
#     plt.tight_layout()

#     if muti_sam == True:
#         plt.savefig('/home/cldai/sciresearch/logcomplete/figure/input_pred_label对比图/muti_sam/input_pred_label_RHOB'+str(order)+'.png')
#     else:
#         plt.savefig('/home/cldai/sciresearch/logcomplete/figure/input_pred_label对比图/input_pred_label_'+str(order)+'.svg')




# def xlim(lis_xlim,i):  #lis[0],[1],[2]的shape：7*batch_size, 64
#     '''选择合适的横坐标范围'''
#     input_min=lis_xlim[0][i,:].min()
#     input_max=lis_xlim[0][i,:].max()
#     pred_min=lis_xlim[1][i,:].min()
#     pred_max=lis_xlim[1][i,:].max()
#     label_min=lis_xlim[2][i,:].min()
#     label_max=lis_xlim[2][i,:].max()
    
#     xlim_left=[input_min, pred_min, label_min]
#     xlim_right=[input_max, pred_max, label_max]
#     #print(xlim_left,xlim_right)

#     return min(xlim_left), max(xlim_right)



#########################################################################################


def plot_figure2(input_sam_order,prediction_sam_order,label_sam_order,
                mask0_sam_order,maskd_sam_order,fea_litho,label_intact=False,pnl=False,order=1,muti_sam=False):  
               
    """pnl:缩写pred(need)_label。true：第三张图画pred_need_label，即需要预测的地方和label，False:第三张图画整条预测和label"""

    figwidth = input_sam_order.shape[0]
    #图形长由16改为18
    fig=plt.figure(figsize=(2*figwidth,18),dpi=300)  #三行七列时（12，12）  #figsize=(width,height)
    y=np.linspace(1, input_sam_order.shape[1], input_sam_order.shape[1])

    input_sam_order_1=np.where(mask0_sam_order==0, np.nan, input_sam_order)
    input_sam_order_2=np.where(maskd_sam_order==0, np.nan, input_sam_order_1)
    #label只涉及到null值，即考虑不画null值时，用mask0即可，不涉及到maskd
    if label_intact==True:
        label_sam_order_1=label_sam_order
    else:
        label_sam_order_1=np.where(mask0_sam_order==0, np.nan, label_sam_order)  

    prediction_sam_order_1=np.where(mask0_sam_order==0, prediction_sam_order, label_sam_order)
    pred_need_label=np.where(maskd_sam_order==0, prediction_sam_order, prediction_sam_order_1)

    if label_intact==True:
        error_lab_pred =label_sam_order-prediction_sam_order
        error_lab_pred_for_xlim =label_sam_order-prediction_sam_order
        #print(label_sam_order,prediction_sam_order)
    else:    
        #在mask0==0的地方，即label处为nan值，则error设为np.nan,在mask0！=0地方，error=
        error_lab_pred =np.where(mask0_sam_order==0, np.nan, label_sam_order-prediction_sam_order) 
        #与上面不同之处在于np.nan 变 0,这样在画error图的xlim时不报错
        error_lab_pred_for_xlim =np.where(mask0_sam_order==0, 0, label_sam_order-prediction_sam_order)
    
    if pnl==True:
        lis=[input_sam_order_2, prediction_sam_order, pred_need_label, label_sam_order_1, error_lab_pred]
    else:
        lis=[input_sam_order_2, prediction_sam_order, prediction_sam_order, label_sam_order_1, error_lab_pred]
    print('len(lis)',len(lis))  #len=3,列表的长度是按照最外围计算的
    
    lis_copy=['input_sam_order_2', 'prediction_sam_order', 'pred_need_label', 'label_sam_order_1', 'error_lab_pred']

    #lis_xlim不需要加pred_need_label，因为pred_need_label的值由input和pred得来，
    #其所以xlim值由input和pred即可得
    lis_xlim=[input_sam_order, prediction_sam_order, label_sam_order]

    lis1=['input', 'pred', 'pred_label', 'error', 'label']
    count=1
    for item in range(len(lis1)):
        
        #print('lis[item].shape[0]',lis[item].shape[0])
        for p in range(lis[item].shape[0]):  #p
            ax=fig.add_subplot(len(lis1)+1, lis[0].shape[0], count)  #3行7列，从左到右第count个
            
            if item==lis1.index('input'):
                ax.plot(lis[item][p,:],y)

            elif item==lis1.index('pred'):
                ax.plot(lis[item][p,:],y, 'r-')
            
            elif item==lis1.index('pred_label'):
                ax.plot(lis[lis1.index('pred_label')][p,:],y, 'r-')  #list不能直接按里面的value取，变相通过index取
                ax.plot(lis[lis_copy.index('label_sam_order_1')][p,:],y) 

            elif item==lis1.index('error'):
                ax.plot(lis[lis_copy.index('error_lab_pred')][p,:],y)
            
            elif item==4:
                #添加一列label列
                ax.plot(lis[lis_copy.index('label_sam_order_1')][p,:],y) 

            if item==0:
                ax.set_title(fea_litho[p],fontsize=15)
            if p==0:
                ax.set_ylabel(lis1[item],fontsize=15)

            ax.set_ylim(1,lis[0].shape[1])


            '''注意这里我现在改了下，将input横坐标与预测、label不保持一致'''
            if item == lis1.index('pred') or item == lis1.index('pred_label'):  #由于error_lab_pred包含很多值，所以lis.index(error_lab_pred)报错
                xlim_min, xlim_max = xlim(lis_xlim,p)
                ax.set_xlim(xlim_min, xlim_max)
            elif item == lis1.index('error'):
                ax.set_xlim(error_lab_pred_for_xlim[p,:].min(), error_lab_pred_for_xlim[p,:].max())
            # if item != lis1.index('error'):  #由于error_lab_pred包含很多值，所以lis.index(error_lab_pred)报错
            #     xlim_min, xlim_max = xlim(lis_xlim,p)
            #     ax.set_xlim(xlim_min, xlim_max)
            # elif item == lis1.index('error'):
            #     ax.set_xlim(error_lab_pred_for_xlim[p,:].min(), error_lab_pred_for_xlim[p,:].max())



            if count%lis[item].shape[0]!=1:
                ax.axes.yaxis.set_ticklabels([])  #去除y轴刻度值，有刻度线
                
            count+=1
    
    plt.tight_layout()

    if muti_sam == True:
        plt.savefig('/home/cldai/sciresearch/logcomplete/figure/input_pred_label对比图/muti_sam/input_pred_label_RHOB'+str(order)+'.jpg')
    else:
        plt.savefig('/home/cldai/sciresearch/logcomplete/figure/input_pred_label对比图/input_pred_label_'+str(order)+'.jpg')




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