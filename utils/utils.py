from re import A
import time
import torch
from tqdm import tqdm
#import pytorch_ssim  #调用SSIM包
#import ssim_msssim
#from ssim_msssim import *
#from pytorch_msssim import msssim, ssim
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from matplotlib import*
from matplotlib import pyplot as plt
from math import exp
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="/home/cldai/sciresearch/logcomplete/tensorboard/ctra_exp/")

def save_model(path, model, optimizer, epoch):
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, path)



######################    ########################  #########################   ##############################
# #只有loss_miss计算
class My_loss(nn.Module):
    def __init__(self,fea_litho,bsize):
        super().__init__()
        self.fea_litho = fea_litho
        self.fea_len = len(fea_litho)
        self.bsize = bsize

    def various_loss(self,lskn,lsms,m0_d,m0_1d):
        
        # print('lskn.size()',lskn.size())
        # print('lskn[0:self.bsize*self.fea_len:self.fea_len]',lskn[0:self.bsize*self.fea_len:self.fea_len].shape)
        # print('lskn[3:self.bsize*self.fea_len:self.fea_len]',lskn[3:self.bsize*self.fea_len:self.fea_len].shape)
        
        #实际上GR不需要计算loss_miss，因为其无miss.其m0_1d中对应值均为0
        #GR_l = torch.sum(lsms[0:self.bsize*self.fea_len:self.fea_len])/(torch.sum(m0_1d[0:self.bsize*self.fea_len:self.fea_len])+1)
        RHOB_l = torch.sum(lsms[1:self.bsize*self.fea_len:self.fea_len])/(torch.sum(m0_1d[1:self.bsize*self.fea_len:self.fea_len])+1)
        NPHI_l = torch.sum(lsms[2:self.bsize*self.fea_len:self.fea_len])/(torch.sum(m0_1d[2:self.bsize*self.fea_len:self.fea_len])+1)
        DTC_l = torch.sum(lsms[3:self.bsize*self.fea_len:self.fea_len])/(torch.sum(m0_1d[3:self.bsize*self.fea_len:self.fea_len])+1)

        return RHOB_l,NPHI_l,DTC_l


    def forward(self, x, y, y_ori, mask0, maskd, y_mean,  y_std, mean, std, print_loss=False):
        '''注意打印出各类loss，看其比重
           缺失段权重注意提高'''
        #print('x.size()',x.size(),'y.size()',y.size())  #即pred和label size都为(4*bsize,512)
        #print('mask0.size()', mask0.size())
        y_mean = torch.flatten(y_mean)
        #print('y_mean', y_mean.size(), y_mean[:8])
        y_std = torch.flatten(y_std)

        f_m0 = torch.sum(mask0, dim=1)
        head_m0 = torch.where(f_m0>0, 1, 0)
        #print('head_m0.size()', head_m0.size(), head_m0)
        
        f_md = torch.sum(maskd, dim=1)
        head_md = torch.where(f_md>0, 1, 0)
        #print('head_md.size()', head_md.size(), head_md)

        head_m0_md = head_m0*(1-head_md)
        #print('torch.sum(head_m0_md)', head_m0_md.size(), torch.sum(head_m0_md))  # torch.Size([8])

        mean = torch.flatten(mean)
        #print('调整好后mean.size()', mean.size(), mean[:8])  #torch.Size([4*bsize])
        std = torch.flatten(std)
        #print('调整好后std.size()', std.size())

        # a = torch.abs(y_mean-mean)*head_m0_md
        # asum = torch.sum(a)
        # print('aaaaaa ', a.size(), a)
        # print('asumasumasum', asum)

        ##除以self.bsize是为计算每个batch的
        #print('1111111',torch.sum(torch.abs(y_mean-mean)*head_m0_md))
        sum_h = torch.sum(head_m0_md)
        if sum_h==0:
            print('@@@@@@ torch.sum(head_m0_md)==0 @@@@@')
            mean_loss = torch.sum(torch.abs(y_mean-mean)*head_m0_md)/(sum_h+1)
            std_loss = torch.sum(torch.abs(y_std-std)*head_m0_md)/(sum_h+1)
        else:
            mean_loss = torch.sum(torch.abs(y_mean-mean)*head_m0_md)/sum_h
            std_loss = torch.sum(torch.abs(y_std-std)*head_m0_md)/sum_h
        #print('mean_loss',mean_loss, 'std_loss',std_loss)
        # mean_loss = torch.sum(torch.abs(y_mean-mean)*head_m0)/torch.sum(head_m0)
        # std_loss = torch.sum(torch.abs(y_std-std)*head_m0)/torch.sum(head_m0)
        ##设置mean和std的权重，尽量使mean降得更多。
        wms = 4
        mean_std_loss = wms*mean_loss+std_loss
        #print('########## mean_std_loss', mean_std_loss )
        #print('$$$$$$$$$  mean_loss', mean_loss)
        # print('std_loss', std_loss)
        
        ##这里没有用反归一化数据计算loss，因为还是想要他们独立开
        e = x - y
        
        m = torch.pow(e, 2)
        lskn = m * mask0 * maskd   #lskn is loss_known
        lsms = m * mask0 *(1-maskd)  #lsms is loss_miss
        m0_d =  mask0 * maskd
        m0_1d = mask0 *(1-maskd)
        
        sum_mask0_maskd=torch.sum(mask0 * maskd)
        if sum_mask0_maskd==0:
            print('#$$$$$$$$$$$$ sum_mask0_maskd==0 !!!')
            loss_known = torch.sum(lskn)  / (sum_mask0_maskd+1)
        else:
            loss_known = torch.sum(lskn)  / sum_mask0_maskd
            

        w_miss = (1-maskd)
        sum_mask0_wmiss = torch.sum(mask0 * w_miss)
        if sum_mask0_wmiss == 0:
            print('########### sum_mask0_wmiss==0 !!!,此时也反映缺失处的loss为0 (loss_miss和ssim_miss均=0)')
            loss_miss = torch.sum(lsms)  / (sum_mask0_wmiss+1)
        else:
            loss_miss = torch.sum(lsms)  / sum_mask0_wmiss

        # GR_l,RHOB_l,NPHI_l,DTC_l = self.various_loss(lskn,lsms,m0_d,m0_1d)
        RHOB_l,NPHI_l,DTC_l = self.various_loss(lskn,lsms,m0_d,m0_1d)

        ssim_loss = SSIM()
        # ssim_known = 1-ssim_loss(x*mask0*maskd, y*mask0*maskd)
        ssim_miss = 1-ssim_loss(x*mask0*(1-maskd), y*mask0*(1-maskd))
        
        w2=1/4    
        # #下面为计算known和miss处的loss，若用全部loss,则需尽可能考虑known和miss的权重设置
        if print_loss == False:
            return loss_miss + 2*w2*mean_std_loss + ssim_miss  # mean_std_loss不加balance的起始大概为110左右
        else:
            #return loss_miss, 0.0005*mean_std_loss, RHOB_l, NPHI_l, DTC_l
            ## plotmean_loss 和std_loss 看一下
            return loss_miss, 2*w2*mean_std_loss, wms*w2*mean_loss, w2*std_loss, ssim_miss
######################  ######################  #########################   ############################





################## 不用maskd @@@@@@@@@@@@@@@@@@@ 不用maskd $$$$$$$$$$$$$$$$$$$$ 不用maskd############################
# # #相当于无maskd时的loss计算
# class My_loss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, y,mask0, maskd, print_loss=False):
#         '''注意打印出各类loss，看其比重
#            缺失段权重注意提高'''

#         e = x - y
#         m = torch.pow(e, 2)

#         loss = torch.sum(m*mask0) / torch.sum(mask0)

#         ssim_loss = SSIM()
#         ssim = 1-ssim_loss(x*mask0, y*mask0)
        
#         if print_loss == False:
#             return  loss + ssim 
#         else:
#             return loss, ssim
################## 不用maskd @@@@@@@@@@@@@@@@@@@ 不用maskd $$$$$$$$$$$$$$$$$$$$ 不用maskd ############################


def train_network(model,device,LR,epochs,model_filename,train_loader,test_loader,loss_name,lossfigpath,bsize,fea_litho):
    lr_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    ################ 自适应学习率 ############### 自适应学习率 ###############
    #scheduler = lr_scheduler.StepLR(optimizer,step_size=200,gamma = 0.8)  #cldaai 原gamma=0.8，每200epoch，学习率降20%
    #factor学习率调整的乘法因子，默认值为0.1。
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=80, verbose=False, 
                                                threshold=0.0001, threshold_mode= 'rel', cooldown=0, min_lr=0, eps=1e-08)
    ################ 自适应学习率 ############### 自适应学习率 ###############

    epoch_total = epochs
    loss_total = np.zeros(epoch_total)
    loss_total_test = np.zeros(epoch_total)
    start = time.time()
    ################  分别画处各类曲线的训练loss和验证loss  ################
    #GR_tra_tl = np.zeros(epoch_total)  #GR_tra_tl is GR_train_total
    RHOB_tra_tl = np.zeros(epoch_total)
    NPHI_tra_tl = np.zeros(epoch_total)
    DTC_tra_tl = np.zeros(epoch_total)

    #GR_test_tl = np.zeros(epoch_total)
    RHOB_test_tl = np.zeros(epoch_total)
    NPHI_test_tl = np.zeros(epoch_total)
    DTC_test_tl = np.zeros(epoch_total)

    mse_tr_loss = np.zeros(epoch_total)
    ssim_tr_loss = np.zeros(epoch_total)
    mse_test_loss = np.zeros(epoch_total)
    ssim_test_loss = np.zeros(epoch_total)
    ################  分别画处各类曲线的训练loss和验证loss  ################

    criterion = My_loss(fea_litho,bsize)

    '''迁移学习算法'''
    # load_model_name = '/home/cldai/sciresearch/logcomplete/model/model_save/xyz_600_8.28.pt'
    # param = torch.load(load_model_name)['net']
    # model.load_state_dict(param)
    # layer_list = ['gd0','gd1','gd2','gd3','u0','cat0','gu0','u1','cat1','gu1',
    #                 'u2','cat2','gu2','u3','cat3','gu3',]  #,'u3','cat3','gu3','result'
    # for name, param in model.named_parameters():
    #     #print('name',name)
    #     #print('param',param)
    #     if name[0:3] in layer_list or name[0:2] in layer_list or name[0:4] in layer_list:
    #         param.requires_grad = False
    #         print('######## False name[0:4]',name[0:4],param.requires_grad)
    #     if param.requires_grad == True:
    #         print('%%%%%%%% True  name[0:4]',name[0:4], param.requires_grad)
    # print('optimizer model')

    for epoch in range(epoch_total):
        print('epoch',epoch)
        model.train()
        loss_all = 0

        ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################
        loss_pa = 0
        ssim_pa = 0
        
        GR_tra = 0
        RHOB_tra = 0
        NPHI_tra = 0
        DTC_tra = 0
        ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################


        for mydata in tqdm(train_loader):
            mydata = mydata.to(device)
            optimizer.zero_grad()
            ############### 用于输出w ############## 用于输出w ####################
            output, mean, std = model(mydata)
            #output, weight = model(mydata)
            ############### 用于输出w ############## 用于输出w ####################
            mydata.y=mydata.y.float()  #cldai transformer long to float
            #print('mydata.y.shape',mydata.y.shape)
            label_ori = mydata.label_ori.float()
            
            loss = criterion(output, mydata.y, label_ori, mydata.mask, mydata.maskd, 
                            mydata.norm_mean,  mydata.norm_std, mean, std)
            #print('loss',loss)   #返回的loss是tensor,tensor(2.4895, device='cuda:5', grad_fn=<DivBackward1>)
            #print('output.shape',output.shape, 'mydata.y.shape',mydata.y.shape)   #torch.Size([1024, 8]) ,torch.Size([1024, 8])
            
            ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################
            loss_dno, ssim_dno,RHOB_l,NPHI_l,DTC_l = criterion(output, mydata.y, label_ori, mydata.mask, 
                                                    mydata.maskd, mydata.norm_mean,  mydata.norm_std, 
                                                    mean, std, print_loss = True)
            loss_pa += mydata.num_graphs * loss_dno.item()
            ssim_pa += mydata.num_graphs * ssim_dno.item()

            #GR_tra += mydata.num_graphs * GR_l.item()
            RHOB_tra += mydata.num_graphs * RHOB_l.item()
            NPHI_tra += mydata.num_graphs * NPHI_l.item()
            DTC_tra += mydata.num_graphs * DTC_l.item() 
            '''num_graphs是一个节点数，mydata.num_graphs=bsize*节点数,可是打印出来显示num_graph=32=bsize,是bsise数？ '''
            ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################
            #print('mydata.num_graphs',mydata.num_graphs)

            loss.backward()
            loss_all += mydata.num_graphs * loss.item()  #
            #cldai notes:Use the gradient corresponding to the current parameter space
            optimizer.step() 

        # if epoch%1==0:
        #     print('mean',mean.size(),mean)
        #     print('std',std.size(),std)
        #     print('y_mean',mydata.norm_mean.size(), mydata.norm_mean)
        #     print('y_std',mydata.norm_std.size(), mydata.norm_std)

        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])   #将每一个epoch的学习率lr保存
        loss = loss_all/len(train_loader.dataset)
        ##len(train_loader.dataset)是train_loader中总的样本数=2560
        #print('len(train_loader.dataset)',len(train_loader.dataset))

        ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################

        loss_ = loss_pa/len(train_loader.dataset)
        ssim_ = ssim_pa/len(train_loader.dataset)
        print('train_main_loss:{:.5f}, train_head_loss:{:.5f}'.format(loss_,ssim_))

        #GR_tra = GR_tra/len(train_loader.dataset)
        RHOB_tra = RHOB_tra/len(train_loader.dataset)
        NPHI_tra = NPHI_tra/len(train_loader.dataset)
        DTC_tra =DTC_tra/len(train_loader.dataset)
        ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################

        end = time.time()

        model.eval()
        loss_all_test = 0

        ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################
        val_loss_pa = 0
        val_ssim_pa = 0

        #GR_test = 0
        RHOB_test = 0
        NPHI_test = 0
        DTC_test= 0
        ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################

        for data in tqdm(test_loader):
            data = data.to(device)
            ############### 用于输出w ############## 用于输出w ####################
            output, mean, std = model(data)
            #output, weight_te = model(data)
            ############### 用于输出w ############## 用于输出w ####################
            
            data.y=data.y.float()  #cldai transformer long to float 
            label_ori = data.label_ori.float()

            loss_test = criterion(output, data.y, label_ori, data.mask, data.maskd,
                                data.norm_mean,  data.norm_std, mean, std)

            ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################
            val_loss_dno, val_ssim_dno,RHOB_l,NPHI_l,DTC_l = criterion(output, data.y, label_ori, data.mask, data.maskd, 
                                                            data.norm_mean,  data.norm_std, mean, std,
                                                            print_loss = True)
            val_loss_pa += data.num_graphs * val_loss_dno.item()
            val_ssim_pa += data.num_graphs * val_ssim_dno.item()

            #GR_test += data.num_graphs * GR_l.item()
            RHOB_test += data.num_graphs * RHOB_l.item()
            NPHI_test += data.num_graphs * NPHI_l.item()
            DTC_test += data.num_graphs * DTC_l.item()
            ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################

            loss_all_test += data.num_graphs * loss_test.item()
            #numcorrect_test+=torch.eq(output.argmax(dim = 1), data.y.argmax(dim = 1)).sum().float().item()  #cldai add accracy
        

        loss_all_test=loss_all_test/len(test_loader.dataset)
        #acc_test = numcorrect_test/len(test_loader.dataset)
        #cldai:be used to update the learning rate of the optimizer

        ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################
        val_loss_ = val_loss_pa/len(test_loader.dataset)
        val_ssim_ = val_ssim_pa/len(test_loader.dataset)
        print('val_main_loss:{:.5f}, val_head_loss:{:.5f}'.format(val_loss_, val_ssim_))

        #GR_test = GR_test/len(test_loader.dataset)
        RHOB_test = RHOB_test/len(test_loader.dataset)
        NPHI_test = NPHI_test/len(test_loader.dataset)
        DTC_test =DTC_test/len(test_loader.dataset)
        ########## 打印各loss ############ 打印各loss ################### 打印各loss ###########################

        ################ 自适应学习率 ############### 自适应学习率 ###############
        #scheduler.step() 
        scheduler.step(loss_all_test)          
        ################ 自适应学习率 ############### 自适应学习率 ###############

        loss_total[epoch] = loss
        loss_total_test[epoch] =  loss_all_test

        #GR_tra_tl[epoch] = GR_tra  #GR_tra_tl is GR_train_total
        RHOB_tra_tl[epoch] = RHOB_tra
        NPHI_tra_tl[epoch] = NPHI_tra
        DTC_tra_tl[epoch] = DTC_tra

        #GR_test_tl[epoch] = GR_test
        RHOB_test_tl[epoch] = RHOB_test
        NPHI_test_tl[epoch] = NPHI_test
        DTC_test_tl[epoch] = DTC_test

        mse_tr_loss[epoch] = loss_
        ssim_tr_loss[epoch] = ssim_
        mse_test_loss[epoch] = val_loss_
        ssim_test_loss[epoch] = val_ssim_

        writer.add_scalars('loss',{'train': loss_total[epoch],
                                  'test': loss_total_test[epoch]}, epoch)    #tag(标题):'loss'； y轴：{} 里的; x轴：epoch
        writer.add_scalars('lr',{'lr': lr_list[epoch]}, epoch)
        # writer.add_scalars('train fea_loss',{'RHOB':RHOB_tra_tl[epoch],
        #                                     'NPHI':NPHI_tra_tl[epoch], 'DTC':DTC_tra_tl[epoch]},epoch)
        # writer.add_scalars('validation fea_loss', {'RHOB':RHOB_test_tl[epoch],
        #                                             'NPHI':NPHI_test_tl[epoch], 'DTC':DTC_test_tl[epoch]},epoch)
        writer.add_scalars('train mean VS std',{'mean':RHOB_tra_tl[epoch],
                                            'std':NPHI_tra_tl[epoch] },epoch)
        writer.add_scalars('validation mean VS std', {'mean':RHOB_test_tl[epoch],
                                                    'std':NPHI_test_tl[epoch] },epoch)
        writer.add_scalars('train main vs head',{'main':mse_tr_loss[epoch], 'head':ssim_tr_loss[epoch]
                                            },epoch)
        writer.add_scalars('val main vs head',{'main':mse_test_loss[epoch], 'head':ssim_test_loss[epoch]
                                            },epoch)

        ############## 这个地方可能改动 ############# 这个地方可能改动 ############# 这个地方可能改动 ############# 这个地方可能改动 #############
        if epoch % 50 == 0:
            savename = 'xyz_' + str(epoch) + '.pt'
            filename = model_filename + savename
            save_model(filename,model,optimizer,epoch)
        # if epoch % 50 ==0:
        #     plot_figure(epoch_total, loss_total, loss_total_test,lossfigpath)
        ############## 这个地方可能改动 ############# 这个地方可能改动 ############# 这个地方可能改动 ############# 这个地方可能改动 #############

        if epoch%1==0:
            # print('Epoch: {:04d}, Loss: {:.5f} , Val_Loss: {:.5f}, Train_acc:{:.5f}, Val_acc:{:.5f}, Time:{:.3f}, lr:{:.7f}'.
            #       format(epoch, loss, loss_all_test, acc_train, acc_test, end-start, lr_list[epoch]))  # ori:loss/6.75=trainsample/valsample
            #     ###########cldai note ratio 470100/30822=15.25(loss/15.25); 153731/10173=15.11##################
            print('Epoch: {:04d}, Train_Loss: {:.5f} , Val_Loss: {:.5f}, Time:{:.3f}, lr:{:.7f}'.
                  format(epoch, loss, loss_all_test, end-start, lr_list[epoch]))  
    writer.close()           

    ####cldai adds: plot loss and accuracy######
    plot=True
    if plot:
        x = [i for i in range(epoch_total)]
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1,2,1)
        ax.plot(x, smooth(loss_total, 0.6), label='train loss')
        ax.plot(x, smooth(loss_total_test, 0.6), label='validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Train and validation loss curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize = 15)

        ax0 = fig.add_subplot(1,2,2)
        ax0.plot(x, lr_list, label='lr')
        ax0.set_xlabel('Epoch', fontsize=15)
        ax0.set_ylabel('lr', fontsize=15)
        ax0.set_title(f'Adaptive learning rate', fontsize=15)
        ax0.grid(True)
        plt.legend(loc='upper right', fontsize = 15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(lossfigpath+'loss_lr.png')
        
    
        fig2=plt.figure(figsize=(12,11))
        ##################  去掉SP  ###################  去掉SP  ##################
        labelList=['RHOB','NPHI','DTC']
        train_fea = [RHOB_tra_tl,NPHI_tra_tl,DTC_tra_tl]
        test_fea = [RHOB_test_tl,NPHI_test_tl,DTC_test_tl]
        ##################  去掉SP  ###################  去掉SP  ##################
        ax2=fig2.add_subplot(2,1,1)
        for j2 in range(len(labelList)):
            ax2.plot(x,smooth(train_fea[j2],0.6),label=labelList[j2])
        ax2.set_xlabel('Epoch',fontsize=15)
        ax2.set_ylabel('loss',fontsize=15)
        ax2.set_title(f' The loss of different logs (Train set)',fontsize=15)
        ax2.grid(True)
        plt.legend(loc='upper right',fontsize=10)
        ax2=fig2.add_subplot(2,1,2)
        for j in range(len(labelList)):
            ax2.plot(x,smooth(test_fea[j],0.6),label=labelList[j])
        ax2.set_xlabel('Epoch',fontsize=15)
        ax2.set_ylabel('loss',fontsize=15)
        ax2.set_title(f'The loss of different logs (Validation set)',fontsize=15)
        ax2.grid(True)
        plt.legend(loc='upper right',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(lossfigpath+'feature_loss.png')



        fig3 = plt.figure(figsize=(12,5))
        labellist3=['MSE','SSIM']
        train_type=[mse_tr_loss,ssim_tr_loss]
        test_type=[mse_test_loss,ssim_test_loss]
        ax3 = fig3.add_subplot(1,2,1)
        for j3 in range(len(labellist3)):
            ax3.plot(x, smooth(train_type[j3], 0.6), label=labellist3[j3])
        ax3.set_xlabel('Epoch', fontsize=15)
        ax3.set_ylabel('Loss', fontsize=15)
        ax3.set_title(f'The type of train loss ', fontsize=15)
        ax3.grid(True)
        plt.legend(loc='upper right',fontsize=15)

        ax3 = fig3.add_subplot(1,2,2)
        for j3i in range(len(labellist3)):
            ax3.plot(x, smooth(test_type[j3i], 0.6), label=labellist3[j3i])
        ax3.set_xlabel('Epoch', fontsize=15)
        ax3.set_ylabel('Loss', fontsize=15)
        ax3.set_title(f'The type of test loss', fontsize=15)
        ax3.grid(True)
        plt.legend(loc='upper right', fontsize = 15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(lossfigpath+'MSE_SSIM.png')
        

    print('Finished Training!')
    print(loss_name)
    np.savez(loss_name,loss_total,loss_total_test)



def plot_figure(epoch_total, loss_total, loss_total_test,lossfigpath):
    x = [i for i in range(epoch_total)]
    fig = plt.figure(figsize=(12,5))
    plt.plot(x, smooth(loss_total, 0.6), label='train loss')
    plt.plot(x, smooth(loss_total_test, 0.6), label='validation loss')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title(f'train and validation loss curve', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize = 15)
    plt.savefig(lossfigpath+'loss_epoch.png')   

###cldai adds#############################
def smooth(v, w= 0.85):
	last = v[0]
	smoothed = []
	for point in v:
		smoothed_val = last*w+(1-w)*point
		smoothed.append(smoothed_val)
		last = smoothed_val
	return smoothed
####cldai adds###############################



############## ssim ############## ssim ############## ssim ############## ssim ##############
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # print('img1.device,img1.type()',img1.device,img1.type())  #cuda:1 torch.cuda.FloatTensor
    # print('window.device,window.type()',window.device,window.type()) #原先是cpu torch.FloatTensor
    # print('window.size()',window.size())  #window.size() torch.Size([1, 1, 11, 11])

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)

    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        
        ############## cldai ############## cldai ############## cldai ############## 
        img1 = img1.unsqueeze(0)
        img1 = img1.unsqueeze(0)
        #print('img1.size()',img1.size())  #torch.Size([1, 1, 224, 64])
        img2 = img2.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        #print('img2.size()',img2.size())  #torch.Size([1, 1, 224, 64])
        ############## cldai ############## cldai ############## cldai ############## 

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        ############## cldai ############## cldai ############## cldai ############## 
        window = window.to(img1.device)
        ############## cldai ############## cldai ############## cldai ############## 
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


############## ssim ############## ssim ############## ssim ############## ssim ##############


