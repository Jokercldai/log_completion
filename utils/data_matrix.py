import os
from re import T
import numpy as np
import pandas as pd
import random
import scipy.io as sio
import scipy.signal as signal
import argparse
import matplotlib.pyplot as plt
import seaborn as sn


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fea_litho',default=['GR','SP', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'RMED'],
                        type=list,help="feature and lithology that we need")
    parser.add_argument('--filePath',default='/home/cldai/sciresearch/logcomplete/data/Teamdata/', help='file path of train and test')  #cldai add
    parser.add_argument('--savePath',default='/home/cldai/sciresearch/logcomplete/data/Teamdata/', help='matr的保存地址')  #cldai add
    parser.add_argument("--figpath", default='/home/cldai/sciresearch/logcomplete/figure/matr/', help="数据matr图存放地址")#

    args = parser.parse_args()

    return args


def select_fea(df,fea_litho):
        '''将所有井取需要的feature'''
        well_fea = df[fea_litho].copy()
        print('well_fea.shape',well_fea.shape)
        
        return well_fea

def data_matrix(wl_fea):
    matr = np.zeros((wl_fea.shape[1],wl_fea.shape[1]))

    for i in range(0,wl_fea.shape[0]):
        print('############ i:',i)

        row = wl_fea.iloc[i,:].copy()
        row = np.array(row)
        #print('row.shape', row.shape)
        #print('row',row)
        #loc = np.where(np.isnan(row))
        loc = np.argwhere(np.isnan(row))   #np.argwhere返回位置,如果没有nan,则返回没有，即 []
        #loc = row[row.isnan().values==True].index.tolist()
        print('loc',loc)

        matr_row = np.ones((wl_fea.shape[1],wl_fea.shape[1]))
        matr_row[loc,:]=0
        matr_row[:,loc]=0
        #print('matr_row',matr_row)

        matr += matr_row  #np.argwhere(np.isnan(x))
        #print('matr',matr)

    return matr

def train_test_matr(args):
    dfTrain=pd.read_csv(args.filePath+'train1.csv',engine='python')
    dfTest=pd.read_csv(args.filePath+'test.csv',engine='python')
    df_board_test=pd.read_csv(args.filePath+'board_test.csv',engine='python') 

    train_data = select_fea(dfTrain,args.fea_litho)
    train_matr = data_matrix(train_data)

    test_data = select_fea(dfTest,args.fea_litho)
    test_matr = data_matrix(test_data)

    board_data = select_fea(df_board_test,args.fea_litho)
    board_matr = data_matrix(board_data)

    print('train_matr',train_matr)
    print('test_matr',test_matr)
    print('board_matr',board_matr)

    np.save(args.savePath+'train_matr',train_matr)
    np.save(args.savePath+'test_matr',test_matr)
    np.save(args.savePath+'board_matr',board_matr)

def fig_matr(args):
    train = np.load(args.savePath + 'train_matr' +'.npy')  #把保存在savePath中的数据读取出来
    test = np.load(args.savePath + 'test_matr' +'.npy')
    board = np.load(args.savePath+'board_matr'+'.npy')

    print('train',train)
    print('test',test)
    print('board',board)

    # fig,ax = plt.subplot(1,1,figsize=(4,4),dpi=150)
    # plt.imshow(train)
    # plt.savefig('train.svg',dpi=300,bbox_inches='tight')
    # plt.imshow(test)
    # plt.savefig('test.svg',dpi=300,bbox_inches='tight')


    df_train_matr = pd.DataFrame(train, index = args.fea_litho, columns = args.fea_litho)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_train_matr, annot=True, cmap="BuPu")
    plt.savefig(args.figpath+'train.svg',dpi=300,bbox_inches='tight')

    df_test_matr = pd.DataFrame(test, index = args.fea_litho, columns = args.fea_litho)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_test_matr, annot=True, cmap="BuPu")
    plt.savefig(args.figpath+'test.svg',dpi=300,bbox_inches='tight')

    df_board = pd.DataFrame(board,index = args.fea_litho, columns = args.fea_litho)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_board, annot=True, cmap="BuPu")
    plt.savefig(args.figpath+'board.svg',dpi=300,bbox_inches='tight')

    



def main(args):
    '''code目的在于统计一种曲线有值，其他曲线是否出现的量'''

    train_test_matr(args)
    fig_matr(args)
    

    


if __name__ == '__main__':
    args = read_args()
    main(args)

 