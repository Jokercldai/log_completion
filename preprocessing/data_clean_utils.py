'''用于数据清洗文件的补充：放置各种处理函数'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal as signal
from skimage.measure import block_reduce
from scipy.interpolate import interp1d


def plot_well(data,fea):
    '''绘制整口井'''
    
    figwidth = data.shape[1]

    y=np.linspace(1, data.shape[0], data.shape[0])  
    unit = ['API','g/cm\u00B3','100Pu','us/ft']
    
    fig,axes = plt.subplots(1,data.shape[1], figsize=(3*figwidth,16), sharex=False,sharey=True,dpi=300)  
    
    for i in range(figwidth):
        axes[i].plot(data.iloc[:,i],y,color='k')
        axes[i].set_title(fea[i],fontsize=25)
        axes[i].set_xlabel(unit[i],fontsize=20)
    axes[0].set_ylabel('Depth(m)',fontsize=20)
    axes[0].set_ylim(data.shape[0],1)


def plot_well_by_depth(data,fea):
    '''绘制整口井'''

    figwidth = data[fea].shape[1]
    #unit = ['API','g/cm\u00B3','100Pu','us/ft']
    
    fig,axes = plt.subplots(1,data[fea].shape[1], figsize=(3*figwidth,16), sharex=False,
                            sharey=True,dpi=300)  
    for i in range(figwidth):
        axes[i].plot(data[fea].iloc[:,i],data['DEPTH_MD'],color='k')
        axes[i].set_title(fea[i],fontsize=25)
        #axes[i].set_xlabel(unit[i],fontsize=20)

    axes[0].set_ylabel('Depth(m)',fontsize=20)
    axes[0].set_ylim(data['DEPTH_MD'].max(),data['DEPTH_MD'].min())

def plot_RDEP_by_depth(data,fea, r_name, color):
    '''绘制整口井'''

    figwidth = 1
    #unit = ['API','g/cm\u00B3','100Pu','us/ft']
    fig_hei = int(data.shape[0]/2000)
    print('fig_hei', fig_hei)
    fig,axes = plt.subplots(1,1, figsize=(3*figwidth,fig_hei), sharex=False,
                            sharey=True,dpi=300)  
    #print('111', data[[fea]])
    
    axes.plot(data[fea],data['DEPTH_MD'],color=color)
        #axes[i].set_title(fea[i],fontsize=25)
        #axes[i].set_xlabel(unit[i],fontsize=20)

    #axes[0].set_ylabel('Depth(m)',fontsize=20)
    axes.set_ylim(data['DEPTH_MD'].max(),data['DEPTH_MD'].min())

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/cldai/sciresearch/logcomplete/figure/RDEP_fig/'+str(r_name)+'.png',dpi=300,bbox_inches='tight')

def plot_GR_RDEP_DTC_by_depth(data,fea, r_name):
    '''绘制整口井'''

    figwidth = len(fea)
    #unit = ['API','g/cm\u00B3','100Pu','us/ft']
    fig_hei = 12
    #print('fig_hei', fig_hei)
    fig,axes = plt.subplots(1,figwidth, figsize=(3*figwidth,fig_hei), sharex=False,
                            sharey=True,dpi=300)  
    #print('111', data[[fea]])
    y=np.linspace(1, data.shape[0], data.shape[0])  
    
    for i in range(figwidth):
        axes[i].plot(data[fea].iloc[:,i],y,color='k', linewidth=3)
        #axes[i].set_title(fea[i],fontsize=25)
        #axes[i].set_xlabel(unit[i],fontsize=20)

        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)
        axes[i].axis('off')

    #axes[0].set_ylabel('Depth(m)',fontsize=20)
    axes[0].set_ylim(data.shape[0],1)

    

    #plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/cldai/sciresearch/logcomplete/figure/RDEP_fig/'+str(r_name)+'.jpg',dpi=300,bbox_inches='tight')


def plot_RMED_well(data,fea, r_name, color):
    '''绘制整口井'''
    
    figwidth = 1

    fig_hei = int(data.shape[0]/2000)
    print('fig_hei', fig_hei)
    fig,axes = plt.subplots(1,1, figsize=(3*figwidth,fig_hei), sharex=False,
                            sharey=True,dpi=300)

    y=np.linspace(1, data.shape[0], data.shape[0])  

    axes.plot(data[fea],y,color=color)
    #axes[i].set_title(fea[i],fontsize=25)
    #axes[i].set_xlabel(unit[i],fontsize=20)
    #axes[0].set_ylabel('Depth(m)',fontsize=20)
    axes.set_ylim(data.shape[0],1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/cldai/sciresearch/logcomplete/figure/RDEP_fig/'+str(r_name)+'.png',dpi=300,bbox_inches='tight')



def medfilt_by_need(data, filt_size, fea_need_filt ):
    '''两次中值滤波'''

    lab_cop=data.copy()

    for i, col in enumerate(fea_need_filt):
        print('col', col)
        #signal.medfilt返回ndarray数组
        lab_cop[col]=signal.medfilt(lab_cop[col],kernel_size=filt_size)

    #第二次进行中值滤波
    for i, col in enumerate(fea_need_filt):
        #signal.medfilt返回ndarray数组
        lab_cop[col]=signal.medfilt(lab_cop[col],kernel_size=filt_size)

    return lab_cop




def resample(input,down):
    '''将input进行下采样，再样条插值回去'''
    '''para: down:下采样倍数'''
    #input = input.iloc[:-2, :]
    input = input.T
    sam_len = input.shape[1]
    print('数据的维度', input.shape)

    #用block_reduce下采样
    input_d = block_reduce(input,block_size=(1,down), func=np.median)  #input_d即input_down   #cval=np.median(input) 是当非整除时，以cval来填充
    #print('input_d.shape',input_d.shape)
    
    input_ip = np.zeros((input_d.shape[0], sam_len))  #input_ip is input_interp
    for i in range(input_d.shape[0]):
        input_ip[i,:] = interp(input_d[i,:],i,down, sam_len)

    #print('np.isnan(input_ip).any()  True说明有nan值: ',np.isnan(input_ip).any())
    input_ip = input_ip.T

    print('input_ip.shape', input_ip.shape)

    input_ip = pd.DataFrame(input_ip, columns = ['RDEP'])
    return input_ip


def interp(seq,i,down, sam_len):
    #print('len(seq)',len(seq))
    x = np.linspace(1,len(seq),len(seq))
    y = seq
    #f是一个关于x和y的样条函数，这样再利用f来插值。'zero', 'slinear', 'quadratic', 'cubic'分别对应0，1，2，3次样条函数
    f = interp1d(x,y,kind='quadratic')   #二次样条函数
    xnew = np.linspace(1,len(seq),sam_len)
    ynew = f(xnew)

    # #
    # fig = plt.figure(figsize=(10,6),dpi=300)
    # plt.plot(x,y,label=self.fea_litho[i]+'-d'+str(down)+'-before interp')
    # plt.plot(xnew,ynew,'r',label=self.fea_litho[i]+'-d'+str(down)+'-after interp')
    # plt.legend(loc='upper right',fontsize=10)
    # plt.tight_layout()
    # plt.savefig('/home/cldai/sciresearch/logcomplete/figure/loda_data_check/'+self.fea_litho[i]+'-d'+str(down)+'.png')

    return ynew











def plot_well_fea_gp_litho(well_data):
    
    lithology_numbers = {30000: 0,  #Sandstone
                 65030: 1,  #Sandstone/Shale
                 65000: 2,  #Shale
                 80000: 3,  #Mart
                 74000: 4,  #Dolomite
                 70000: 5,  #Limestone
                 70032: 6,  #Chalk
                 88000: 7,  #Halite
                 86000: 8,  #Anhydrite
                 99000: 9,  #Tuff
                 90000: 10,  #Coal
                 93000: 11  #Basement
                }
    group_numbers = {'NORDLAND GP.':0,
                    'HORDALAND GP.':1,
                    'ROGALAND GP.':2,
                    'SHETLAND GP.':3,
                    'CROMER KNOLL GP.':4,
                    'VIKING GP.': 5,
                    'VESTLAND GP.': 6,
                    'ZECHSTEIN GP.':7,
                    'HEGRE GP.': 8,
                    'ROTLIEGENDES GP.': 9,
                    'TYNE GP.': 10,
                    'BOKNFJORD GP.': 11,
                    'DUNLIN GP.':12,
                    'BAAT GP.':13,
                    np.nan: 14}

    wdata = well_data.copy()
    wdata.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY': 'LITHOLOGY'},inplace=True)
    #print('wdata.columns', wdata.columns)
    #print('该井岩性种类',wdata['LITHOLOGY'].value_counts()) #ascending=False,默认降序排列
    wdata['LITHOLOGY'] = wdata['LITHOLOGY'].map(lithology_numbers)
    #print('映射后 该井岩性种类',wdata['LITHOLOGY'].value_counts()) 
    #print('wdata.LITHOLOGY', wdata.LITHOLOGY)
    #print('该井group种类以及数目', wdata['GROUP'].value_counts()) 
    wdata['GROUP'] = wdata['GROUP'].map(group_numbers)
    #print('映射后该井group种类以及数目', wdata['GROUP'].value_counts()) 
    #print("wdata[['GROUP']]", wdata[['GROUP']].shape, wdata[['GROUP']])

    figwidth = len(wdata.columns)-len(['WELL', 'DEPTH_MD'])
    fig, axs = plt.subplots(1, figwidth, figsize=(3*figwidth,16), 
                                sharey=True, dpi=300)
    #print('set(wdata.columns)', set(wdata.columns))
    for ic, col in enumerate(set(wdata.columns)-set(['WELL', 'DEPTH_MD', 
                                'LITHOLOGY', 'GROUP'])):                                             
        #print('ic',ic)   #ic是序号
        #print('col',col)  #col是其中元素，如columns中的feature
        axs[ic].plot(wdata[col], wdata['DEPTH_MD'], color='k')
        axs[ic].set_title(col, fontsize=25)

    # ##vmin和vmax由group_numbers的范围来确定
    # axs[-2].imshow(wdata[['GROUP']],interpolation='nearest',
    #                 vmin=0, vmax=14000, aspect='auto')  #cmap="jet"
    # axs[-2].set_title('GROUP', fontsize=25)
    # axs[-2].set_xticks([])
    # axs[-1].imshow(wdata[['LITHOLOGY']],interpolation='nearest',
    #                 vmin=0, vmax=11000, aspect='auto')
    # axs[-1].set_title('LITHOLOGY', fontsize=25)
    # axs[-1].set_xticks([])
    
    axs[0].set_ylim(wdata['DEPTH_MD'].values[-1], wdata['DEPTH_MD'].values[0])
    axs[0].set_ylabel('Depth(m)',fontsize=20)

    axs[-2].plot(wdata['GROUP'], wdata['DEPTH_MD'], color='k')
    axs[-2].set_title('GROUP', fontsize=25)
    axs[-1].plot(wdata['LITHOLOGY'], wdata['DEPTH_MD'], color='k')
    axs[-1].set_title('LITHOLOGY', fontsize=25)


def standard_by_group(ori_data, stand_info, refer_well, fea):
    '''按照视标准井的每个GROUP分别归一化每口井对应的GROUP数据。'''

    data = ori_data.copy()
    #print('data.columns', data.columns)

    gp_name = refer_well.index.unique()
    #print('视标准井GROUP名', gp_name)

    wells_name = data['WELL'].unique()
    wells_num = len(data['WELL'].unique())
    #print('数据集井数目', wells_num)

    mfn = ['GR_m', 'RHOB_m', 'NPHI_m', 'DTC_m']  #mfn = mean_fea_name
    sfn = ['GR_s', 'RHOB_s', 'NPHI_s', 'DTC_s']  #sfn = std_fea_name

    for i, wn in enumerate(wells_name):
        print('当前井名', wn)

        well = data[data.WELL == wn].copy()  #wn  '16/7-4'  #'17/4-1'
        #print('well  之后打印全井数据看看', well.head())
        well_groups = well['GROUP'].unique()
        #print('well_groups', well_groups)

        info_well = stand_info[stand_info['WELL'] == wn].copy()

        for i, gp in enumerate(well_groups):
            #print('GROUP', gp)

            if gp is not np.nan:

                ##提取视标准井GROUP的均值、标准差数据以其他井对应GROUP段的均值、std数据
                refer_info = refer_well[refer_well.index == gp].copy()
                #print('refer_info', refer_info.shape, refer_info)
                info = info_well[info_well['GROUP'] == gp].copy()
                info_col = info.columns
                #print('info', info.shape, info)

                ##计算缩放scale因子
                #print('refer_info[sfn]', refer_info[sfn].shape, refer_info[sfn])
                #print('info[sfn]', info[sfn].shape, info[sfn])

                a = np.array((info[sfn] == 0)).flatten()
                #print('a', a)
                if a.any():
                    m = pd.Series(info[info[sfn] == 0].values.flatten(), index=info_col)
                    #print('m', m)
                    m_l = m[m == 0].index
                    #print('0值所在的列名', m_l)
                    #print('赋值', refer_info[m_l])
                    info[m_l] =  refer_info[m_l].values
                    print('info更改零值后', info[sfn])

                std_fac = refer_info[sfn].values/info[sfn].values
                #print('std_fac 注意计算中维度，广播机制', std_fac.shape,std_fac)
                mean_fac = refer_info[mfn].values - info[mfn].values*std_fac
                #print('mean_fac', mean_fac)

                ##stand_layer = layer*std_fac+mean_fac, 注意按不同log分别做
                layer = well[well['GROUP'] == gp][fea].values
                #print('layer', layer.shape, layer[:5, :])
                stand_layer =  layer*std_fac+mean_fac
                #print('stand_layer[:5, :]', stand_layer.shape, stand_layer[:5, :])

                #print('标准化前数据', data[(data.WELL == wn) & (data.GROUP == gp)].head())
                mask = (data.WELL == wn) & (data.GROUP == gp)
                #print('mask.shape', mask.shape, mask)
                #data[(data.WELL == wn) & (data.GROUP == gp)].loc[:, fea] = stand_layer
                data.loc[mask, fea] = stand_layer
                #print('查看数据是否标准化&赋值OK', data[(data.WELL == wn) & (data.GROUP == gp)].head())
            
        

    return data


def cat_well(well_name, data):
    '''选择井并拼接成训练集、验证集、测试集'''
    well_set = pd.DataFrame(np.full((1,data.shape[1]), np.nan), columns=data.columns)
    #print('well_set', well_set)

    for i in well_name:  #well_name 是list
        print('井名', i)
        well = data[data['WELL'] == i].copy()
        #print('每口井长度',len(well))

        ##拼接井
        well_set= pd.concat([well_set, well], axis=0, ignore_index=True)

    ##去掉认为制造的null行
    well_set = well_set.iloc[1:, :].copy()
    #print('数据集长度', len(well_set))
    well_set = well_set.reset_index(drop=True)
    #print('well_set', well_set)

    return well_set


def medfilt(label,filt_size):
    '''两次中值滤波'''

    b=label.shape[1]
    print('数据的维度',label.shape)
    lab_cop=label.copy()

    for i in range(b):
        #signal.medfilt返回ndarray数组
        lab_cop.iloc[:,i]=signal.medfilt(lab_cop.iloc[:,i],kernel_size=filt_size)

    #第二次进行中值滤波
    for i in range(b):
        #signal.medfilt返回ndarray数组
        lab_cop.iloc[:,i]=signal.medfilt(lab_cop.iloc[:,i],kernel_size=filt_size)

    return lab_cop


def mean_var_norm(label, fea):
    df1_mean=label.mean()
    df1_std=label.std(ddof=0)  #避免std使用index=0的影响
    label_mean_var_norm=(label-df1_mean)/df1_std

    #由于进行gardner比较时，需要用到有实际物理意义的数据，所以这里返回numpy格式的均值和方差
    #直接np.array(df1_mean)时，其shape=(4,) 是一维数组
    np_mean = np.array(df1_mean).reshape((1,len(fea)))   #np_mean.shape=(1,4)
    np_std = np.array(df1_std).reshape((1,len(fea)))

    print('np_mean',np_mean.shape,np_mean)
    print('np_std',np_std.shape,np_std)

    return label_mean_var_norm, np_mean, np_std


def plot_all_wells_fea(dfdata,fea,fig_path,state):
    '''绘制同一类型log的所有井'''

    well_name = dfdata['WELL'].unique()  #well_name is ndarray()
    num_well = len(well_name)
    # print('num_well',num_well)
    # print('well_name',well_name)
    # print('dfdata.columns',dfdata.columns)

    fig = plt.figure(figsize=(9,16))
    plt.title(fea,fontsize=30)
    plt.ylabel('Depth(m)',fontsize=25)
    unit = ['API','g/cm\u00B3','100Pu','us/ft']
    if fea == 'GR':
        plt.xlabel(unit[0],fontsize=25)
    elif fea == 'RHOB':
        plt.xlabel(unit[1],fontsize=25)
    elif fea == 'NPHI':
        plt.xlabel(unit[2],fontsize=25)
    elif fea == 'DTC':
        plt.xlabel(unit[3],fontsize=25)
    
    
    for i in range(num_well):  #num_well
        well_fea = dfdata[dfdata['WELL'] == well_name[i]][fea]
        well_md = dfdata[dfdata['WELL'] == well_name[i]]['DEPTH_MD']
        plt.plot(well_fea,well_md)
        
    plt.ylim(dfdata['DEPTH_MD'].max(),dfdata['DEPTH_MD'].min())
    # print("dfdata['DEPTH_MD'].max()",dfdata['DEPTH_MD'].max())
    # print("dfdata['DEPTH_MD'].min()",dfdata['DEPTH_MD'].min())
    plt.tick_params(labelsize=25)
    
    plt.tight_layout()
    plt.savefig(fig_path+'all_wells_fea/'+str(fea)+'_'+state+'.jpg',dpi=300,bbox_inches='tight')


def handle_outlier(data_ori,GR_min,GR_max,RHOB_min,RHOB_max,NPHI_min,NPHI_max,
                    DTC_min,DTC_max,method):
    '''分别处理GR RHOB NPHI DTC的异常值'''
    data = data_ori.copy()
    #print('GR异常小的值的个数:', (data['GR']<=GR_min).sum())
    #print('GR异常大的值的个数:', (data['GR']>GR_max).sum())  #GR_max
    data['GR'][data['GR']<=GR_min] = np.nan
    data['GR'][data['GR']>GR_max] = np.nan
    #GR型数据可以插值回去
    data['GR'] = data['GR'].interpolate(method=method)

    ##处理RHOB异常
    # print('RHOB异常小的值的个数:', (data['RHOB']<=RHOB_min).sum())
    # print('RHOB异常大的值的个数:', (data['RHOB']>RHOB_max).sum())
    data['RHOB'][data['RHOB']<=RHOB_min] = np.nan
    data['RHOB'][data['RHOB']>RHOB_max] = np.nan


    ##处理NPHI异常值
    #print('NPHI异常小的值的个数:', (data['NPHI']<=NPHI_min).sum())
    #print('NPHI异常大的值的个数:', (data['NPHI']>NPHI_max).sum())
    data['NPHI'][data['NPHI']<=NPHI_min] = np.nan
    data['NPHI'][data['NPHI']>NPHI_max] = np.nan
    

    ##处理DTC异常值
    #print('DTC异常小的值的个数:', (data['DTC']<=DTC_min).sum())
    #print('DTC异常大的值的个数:', (data['DTC']>DTC_max).sum())
    data['DTC'][data['DTC']<=DTC_min] = np.nan
    data['DTC'][data['DTC']>DTC_max] = np.nan

    return data


def reference_log(data, sam_interval):
    '''计算各种log的均值,即参考log'''
    si = sam_interval #si is sample interval 
    well_name = data['WELL'].unique()
    dep_min = data['DEPTH_MD'].min()
    dep_max = data['DEPTH_MD'].max()

    index_num = int((dep_max-dep_min)/si)+1
    col_num = len(well_name)  #col_num也就是数据中井的数目
    print('井数目:{},所有井长度格数:{}'.format(col_num, index_num))
    ##将参考均值井的对应深度也做出来
    refer_md = np.linspace(dep_min, dep_min+si*(index_num-1), index_num)
    print('refer_md的最浅位置：{}，最深位置：{},数目{})'.format(dep_min, 
            dep_min+si*(index_num-1), len(refer_md)))
    refer_md = pd.DataFrame(refer_md, columns=['DEPTH_MD'])

    GR_logs = pd.DataFrame(np.full((index_num,col_num),np.nan),columns=well_name)
    RHOB_logs = pd.DataFrame(np.full((index_num,col_num),np.nan),columns=well_name)
    NPHI_logs = pd.DataFrame(np.full((index_num,col_num),np.nan),columns=well_name)
    DTC_logs = pd.DataFrame(np.full((index_num,col_num),np.nan),columns=well_name)

    for order in range(col_num):

        well = data[data['WELL']==well_name[order]]
        well_len = len(well)
        stap_loc = int((well['DEPTH_MD'].min()-dep_min)/si)
        #将井各log值分别放在对应的位置
        GR_logs.iloc[stap_loc:(stap_loc+well_len),order] = well['GR'].values
        RHOB_logs.iloc[stap_loc:(stap_loc+well_len),order] = well['RHOB'].values
        NPHI_logs.iloc[stap_loc:(stap_loc+well_len),order] = well['NPHI'].values
        DTC_logs.iloc[stap_loc:(stap_loc+well_len),order] = well['DTC'].values
    GR_mean = GR_logs.mean(axis=1)
    RHOB_mean = RHOB_logs.mean(axis=1)
    NPHI_mean = NPHI_logs.mean(axis=1)
    DTC_mean = DTC_logs.mean(axis=1)
    #print('GR_mean的columns', GR_mean.columns) ##GR_mean是Series,无columns
    return GR_mean, RHOB_mean, NPHI_mean, DTC_mean, refer_md

def log_interpolate(data_ori, fea, method='quadratic', limit_direction='forward'):
    '''查找定位哪些log有缺失，并插值回去'''
    data = data_ori.copy()
    logs_null_num = data[fea].isnull().sum()
    print('DEPTH_MD及各log含空值的数目', logs_null_num)

    len_data_columns = len(data.columns)
    len_fea = len(fea)
    dif = len_data_columns-len_fea
    print('数据columns数目与fea之间差距个数', dif)

    ##.get_loc('名称') 根据名称获取所在的位置
    loc_GR = data.columns.get_loc('GR')
    loc_DTC = data.columns.get_loc('DTC')
    print('GR所在列位置', loc_GR)
    print('DTC所在列位置', loc_DTC)
    
    for i in range(loc_GR, loc_DTC+1):
        ##大于0说明有null值
        if logs_null_num[i-dif] > 0:
            ##limit_direction表示插值方向 ： {‘forward’，‘backward’，‘both’}
            data.iloc[:,i] = data.iloc[:,i].interpolate(method = method, 
                                            limit_direction = limit_direction)

    return data
    
    
