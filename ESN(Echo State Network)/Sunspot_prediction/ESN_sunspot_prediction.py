# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:34:46 2023

@author: yoshi
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# データの読み込み
def read_sunspot_data(file_name):
    ''' 
    :入力：データファイル名, file_name
    :出力：黒点数データ, data
    '''
    data = np.empty(0)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            data = np.hstack((data, float(tmp[3])))  # 3rd column
    return data


# 黒点数データ
sunspots = read_sunspot_data(file_name='SN_ms_tot_V2.0.txt')

# データのスケーリング
data_scale = 1.0
data = sunspots*data_scale

# 入力層
N_u = 1
N_x = 400
input_scale = 1

'''
param N_u: 入力次元
param N_x: リザバーのノード数
param input_scale: 入力スケーリング
'''

# 入力結合重み行列Winの生成
np.random.seed(seed=0)
Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

# 入力結合重み行列Winによる重みづけの関数の定義
def Input(u_in_train_input):
    u_in = np.dot(Win, u_in_train_input)
    return u_in
'''
param u_in_train_input: 入力層への入力ベクトル
param u_in: 更新前の状態ベクトル（リザバーへの入力ベクトル）
'''



# リザバー層
N_x = 400
density = 0.1
rho = 0.9
activation_func = np.tanh
leaking_rate = 1
seed = 0

'''
param N_x: リザバーのノード数
param density: ネットワークの結合密度
param rho: リカレント結合重み行列のスペクトル半径
param activation_func: ノードの活性化関数
param leaking_rate: leaky integratorモデルのリーク率
param seed: 乱数の種
'''

# リカレント結合重み行列Wの生成
# Erdos-Renyiランダムグラフ
m = int(N_x*(N_x-1)*density/2)  # 総結合数
G = nx.gnm_random_graph(N_x, m, seed=seed)

# 行列への変換(結合構造のみ）
connection = nx.to_numpy_matrix(G)
W = np.array(connection)

# 非ゼロ要素を一様分布に従う乱数として生成
rec_scale = 1.0
np.random.seed(seed=seed)
W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

# スペクトル半径の計算
eigv_list = np.linalg.eig(W)[0]
sp_radius = np.max(np.abs(eigv_list))

# 指定のスペクトル半径rhoに合わせてスケーリング
W *= rho / sp_radius

# リザバー状態ベクトルの更新
alpha = leaking_rate
x = np.zeros(N_x)  # リザバー状態ベクトルの初期化


# NRMSEのリスト
NRMSE_train = []
NRMSE_test = []

# 複数ステップ先の予測
step_list = [1,5,10,15,20,25,30,35,40,45,50,100]
for step in step_list:
    st = step
    
    # 訓練・検証データ長
    T_train = 2500
    T_test = data.size-T_train-step
    
    # 訓練・検証用情報
    train_U = data[:T_train].reshape(-1, 1)
    train_D = data[step:T_train+step].reshape(-1, 1)
    
    test_U = data[T_train:T_train+T_test].reshape(-1, 1)
    test_D = data[T_train+step:T_train+T_test+step].reshape(-1, 1)
    
    ######## 学習データに対して
    # リザバー状態行列
    stateCollectMat = np.empty((0, N_x))
    x = np.zeros(N_x)  # リザバー状態ベクトルの初期化
    for i in range(len(train_U)):
        if i%100 == 0:
            print(i)
            print((i/len(train_U))*100, '%　完了')
        if i == len(train_U) - 1:
            print('リザバー状態行列の計算完了')
        u_in = Input(train_U[i])
        x = (1.0 - alpha) * x + alpha * activation_func(np.dot(W, x) + u_in)
        stateCollectMat = np.vstack((stateCollectMat, x))

    # 教師出力データ行列
    teachCollectMat = train_D

    # 学習（疑似逆行列）
    Wout = np.dot(teachCollectMat.T, np.linalg.pinv(stateCollectMat.T))
    
    # 予測出力
    Y_pred_train = np.dot(Wout, stateCollectMat.T)
    
    
    ######## テストデータに対して
    # リザバー状態行列
    stateCollectMat = np.empty((0, N_x))
    for i in range(len(test_U)):
        if i%100 == 0:
            print(i)
            print((i/len(test_U))*100, '%　完了')
        if i == len(test_U) - 1:
            print('リザバー状態行列の計算完了')
        u_in = Input(test_U[i])
        x = (1.0 - alpha) * x + alpha * activation_func(np.dot(W, x) + u_in)
        stateCollectMat = np.vstack((stateCollectMat, x))
        
    # ラベル出力
    Y_pred_test = np.dot(Wout, stateCollectMat.T)
    
    # 訓練誤差評価（NRMSE）
    RMSE = np.sqrt(((train_D/data_scale - Y_pred_train/data_scale) ** 2)
                         .mean())
    NRMSE = RMSE/np.sqrt(np.var(train_D/data_scale))
    NRMSE_train.append(NRMSE)


    # 検証誤差評価（NRMSE）
    RMSE = np.sqrt(((test_D/data_scale - Y_pred_test/data_scale) ** 2)
                        .mean())
    NRMSE = RMSE/np.sqrt(np.var(test_D/data_scale))
    NRMSE_test.append(NRMSE)


    # 全範囲グラフ描画
    x = np.arange(0,len(train_U),1)
    y = train_D.reshape(len(train_U),)
    plt.plot(x,y)

    x = np.arange(0,len(train_U),1)
    y = Y_pred_train.reshape(len(train_U),)
    plt.plot(x,y)
    
    x = np.arange(len(train_U),len(data)-st,1)
    y = test_D.reshape(len(data)-len(train_U)-st,)
    plt.plot(x,y)

    x = np.arange(len(train_U),len(data)-st,1)
    y = Y_pred_test.reshape(len(data)-len(train_U)-st,)
    plt.ylim(0,300) #y軸範囲指定
    plt.plot(x,y)
    
    plt.show()
    
    
    # x軸指定グラフ描画
    x = np.arange(0,len(train_U),1)
    y = train_D.reshape(len(train_U),)
    plt.plot(x,y)

    x = np.arange(0,len(train_U),1)
    y = Y_pred_train.reshape(len(train_U),)
    plt.plot(x,y)
    
    x = np.arange(len(train_U),len(data)-st,1)
    y = test_D.reshape(len(data)-len(train_U)-st,)
    plt.plot(x,y)

    x = np.arange(len(train_U),len(data)-st,1)
    y = Y_pred_test.reshape(len(data)-len(train_U)-st,)
    plt.ylim(0,300) #y軸範囲指定
    plt.xlim(2200,2800) #x軸範囲指定
    plt.plot(x,y)
    
    plt.show()
 


for i, step in enumerate(step_list):
    print(step, 'ステップ先予測')
    print('訓練誤差：NRMSE =', NRMSE_train[i])
    print('検証誤差：NRMSE =', NRMSE_test[i])


    