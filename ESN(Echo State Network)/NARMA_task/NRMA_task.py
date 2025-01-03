import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# NARMAモデルのパラメータ
m = 10
n = m
a1 = 0.3
a2 = 0.05
a3 = 1.5
a4 = 0.1
# データ長
T = 200
d = [0]*m

np.random.seed(seed=0)

# 時系列入力データ生成
u = np.random.uniform(0, 0.5, T)
# 時系列入力データ グラフ描画
x = np.arange(0,200,1)
y = u.reshape(T,)
plt.plot(x,y)
plt.show()

u = u.reshape(-1, 1)

# 時系列目標データ作成
for i in range(m,T):
    d_n = a1*d[n-1] + a2*d[n-1]*(np.sum(d[n-m+1:n])) + a3*u[n-m+1]*u[n] + a4
    d.append(d_n)
    n += 1

d = np.array(d)
# 時系列目標データ グラフ描画
x = np.arange(0,200,1)
y = d.reshape(T,)
plt.plot(x,y)
plt.show()



# 入力層
N_u = 1
N_x = 500
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
N_x = 500
density = 0.15
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


######## データに対して
# リザバー状態行列
stateCollectMat = np.empty((0, N_x))
x = np.zeros(N_x)  # リザバー状態ベクトルの初期化
for i in range(len(u)):
    if i%100 == 0:
        print((i/len(u))*100, '%　完了')
    if i == len(u) - 1:
        print('リザバー状態行列の計算完了')
    u_in = Input(u[i])
    x = (1.0 - alpha) * x + alpha * activation_func(np.dot(W, x) + u_in)
    stateCollectMat = np.vstack((stateCollectMat, x))

# 教師出力データ行列
teachCollectMat = d

# 学習（疑似逆行列）
Wout = np.dot(teachCollectMat.T, np.linalg.pinv(stateCollectMat.T))

# 予測出力
Y_pred_train = np.dot(Wout, stateCollectMat.T)

# データ全範囲グラフ描画
x = np.arange(0,len(u),1)
y = d.reshape(len(u),)
plt.plot(x,y)

x = np.arange(0,len(u),1)
y = Y_pred_train.reshape(len(u),)
plt.plot(x,y)
plt.show()

# データ後半グラフ描画
x = np.arange(0,len(u),1)
y = d.reshape(len(u),)
plt.plot(x,y)

x = np.arange(0,len(u),1)
y = Y_pred_train.reshape(len(u),)
plt.xlim(100,200)
plt.plot(x,y)
plt.show()

# 評価（テスト誤差RMSE, NRMSE）
RMSE = np.sqrt(((d - Y_pred_train) ** 2).mean())
NRMSE = RMSE/np.sqrt(np.var(d))
print('RMSE =', RMSE)
print('NRMSE =', NRMSE)
