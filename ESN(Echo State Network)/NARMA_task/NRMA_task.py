import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# NARMAモデルのパラメータ
n = 10
a1, a2, a3, a4 = 0.3, 0.05, 1.5, 0.1
# データ長
T = 1000
d = [0]*n

np.random.seed(seed=0)

# 時系列入力データ生成
u = np.random.uniform(0, 0.5, T)
u = u.reshape(-1, 1)

# 時系列目標データ作成
for i in range(n, T):
    d_n = a1*d[-1] + a2*d[-1]*np.sum(d[-n:]) + a3*u[i-n]*u[i-1] + a4
    d.append(d_n)
d = np.array(d)

# データ分割
u_train, u_test = u[n*2:T//2], u[T//2-n*2:]
d_train, d_test = d[n*2:T//2], d[T//2-n*2:]

# 入力層
N_u = 1
N_x = 3000
input_scale = 1

# 入力結合重み行列Winの生成
np.random.seed(seed=0)
Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

def Input(u_in_train_input):
    u_in = np.dot(Win, u_in_train_input)
    return u_in

# リザバー層
density = 0.20
rho = 0.9
activation_func = np.tanh
leaking_rate = 0.9
seed = 0

# リカレント結合重み行列Wの生成
# Erdos-Renyiランダムグラフ
m = int(N_x*(N_x-1)*density/2) # 総結合数
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

# リザバー状態行列の計算(リザバー状態ベクトルの更新)
def compute_reservoir_states(u_data, W, alpha, activation_func):
    stateCollectMat = np.zeros((len(u_data), N_x))
    x = np.zeros(N_x) # リザバー状態ベクトルの初期化
    
    # tqdmを使って進捗バーを表示
    for i in tqdm(range(len(u_data)), desc="Computing Reservoir States", ncols=100): 
        u_in = Input(u_data[i])
        x = (1.0 - alpha) * x + alpha * activation_func(np.dot(W, x) + u_in)
        stateCollectMat[i] = x
    
    return stateCollectMat

# 学習データのリザバー状態行列
stateCollectMat_train = compute_reservoir_states(u_train, W, leaking_rate, activation_func)

# テストデータのリザバー状態行列
stateCollectMat_test = compute_reservoir_states(u_test, W, leaking_rate, activation_func)

# リードアウト重みの計算
Wout_random = np.random.uniform(-0.03, 0.03, N_x) # 未学習のリードアウト重み
Wout = np.dot(d_train.T, np.linalg.pinv(stateCollectMat_train.T)) # 学習済みのリードアウト重み

# 学習データでの予測（未学習重み使用）
Y_pred_train_random = np.dot(Wout_random, stateCollectMat_train.T)

# 学習データでの予測(学習済み重み使用)
Y_pred_train = np.dot(Wout, stateCollectMat_train.T)

# テストデータでの予測
Y_pred_test = np.dot(Wout, stateCollectMat_test.T)

# 評価（RMSE, NRMSE）
RMSE_train = np.sqrt(((d_train[-100:] - Y_pred_train[-100:]) ** 2).mean())
NRMSE_train = RMSE_train / np.sqrt(np.var(d_train[-100:]))
RMSE_test = np.sqrt(((d_test[n*2:100+n*2] - Y_pred_test[n*2:100+n*2]) ** 2).mean())
NRMSE_test = RMSE_test / np.sqrt(np.var(d_test[n*2:100+n*2]))

print("Train RMSE =", RMSE_train)
print("Train NRMSE =", NRMSE_train)
print("Test RMSE =", RMSE_test)
print("Test NRMSE =", NRMSE_test)


# グラフ表示
u = np.concatenate([u[-100:], u[n*2:100+n*2]])
d_target = np.concatenate([d_train[-100:], d_test[n*2:100+n*2]])
Y_pred = np.concatenate([Y_pred_train_random[-100:], Y_pred_test[n*2:100+n*2]])
plt.rcParams['font.size'] = 13
fig = plt.figure(figsize=(7, 6), dpi=500)
plt.subplots_adjust(hspace=0.3)
x_range = np.arange(-100, 100)

ax1 = fig.add_subplot(2, 1, 1)
ax1.text(0.14, 1.05, 'Before Training', transform=ax1.transAxes)
ax1.text(0.64, 1.05, 'After Training', transform=ax1.transAxes)
plt.plot(x_range, u, color='black')
plt.ylabel('Input', fontsize=16)
plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

ax2 = fig.add_subplot(2, 1, 2)
plt.plot(x_range, d_target, label='Target')
plt.plot(x_range, Y_pred, linestyle='--', label='Prediction')
plt.xlabel('Time Step', fontsize=16)
plt.ylabel('Output', fontsize=16)
plt.legend(bbox_to_anchor=(1, 0), loc='lower right')
plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

plt.show()
