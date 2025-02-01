import numpy as np
import networkx as nx
from tqdm import tqdm

class ESN:
    def __init__(self, N_u, N_x, input_scale, rec_scale, rho, leaking_rate, activation_func, seed):
        self.N_u = N_u  # 入力次元
        self.N_x = N_x  # リザバー次元
        self.input_scale = input_scale
        self.rec_scale = rec_scale
        self.rho = rho  # スペクトル半径
        self.leaking_rate = leaking_rate
        self.activation_func = activation_func
        self.seed = seed
        
        np.random.seed(self.seed)

        # 入力結合重み行列 Win の生成
        self.Win = np.random.uniform(-self.input_scale, self.input_scale, (self.N_x, self.N_u))

        # リカレント結合重み行列 W の生成
        self.W = self._generate_recurrent_weights()

    def _generate_recurrent_weights(self):
        # Erdos-Renyiランダムグラフを使用してW行列を作成
        m = int(self.N_x * (self.N_x - 1) * 0.20 / 2)  # 総結合数
        G = nx.gnm_random_graph(self.N_x, m, seed=self.seed)
        connection = nx.to_numpy_matrix(G)
        W = np.array(connection)

        # 非ゼロ要素を一様分布に従って乱数で生成
        W *= np.random.uniform(-self.rec_scale, self.rec_scale, (self.N_x, self.N_x))

        # スペクトル半径のスケーリング
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))
        W *= self.rho / sp_radius
        
        return W

    def Input(self, u_in_train_input):
        return np.dot(self.Win, u_in_train_input)

    def compute_reservoir_states(self, u_data):
        stateCollectMat = np.zeros((len(u_data), self.N_x))
        x = np.zeros(self.N_x)  # リザバー状態ベクトルの初期化

        # tqdmを使って進捗バーを表示
        for i in tqdm(range(len(u_data)), desc="Computing Reservoir States", ncols=100): 
            u_in = self.Input(u_data[i])
            x = (1.0 - self.leaking_rate) * x + self.leaking_rate * self.activation_func(np.dot(self.W, x) + u_in)
            stateCollectMat[i] = x
        
        return stateCollectMat

    def train(self, stateCollectMat_train, d_train):
        # リードアウト重みの学習
        Wout = np.dot(d_train.T, np.linalg.pinv(stateCollectMat_train.T))  # 学習済みのリードアウト重み
        return Wout

    def predict(self, Wout, stateCollectMat):
        return np.dot(Wout, stateCollectMat.T)