import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from esn import ESN

if __name__ == '__main__':

    # 時系列入力データ生成
    T = 500  # 長さ
    period = 50  # 周期
    time = np.arange(T)  # 時間
    u = np.sin(2*np.pi*time/period)  # 正弦波

    # 入力層
    N_u = 1
    N_x = 400
    input_scale = 1
    rec_scale = 1.0
    rho = 1.3
    leaking_rate = 1
    activation_func = np.tanh
    seed = 0
    
    # ESNのインスタンスを作成
    esn = ESN(N_u, N_x, input_scale, rec_scale, rho, leaking_rate, activation_func, seed)

    # リザバー状態の時間発展
    U = u[:T].reshape(-1, 1)
    x_all = esn.compute_reservoir_states(U)

    # グラフ表示
    fig = plt.figure(figsize=(12, 8), dpi=1000)
    gs = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.4)

    # (n, x_1)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(time, x_all[:, 0], color='k', linewidth=2)
    ax1.set_xlabel('Time Step', fontsize=18)
    ax1.set_ylabel('x_1 Output', fontsize=18)
    ax1.grid(True)

    # (n, x_2)
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(time, x_all[:, 1], color='k', linewidth=2)
    ax2.set_xlabel('Time Step', fontsize=18)
    ax2.set_ylabel('x_2 Output', fontsize=18)
    ax2.grid(True)

    # (n, x_3)
    ax3 = plt.subplot(gs[0, 2])
    ax3.plot(time, x_all[:, 2], color='k', linewidth=2)
    ax3.set_xlabel('Time Step', fontsize=18)
    ax3.set_ylabel('x_3 Output', fontsize=18)
    ax3.grid(True)

    # (u, x_1)
    ax4 = plt.subplot(gs[1, 0])
    ax4.plot(u, x_all[:, 0], color='k', linewidth=2)
    ax4.set_xlabel('Input', fontsize=18)
    ax4.set_ylabel('x_1 Output', fontsize=18)
    ax4.grid(True)

    # (u, x_2)
    ax5 = plt.subplot(gs[1, 1])
    ax5.plot(u, x_all[:, 1], color='k', linewidth=2)
    ax5.set_xlabel('Input', fontsize=18)
    ax5.set_ylabel('x_2 Ouput', fontsize=18)
    ax5.grid(True)

    # (u, x_3)
    ax6 = plt.subplot(gs[1, 2])
    ax6.plot(u, x_all[:, 2], color='k', linewidth=2)
    ax6.set_xlabel('Input', fontsize=18)
    ax6.set_ylabel('x_3 Ouput', fontsize=18)
    ax6.grid(True)
    
    # 目盛りの文字サイズを変更
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(axis='both', labelsize=16)

    plt.show()