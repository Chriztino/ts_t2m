import numpy as np
import matplotlib.pyplot as plt

from .metrics import all_metrics

def set_matplotlib(plot_dpi=80,save_dpi=300,font_size=12):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = font_size
    plt.rcParams['figure.dpi'] = plot_dpi
    plt.rcParams['savefig.dpi'] = save_dpi


def plot_dataset(train,test,size=(6,3.5),xlabel="",ylabel="",fig_name=""):
    x_train = np.linspace(1,len(train),len(train))
    x_test = np.linspace(len(train),len(train)+len(test),len(test)+1)

    plt.figure(figsize=size)
    plt.plot(x_train,train,label='训练数据集')
    plt.plot(x_test,np.append(train[-1],test),label='测试数据集')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.png',bbox_inches='tight')
    plt.show()


def plot_decomposition(series,decomposition,size=(6,7),xlabel='',ylabel='',fig_name=''):
    plt.figure(figsize=size)

    plt.subplot(411)
    plt.plot(series,label='原始数据')
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 趋势项
    plt.subplot(412)
    plt.plot(decomposition.trend,color='r',label='趋势项')
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 季节项
    plt.subplot(413)
    plt.plot(decomposition.seasonal,color='g',label='季节项')
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 残差
    plt.subplot(413)
    plt.plot(decomposition.resid,color='b',label='残差项')

    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.png',bbox='tight')
    plt.show()


def plot_losses(train_loss,val_loss=None,size=(6,3.5),xlabel='',ylabel='',fig_name=''):
    plt.figure(figsize=size)
    plt.plot(train_loss,label='训练损失')
    if val_loss:
        plt.plot(val_loss,label='验证损失')
    
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.png',bbox_inches='tight')
    plt.show()


def plot_results(y_obs,y_sim,size=(6,3.5),xlabel='',ylabel='',fig_name=''):
    plt.figure(figsize=size)
    plt.plot(y_obs.squeeze(),label='观测值')
    plt.plot(y_sim.squeeze(),label='预测值')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.png',bbox_inches='tight')
    plt.show()


def plot_parity(y_obs,y_sim,size=(6,3.5),xlabel='',ylabel='',fig_name=''):
    x = y_obs
    y = y_sim

    bounds = (
        min(x.min(),y.min()) - int(0.1*x.min()),
        max(x.max(),y.max()) + int(0.1*x.max())
    )

    plt.figure(figsize=size)
    ax = plt.gca()
    ax.plot(x,y,'.',label='观测-预测')
    ax.plot([0,1],[0,1],lw=2,alpha=1.0,transform=ax.transAxes,label='$y=x$')

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect('equal',adjustable='box')

    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.png',bbox_inches='tight')
    plt.show()


def plot_metrics_distribution(y_obs,y_sim,size=(10,3),xlabel='',ylabel='',fig_name=''):
    all_rmse = [ ]
    all_mae = [ ]
    all_sde = [ ]

    N = y_obs.shape[1]
    for idx_node in range(N):
        metrics_value = all_metrics(
            y_obs[:,idx_node],
            y_sim[:,idx_node],
            return_metrics=True
        )
        all_rmse.append(metrics_value['rmse'])
        all_mae.append(metrics_value['mae'])
        all_sde.append(metrics_value['sde'])

    plt.figure(figsize=size)
    plt.bar(
        x = np.arange(N),
        height=y_obs.mean(axis=0).squeeze(),
        color='lightgeay',
        label='Mean'
    )
    plt.plot(all_rmse,'v--',label='RMSE')
    plt.plot(all_mae,'s--',label='MAE')
    plt.plot(all_sde,'d--',label='SDE')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.png',bbox_inches='tight')
    plt.show()
