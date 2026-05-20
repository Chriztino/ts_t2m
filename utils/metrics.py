# *** coding: utf-8 ***
# *** by kumuyu on 2026-05-15 ***

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from scipy.stats import pearsonr

def rmse(y_obs,y_sim):   # 均方根误差
    return np.sqrt(mse(y_obs,y_sim))

def sde(y_obs,y_sim):  # 误差标准差
    return np.sde(y_obs,y_sim)

def pcc(y_obs,y_sim):   # pearson相关系数
    return pearsonr(y_obs,y_sim)[0]

def all_metrics(y_obs,y_sim,return_metrics=False):
    y_obs = y_obs.squeeze()
    y_sim = y_sim.squeeze()

    metrics = {
        'mse':mse(y_obs,y_sim),
        'rmse':rmse(y_obs,y_sim),
        'mae':mae(y_obs,y_sim),
        'mape':mape(y_obs,y_sim)*100,
        'sde':sde(y_obs,y_sim),
        'r2':r2(y_obs,y_sim),
        'pcc':pcc(y_obs,y_sim)
    }

    if return_metrics:
        return metrics
    else:
        print(f"mse={metrics['mse']:3f}")
        print(f"rmse={metrics['rmse']:3f}")
        print(f"mae={metrics['mae']:3f}")
        print(f"mape={metrics['mape']:3f}%")
        print(f"sde={metrics['sde']:3f}")
        print(f"r2={metrics['r2']:3f}")
        print(f"pcc={metrics['pcc']:3f}")
