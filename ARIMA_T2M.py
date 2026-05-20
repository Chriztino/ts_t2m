import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from tqdm import tqdm



import utils.dataset as udataset
import utils.metrics as umetrics
import utils.plot as uplot

import warnings
warnings.filterwarnings("ignore")

name_model = 'GRID_SMARIMA'
name_var = 'T2M'
name_unit = '℃'
uplot.set_matplotlib(plot_dpi=90,save_dpi=300,font_size=12)

data = pd.read_csv('D:/Apy0/ts_t2m/data/ts_T2M_2000_2024_grid_R0_C0.csv')

df['date'] = pd.to_datetime(df['date'])
df = df.resample('ME').mean()     # 日数据转化成月数据
udataset.series_to_supervised(df.values)
print(f'{df.shape=}')

num_train = 240
train = df.iloc[:num_train]
test = df.iloc[num_train:]
print(f'{train.shape=},{test.shape=}')

uplot.plot_dataset(
    train.values,
    test.values,
    xlabel="Month",
    ylabel=f'{name_var}/{name_unit}',
    fig_name=f'原始序列_{name_model}'
)

decomposition = seasonal_decompose(
    train,
    model='addictive'
)

uplot.plot_decomposition(
    series=train,
    decomposition=decomposition,
    xlabel="Year",
    ylabe=f'{name_var}/{name_unit}',
    fig_name=f'Sonsonal_{name_model}'
)

train_diff = train.copy(deep=True)

max_d = 3

for d in range(0,max_d):
    adftest = adfuller(train_diff)
    pvalue = adftest[1]
    if pvalue < 0.05:
        print(f'差分次数{d=}:拒绝原假设，序列没有单位根，序列平稳')
        order_diff = d
        break
    else:
        print(f'差分次数{d=}:接受原假设，序列有单位根，序列不平稳，需要差分')
        train_diff = train_diff.deff(peried=1)
        train_diff.dropna(inplace=True)

print(f'平稳差分次数{order_diff=}')

# 去除季节性
train_diff = train_diff(12)
train_diff.dropna(inplace=True)

ljungboxtest = acorr_ljungbox(train_diff,lags=[6,12,18],return_df=True)
pvalues = ljungboxtest['lb_pvalue'].values
for pvalue in pvalues:
    if pvalue < 0.05:
        print('拒绝原假设H0,序列为非白噪声')
    else:
        print('接受原假设H0,序列为白噪声,终止分析')
        break

plt.plot(train_diff,label='差分序列')
plt.legend(loc='upper left')
plt.xlabel('年份')
plt.ylabel(f'{name_var}/{name_unit}')
plt.savefig(f'./fig/差分序列_{name_model}.png',bbox_inches='tight')
plt.show()


max_value = 3

p = range(max_value)
d = order_diff
q = range(max_value)

P = range(max_value)
D = 1
Q = range(max_value)
S = 12

# 参数说明：
# SARIMA(p,d,q)(P,D,Q,S)
# p,d,q = 非季节性参数  P,D,Q = 季节性参数  s = 季节周期

param_grid = itertools.product(p,q,P,Q)
param_grid = list(param_grid)
print(f'待搜索参数数量：{len(param_grid)=}')

results = []

best_bic = float('inf')

for param in tqdm(param_grid):
    p,q,P,Q = param
    try:
        model = SARIMAX(
            train,
            order=(p,d,q),
            seasonal_order=(P,D,Q,S)
        ).fit(disp=-1)
    except Exception:
        continue

    bic = model.bic

    if bic < best_bic:
        best_bic = bic
        best_model = model
        best_param = param

    results.append([param,bic])

results = pd.DataFrame(
    results,
    columns = ['parameters','bic']
)

results = results.sort_values(
    by='bic',
    ascending=True
).reset_index(drop=True)

p,q,P,Q = best_param

print(f'最优参数：{p=},{d=},{q=},{P=},{D=},{S=},最优BIC：{best_bic:2f}')

pred = best_model.forecast(test.shape[0])

umetrics.all_metrics(test.values,pred.values)

plt.plot(train,label='训练集')
plt.plot(test,label='测试集')
plt.plot(pred,label='预测')
plt.legend(loc='upper left')
plt.xlabel('年份')
plt.ylabel(f'{name_var}/{name_unit}')
plt.savefig(f'./fig/预测结果_{name_model}.png',bbox_inches='tight')
plt.show()

uplot.plot_parity(
    y_true = test.values,
    y_pred = pred.values,
    xlabel=f'Observation/{name_unit}',
    ylabel=f'Simulation/{name_unit}',
    fig_name=f'{name_model}_Parity'
)