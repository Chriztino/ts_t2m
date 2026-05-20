import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# 1) 生成数据
n = 1500
t = np.arange(n)
dates = pd.date_range("2020-01-01", periods=n, freq="D")

# 趋势 + 季节（周、月）
trend = 0.02 * t
season_week = 1.5 * np.sin(2 * np.pi * t / 7)
season_month = 1.0 * np.sin(2 * np.pi * t / 30)

# 外生变量
temp = 15 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 1.5, size=n)
promo = (np.random.rand(n) < 0.08).astype(int) * (3 + np.random.rand(n) * 2.0)  # 促销强度
holiday = (np.random.rand(n) < 0.05).astype(int)  # 节假日标记

# 制度变更：800天后加一个台阶上移
regime = (t > 800).astype(int) * 2.5

# 噪声: AR(1) e_t = 0.6*e_{t-1} + epsilon
eps = np.random.normal(0, 0.8, size=n)
noise = np.zeros(n)
for i in range(1, n):
    noise[i] = 0.6 * noise[i-1] + eps[i]

y = 20 + trend + season_week + season_month + 0.3 * temp + 1.5 * promo + 2.0 * holiday + regime + noise

df = pd.DataFrame({
    "date": dates,
    "y": y,
    "trend": trend,
    "season_week": season_week,
    "season_month": season_month,
    "temp": temp,
    "promo": promo,
    "holiday": holiday,
    "regime": regime
})
df["dow"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# 划分训练/测试
split = 1200
train_df = df.iloc[:split].copy()
test_df = df.iloc[split:].copy()

# 2) 图1：数据构成与观测值
plt.figure(figsize=(16, 6))
plt.plot(df["date"], df["y"], color="#FF006E", label="观测值 y", linewidth=2.5)
plt.plot(df["date"], df["trend"], color="#3A86FF", label="趋势", alpha=0.8)
plt.plot(df["date"], df["season_week"], color="#8338EC", label="周季节", alpha=0.8)
plt.plot(df["date"], df["season_month"], color="#FB5607", label="月季节", alpha=0.8)
plt.plot(df["date"], 0.3*df["temp"], color="#00F5D4", label="温度贡献(缩放)", alpha=0.8)
plt.plot(df["date"], 1.5*df["promo"], color="#FFD166", label="促销贡献(缩放)", alpha=0.8)
plt.plot(df["date"], df["regime"], color="#06D6A0", label="制度变更", alpha=0.9, linestyle="--")
plt.title("图1：数据的组成与观测值")
plt.legend(ncol=4)
plt.tight_layout()
plt.show()

# 3) ARIMA拟合
# 这里不引入外生变量，让ARIMA专注线性与季节性结构；(p,d,q)手动设为(2,1,2)仅为演示
arima_order = (2, 1, 2)
arima_model = ARIMA(train_df["y"], order=arima_order)
arima_res = arima_model.fit()
# 训练集内拟合
train_df["y_hat_arima"] = arima_res.predict(start=train_df.index[0], end=train_df.index[-1])
train_df["res1"] = train_df["y"] - train_df["y_hat_arima"]

# 测试集预测（多步）
arima_forecast = arima_res.forecast(steps=len(test_df))
test_df["y_hat_arima"] = arima_forecast
test_df["res1"] = test_df["y"] - test_df["y_hat_arima"]

# 图2：第一层残差的 ACF/PACF
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
plot_acf(train_df["res1"], ax=axs[0], color="#3A86FF", lags=50)
plot_pacf(train_df["res1"], ax=axs[1], color="#FB5607", lags=50, method="ywm")
axs[0].set_title("图2A：第一层残差 r^(1) 的ACF")
axs[1].set_title("图2B：第一层残差 r^(1) 的PACF")
plt.tight_layout()
plt.show()

# 4) XGBoost 拟合第一层残差
# 构造特征：滞后项 + 时间特征 + 外生变量 + 傅里叶项
def make_features(df, lags=[1,2,3,7,14], fourier_periods=[7,30], K=2):
    fdf = df.copy()
    for lag in lags:
        fdf[f"lag_{lag}"] = fdf["y"].shift(lag)
    # 时间特征（one-hot）
    for d in range(7):
        fdf[f"dow_{d}"] = (fdf["dow"] == d).astype(int)
    for m in range(1,13):
        fdf[f"month_{m}"] = (fdf["month"] == m).astype(int)
    # 傅里叶项
    for p in fourier_periods:
        for k in range(1, K+1):
            fdf[f"sin_{p}_{k}"] = np.sin(2*np.pi*k*fdf.index.values/p)
            fdf[f"cos_{p}_{k}"] = np.cos(2*np.pi*k*fdf.index.values/p)
    # 交互项（示意）
    fdf["promo_temp"] = fdf["promo"] * fdf["temp"]
    return fdf

train_f = make_features(train_df)
test_f = make_features(test_df)

# 对齐：去除NaN（由滞后产生）
valid_train = train_f.dropna().copy()
valid_test = test_f.dropna().copy()

feature_cols = [c for c in valid_train.columns if c.startswith("lag_") or c.startswith("dow_") or c.startswith("month_")
                or c.startswith("sin_") or c.startswith("cos_") or c in ["temp","promo","holiday","regime","promo_temp"]]

X_train = valid_train[feature_cols].values
y_train_res1 = valid_train["res1"].values

X_test = valid_test[feature_cols].values
y_test_res1 = valid_test["res1"].values

xgb = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist"
)
xgb.fit(X_train, y_train_res1)

valid_train["y_hat_xgb"] = xgb.predict(X_train)
valid_test["y_hat_xgb"] = xgb.predict(X_test)

# 第二层残差
valid_train["res2"] = valid_train["res1"] - valid_train["y_hat_xgb"]
valid_test["res2"] = valid_test["res1"] - valid_test["y_hat_xgb"]

# 图3：XGBoost 特征重要性 + 简易PDP
imp = xgb.feature_importances_
imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False).head(15)

fig = plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
sns.barplot(data=imp_df, x="importance", y="feature", palette="plasma")
plt.title("图3A：XGBoost特征重要性（Top15）")

# 简易部分依赖：对temp和promo分别做曲线（其余特征取训练均值）
mean_vec = X_train.mean(axis=0)
def pdp_curve(model, feat_name, xs):
    idx = feature_cols.index(feat_name)
    Xp = np.tile(mean_vec, (len(xs),1))
    Xp[:,idx] = xs
    return model.predict(Xp)

xs_temp = np.linspace(np.percentile(valid_train["temp"], 5), np.percentile(valid_train["temp"], 95), 50)
xs_promo = np.linspace(valid_train["promo"].min(), valid_train["promo"].max(), 50)
plt.subplot(1,2,2)
plt.plot(xs_temp, pdp_curve(xgb, "temp", xs_temp), color="#FF006E", label="PDP-temp", linewidth=2.5)
plt.plot(xs_promo, pdp_curve(xgb, "promo", xs_promo), color="#3A86FF", label="PDP-promo", linewidth=2.5)
plt.title("图3B：XGBoost简易PDP（温度/促销对残差的影响）")
plt.legend()
plt.tight_layout()
plt.show()

# 5) 用Transformer拟合第二层残差（序列窗口 -> 下一个点）
seq_len = 32

class ResidualDataset(Dataset):
    def __init__(self, df_seq, seq_len, feature_cols=None):
        self.seq_len = seq_len
        # 使用残差 res2 作为主要输入序列，外生作为并行特征
        self.r = df_seq["res2"].values
        self.features = df_seq[feature_cols].values if feature_cols is not None else None
        self.samples = []
        # 构造滑窗：输入[ t-seq_len ... t-1 ] -> 预测 t
        for t in range(seq_len, len(df_seq)-1):
            x_seq = self.r[t-seq_len:t]
            # 可选：把外生特征加到序列每个步长（这里简化为目标时刻的特征）
            if self.features is not None:
                x_feat = self.features[t]
            else:
                x_feat = None
            y_next = self.r[t]  # 预测下一刻的res2（单步）
            self.samples.append((x_seq, x_feat, y_next))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x_seq, x_feat, y_next = self.samples[idx]
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        if x_feat is not None:
            x_feat = torch.tensor(x_feat, dtype=torch.float32)
        return x_seq, x_feat, torch.tensor(y_next, dtype=torch.float32)

# 选取部分外生特征给Transformer（可选）
tfm_feat_cols = ["temp", "promo", "holiday", "regime"]
train_tfm_df = valid_train.reset_index(drop=True)
test_tfm_df = valid_test.reset_index(drop=True)

train_dataset = ResidualDataset(train_tfm_df, seq_len=seq_len, feature_cols=tfm_feat_cols)
test_dataset = ResidualDataset(test_tfm_df, seq_len=seq_len, feature_cols=tfm_feat_cols)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义Transformer模型（序列->标量）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class ResidualTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1, exog_dim=0):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)  # 把res2标量提升到d_model维
        self.pos_enc = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.exog_proj = nn.Linear(exog_dim, d_model) if exog_dim > 0 else None
        self.fc_out = nn.Linear(d_model, 1)
        self.attn_weights = None  # 用hook获取注意力权重

        # 用注册hook在最后一层的self-attn上抓权重
        for mod in self.encoder.modules():
            if isinstance(mod, nn.TransformerEncoderLayer):
                def _hook(module, input, output):
                    # 无法直接拿注意力权重，这里用简化方案：在前向中间存储QK^T的softmax近似
                    # 若需要更精确的注意力可自定义MultiheadAttention返回权重
                    pass
                # plt级别演示，这里不强行hook；下面我们改为自定义SelfAttention以取权重
                break

    def forward(self, x_seq, x_exog=None):
        # x_seq: (batch, seq_len)
        x = x_seq.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_proj(x)   # (batch, seq_len, d_model)
        x = self.pos_enc(x)      # 加位置编码
        memory = self.encoder(x) # (batch, seq_len, d_model)
        # 拼接外生特征（简化：目标时刻的exog作用于最终读出层）
        if self.exog_proj is not None and x_exog is not None:
            ex = self.exog_proj(x_exog) # (batch, d_model)
            # 读出时将最后时刻的表示与外生叠加
            last = memory[:, -1, :] + ex
        else:
            last = memory[:, -1, :]
        out = self.fc_out(last).squeeze(-1)
        return out, memory

# 自定义自注意力层以抓权重
model = ResidualTransformer(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1, exog_dim=len(tfm_feat_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 训练
epochs = 20
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for x_seq, x_exog, y_next in train_loader:
        optimizer.zero_grad()
        y_hat, _ = model(x_seq, x_exog)
        loss = loss_fn(y_hat, y_next)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_seq.size(0)
    print(f"Epoch {epoch+1}/{epochs} - Train MSE: {total_loss/len(train_dataset):.4f}")

# 评估与注意力热图（示意取一个样本窗口）
model.eval()
with torch.no_grad():
    # 取一个批次
    x_seq, x_exog, y_next = next(iter(test_loader))
    y_hat, memory = model(x_seq, x_exog)
    # 注意力热图：用 memory 的时间步相似度近似（cosine）
    mem = memory[0].cpu().numpy()  # 取第一个样本 (seq_len, d_model)
    # 时间步相似度矩阵
    sim = mem @ mem.T
    # 归一化到[0,1]
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

plt.figure(figsize=(8,6))
plt.imshow(sim, cmap="inferno", aspect="auto")
plt.colorbar()
plt.title("图4：Transformer编码表示的时间步相似度热图（近似注意力）")
plt.xlabel("时间步")
plt.ylabel("时间步")
plt.tight_layout()
plt.show()

# 6) 合成最终预测：y_hat = y_hat_arima + y_hat_xgb + y_hat_transformer
# 为测试集生成Transformer预测（滑窗滚动）
def rolling_transformer_predict(df_seq, model, seq_len, feat_cols):
    model.eval()
    res2_vals = df_seq["res2"].values
    X_feat = df_seq[feat_cols].values
    yhat_tfm = []
    with torch.no_grad():
        for t in range(seq_len, len(df_seq)):
            x_seq = torch.tensor(res2_vals[t-seq_len:t], dtype=torch.float32).unsqueeze(0)
            x_exog = torch.tensor(X_feat[t], dtype=torch.float32).unsqueeze(0)
            y_hat, _ = model(x_seq, x_exog)
            yhat_tfm.append(y_hat.item())
    # 前seq_len位置用NaN填充
    return np.array([np.nan]*seq_len + yhat_tfm)

valid_train["y_hat_tfm"] = rolling_transformer_predict(train_tfm_df, model, seq_len, tfm_feat_cols)
valid_test["y_hat_tfm"] = rolling_transformer_predict(test_tfm_df, model, seq_len, tfm_feat_cols)

# 汇总最终预测
valid_train["y_hat_total"] = valid_train["y_hat_arima"] + valid_train["y_hat_xgb"] + valid_train["y_hat_tfm"]
valid_test["y_hat_total"] = valid_test["y_hat_arima"] + valid_test["y_hat_xgb"] + valid_test["y_hat_tfm"]

# 图5：最终预测 vs 真实值 + 误差直方图
fig = plt.figure(figsize=(16, 6))
plt.subplot(1,2,1)
plt.plot(valid_test.index, valid_test["y"], color="#FF006E", label="真实 y", linewidth=2.5)
plt.plot(valid_test.index, valid_test["y_hat_total"], color="#06D6A0", label="预测 y_hat_total", linewidth=2.5)
plt.title("图5A：测试集真实 vs 组合预测（总加性）")
plt.legend()

plt.subplot(1,2,2)
err = (valid_test["y_hat_total"] - valid_test["y"]).dropna()
plt.hist(err, bins=30, color="#8338EC", alpha=0.9)
plt.title("图5B：预测误差分布（残差直方图）")
plt.tight_layout()
plt.show()

# 简易指标评估
def rmse(a,b):
    return np.sqrt(np.nanmean((a-b)**2))
def mae(a,b):
    return np.nanmean(np.abs(a-b))
def mape(a,b):
    return np.nanmean(np.abs((a-b)/a)) * 100

# 对齐非NaN
mask = ~valid_test["y_hat_total"].isna()
y_true = valid_test.loc[mask, "y"].values
y_pred = valid_test.loc[mask, "y_hat_total"].values
print("测试集评估：")
print("RMSE:", rmse(y_true, y_pred))
print("MAE:", mae(y_true, y_pred))
print("MAPE(%):", mape(y_true, y_pred))
