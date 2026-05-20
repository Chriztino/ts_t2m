# *** coding: utf-8 ***
# *** by kumuyu on 2026-05-14 ***

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载EAR5气象数据
def generate_weather_data(n_days=9132):
    ds = pd.read_csv('D:/Apy0/ts_t2m/data/ts_T2M_2000_2024_L77.0_R45.0.csv', parse_dates=['date'], index_col='date')
    temp = ds['ts_sel_season'].values
    date = pd.date_range("2000-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "date": date,
        "temp": temp,
    })
    return df

# 2. 构造时序滞后特征
def add_lag_features(df, vars_to_lag, max_lag=7):
    df = df.copy()
    for var in vars_to_lag:
        for lag in range(1, max_lag + 1):
            df[f"{var}_lag{lag}"] = df[var].shift(lag)
    # 时间特征（年内位置，正弦余弦编码）
    day_of_year = df["date"].dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df

# 3. 简化版级联森林（DeepForest风格）
class SimpleCascadeForestRegressor:
    """
    使用两种树模型（RF + ExtraTrees）组成多层级联，每层把两种模型的预测拼接到原特征上。
    用验证集MSE改善来早停。
    """
    def __init__(self, n_layers_max=3, random_state=42,
                 rf_params=None, et_params=None, tol=1e-4):
        self.n_layers_max = n_layers_max
        self.random_state = random_state
        self.tol = tol
        self.layers_ = []
        self.rf_params = rf_params or {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "random_state": random_state, "n_jobs": -1}
        self.et_params = et_params or {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "random_state": random_state, "n_jobs": -1}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xl_train = np.array(X_train, dtype=np.float32)
        Xl_val = np.array(X_val, dtype=np.float32) if X_val is not None else None

        best_val_mse = np.inf
        for xl in range(self.n_layers_max):
            rf = RandomForestRegressor(**self.rf_params)
            et = ExtraTreesRegressor(**self.et_params)
            rf.fit(Xl_train, y_train)
            et.fit(Xl_train, y_train)

            self.layers_.append((rf, et))

            # 拼接预测作为下一层的附加特征
            rf_train_pred = rf.predict(Xl_train).reshape(-1, 1)
            et_train_pred = et.predict(Xl_train).reshape(-1, 1)
            Xl_train = np.concatenate([Xl_train, rf_train_pred, et_train_pred], axis=1)

            if Xl_val is not None:
                rf_val_pred = rf.predict(Xl_val).reshape(-1, 1)
                et_val_pred = et.predict(Xl_val).reshape(-1, 1)
                Xl_val_next = np.concatenate([Xl_val, rf_val_pred, et_val_pred], axis=1)

                # 计算本层的验证MSE（使用平均预测）
                val_pred_mean = (rf_val_pred.flatten() + et_val_pred.flatten()) / 2.0
                val_mse = mean_squared_error(y_val, val_pred_mean)

                if best_val_mse - val_mse < self.tol:
                    # 早停
                    break
                else:
                    best_val_mse = val_mse
                    Xl_val = Xl_val_next

    def predict(self, X):
        Xl = np.array(X, dtype=np.float32)
        # 逐层生成并拼接预测特征
        for (rf, et) in self.layers_:
            rf_pred = rf.predict(Xl).reshape(-1, 1)
            et_pred = et.predict(Xl).reshape(-1, 1)
            Xl = np.concatenate([Xl, rf_pred, et_pred], axis=1)
        # 最终预测用最后一层两模型的平均
        rf_last, et_last = self.layers_[-1]
        y_hat = (rf_last.predict(np.array(X, dtype=np.float32)) + et_last.predict(np.array(X, dtype=np.float32))) / 2.0
        return y_hat

    def meta_features(self, X):
        """
        返回最后一层的两个模型预测作为元特征（2维），可用于与原特征拼接。
        注意：在避免泄露流程中，X应该严格来自模型训练时允许使用的时间段。
        """
        Xl = np.array(X, dtype=np.float32)
        # Build up features through all layers except the last
        for i, (rf, et) in enumerate(self.layers_[:-1]):
            rf_pred = rf.predict(Xl).reshape(-1, 1)
            et_pred = et.predict(Xl).reshape(-1, 1)
            Xl = np.concatenate([Xl, rf_pred, et_pred], axis=1)

        # Now use the last layer to generate meta features
        rf_last, et_last = self.layers_[-1]
        rf_out = rf_last.predict(Xl).reshape(-1, 1)
        et_out = et_last.predict(Xl).reshape(-1, 1)
        return np.concatenate([rf_out, et_out], axis=1)

    def feature_importances_(self):
        """
        粗略整合最后一层两个模型的特征重要性；由于级联后特征维度增加，这里仅展示最初始特征的重要性。
        """
        rf_last, et_last = self.layers_[-1]
        # 两者取平均作为一个估计
        imp = (rf_last.feature_importances_ + et_last.feature_importances_) / 2.0
        return imp

# 4. 时间感知的OOF元特征生成
def time_aware_oof_features(X_all, y_all, idx_train, idx_val, idx_test, n_splits=3):
    """
    按扩展窗口将训练段划分成若干折，获取训练段的OOF元特征；
    验证段用仅训练集训练的森林生成元特征；
    测试段用训练+验证集训练的森林生成元特征。
    """
    # 将训练段细分为n_splits个时间折
    n_train = len(idx_train)
    fold_sizes = np.array_split(np.arange(n_train), n_splits)
    oof_train_meta = np.zeros((n_train, 2))  # 两个元特征
    models = []

    i = 0
    for i in range(n_splits):
        # 第i折：训练=前i个fold的并集，验证=第i个fold（相对训练段索引）
        train_sub_idx_rel = np.concatenate(fold_sizes[:i]) if i > 0 else np.array([], dtype=int)
        val_sub_idx_rel = fold_sizes[i]
        # 若i=0，训练为空，不合理；我们让i=1开始有效训练
        if i == 0:
            # 暂不产生OOF，下一折覆盖它
            continue
        train_abs_idx = idx_train[train_sub_idx_rel]
        val_abs_idx = idx_train[val_sub_idx_rel]

        X_tr = X_all[train_abs_idx]
        y_tr = y_all[train_abs_idx]
        X_va = X_all[val_abs_idx]
        y_va = y_all[val_abs_idx]

        model = SimpleCascadeForestRegressor(n_layers_max=3, tol=1e-4)
        model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        models.append(model)

        oof_meta = model.meta_features(X_va)
        oof_train_meta[val_sub_idx_rel] = oof_meta

    # 对训练段中第一折（i=0）的样本，由于没有更早的数据，我们可以用第二折的模型产生元特征，但必须强调其可能较弱（或设为均值）
    # 这里用第一个训练好的模型（第二折的模型）生成，以免空值：
    if len(models) > 0:
        first_model = models[0]
        first_fold_rel = fold_sizes[0]
        first_fold_abs = idx_train[first_fold_rel]
        oof_meta_first = first_model.meta_features(X_all[first_fold_abs])
        oof_train_meta[first_fold_rel] = oof_meta_first

    # 验证段元特征：用**整个训练段**训练的模型来生成
    model_val = SimpleCascadeForestRegressor(n_layers_max=3, tol=1e-4)
    model_val.fit(X_all[idx_train], y_all[idx_train], X_val=X_all[idx_val], y_val=y_all[idx_val])
    val_meta = model_val.meta_features(X_all[idx_val])

    # 测试段元特征：用**训练+验证段**训练的模型来生成
    train_val_idx = np.concatenate([idx_train, idx_val])
    model_test = SimpleCascadeForestRegressor(n_layers_max=3, tol=1e-4)
    model_test.fit(X_all[train_val_idx], y_all[train_val_idx], X_val=None, y_val=None)
    test_meta = model_test.meta_features(X_all[idx_test])

    return oof_train_meta, val_meta, test_meta, model_test  # 返回最终测试模型以便获取特征重要性等

# 5. 构造序列数据集
class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y) - 1# y[t+1]作为目标，因此最后一个y无法被当前t预测
    def __getitem__(self, idx):
        # 序列窗口是 [idx - seq_len + 1, ..., idx]
        start = idx - self.seq_len + 1
        if start < 0:
            start = 0
        x_seq = self.X[start:idx+1]  # 保证使用到当前t为止的信息
        # 若序列不足seq_len，则在前面用零填充（或复制第一条）
        if x_seq.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - x_seq.shape[0], x_seq.shape[1]), dtype=np.float32)
            x_seq = np.vstack([pad, x_seq])
        y_target = self.y[idx + 1]  # 预测下一天
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

# 6. LSTM时序模型（PyTorch）
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        h_last = out[:, -1, :]  # 使用最后时刻的隐藏状态
        y_hat = self.fc(h_last).squeeze(-1)
        return y_hat

# 7. 主流程：数据、特征、OOF、训练、评估、绘图
df = generate_weather_data(n_days=9132)
df = add_lag_features(df, vars_to_lag=["temp"], max_lag=7)

# 目标是预测下一天的温度
df["target_next_temp"] = df["temp"].shift(-1)

# 去掉因滞后和目标移位产生的NaN
df = df.dropna().reset_index(drop=True)

# 划分时间段（70%训练，15%验证，15%测试）
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

idx_train = np.arange(0, train_end)
idx_val = np.arange(train_end, val_end)
idx_test = np.arange(val_end, n)

feature_cols = [c for c in df.columns if c not in ["date", "target_next_temp"]]

X_all = df[feature_cols].values.astype(np.float32)
y_all = df["target_next_temp"].values.astype(np.float32)

# 标准化（只用训练段拟合，以避免泄露）
scaler = StandardScaler()
X_all_scaled = np.zeros_like(X_all)
X_all_scaled[idx_train] = scaler.fit_transform(X_all[idx_train])
X_all_scaled[idx_val] = scaler.transform(X_all[idx_val])
X_all_scaled[idx_test] = scaler.transform(X_all[idx_test])

# 时间感知的OOF元特征（避免泄露）
oof_train_meta, val_meta, test_meta, deepforest_final_model = time_aware_oof_features(
    X_all_scaled, y_all, idx_train, idx_val, idx_test, n_splits=4
)

# 把元特征拼接回整体特征（分别在不同段）
X_train_final = np.hstack([X_all_scaled[idx_train], oof_train_meta])
X_val_final = np.hstack([X_all_scaled[idx_val], val_meta])
X_test_final = np.hstack([X_all_scaled[idx_test], test_meta])

# 把不同段拼接为完整序列（仍按时间顺序拼接）
X_full_final = np.vstack([X_train_final, X_val_final, X_test_final])
y_full = y_all  # 同步目标

# 构造PyTorch数据集与加载器
seq_len = 14
train_ds = SeqDataset(X_full_final[idx_train], y_full[idx_train], seq_len=seq_len)
val_ds = SeqDataset(X_full_final[idx_val], y_full[idx_val], seq_len=seq_len)
test_ds = SeqDataset(X_full_final[idx_test], y_full[idx_test], seq_len=seq_len)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# 定义模型、损失、优化器
input_dim = X_full_final.shape[1]
model = LSTMRegressor(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练（带早停）
n_epochs = 50
best_val_loss = np.inf
patience = 6
patience_counter = 0
train_losses = []
val_losses = []

def evaluate_loader(model, loader, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            losses.append(loss.item())
    return np.mean(losses)

for epoch in range(n_epochs):
    model.train()
    batch_losses = []
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)
    val_loss = evaluate_loader(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
    # 早停
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# 恢复最佳验证状态
model.load_state_dict(best_state)

# 测试集预测
model.eval()
y_pred_test = []
y_true_test = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_hat = model(x_batch)
        y_pred_test.append(y_hat.numpy())
        y_true_test.append(y_batch.numpy())
y_pred_test = np.concatenate(y_pred_test)
y_true_test = np.concatenate(y_true_test)

# 评估指标
rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
mae = mean_absolute_error(y_true_test, y_pred_test)
mape = np.mean(np.abs((y_pred_test - y_true_test) / (y_true_test + 1e-6))) * 100
print(f"Test RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%")

# 8. 可视化分析

# Fig-1：测试集真实值 vs 预测值（时间曲线）
plt.figure(figsize=(12, 4))
plt.plot(y_true_test, color="#00FF7F", label="真实温度", linewidth=2)
plt.plot(y_pred_test, color="#FF00FF", label="预测温度", linewidth=2, alpha=0.8)
plt.title("ERA5-T2M：测试集真实值与预测值（时间曲线）")
plt.xlabel("测试集时间索引")
plt.ylabel("温度(°C)")
plt.legend()
plt.tight_layout()
plt.show()

# Fig-2：测试集残差（预测-真实）随时间
residuals = y_pred_test - y_true_test
plt.figure(figsize=(12, 4))
plt.plot(residuals, color="#FF4500", label="残差(预测-真实)", linewidth=1.5)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("EAR5-T2M：测试集残差随时间变化")
plt.xlabel("测试集时间索引")
plt.ylabel("残差(°C)")
plt.legend()
plt.tight_layout()
plt.show()

# Fig-3：训练与验证损失曲线（学习过程）
plt.figure(figsize=(12, 4))
plt.plot(train_losses, label="训练损失", color="#1E90FF", linewidth=2)
plt.plot(val_losses, label="验证损失", color="#FFD700", linewidth=2)
plt.title("EAR5-T2M：训练与验证损失随Epoch变化")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Fig-4：深度森林（最后一层）特征重要性条形图
# 注意：级联后特征维度已扩展，这里仅以初始特征重要性为例做说明
imp = deepforest_final_model.feature_importances_()
# 由于重要性对应的是最后一层模型的输入维度；我们仅截取到初始特征长度进行示意：
init_feat_len = len(feature_cols)
imp_init = imp[:init_feat_len] if len(imp) >= init_feat_len else imp
feat_importance_df = pd.DataFrame({"feature": feature_cols[:len(imp_init)], "importance": imp_init})
feat_importance_df = feat_importance_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_importance_df, x="importance", y="feature", palette="viridis")
plt.title("EAR5-T2M：深度森林特征重要性（Top 15）")
plt.xlabel("重要性(平均)")
plt.ylabel("特征名")
plt.tight_layout()
plt.show()

# Fig-5：残差自相关（ACF）图
def acf(x, nlags=30):
    x = x - np.mean(x)
    acf_vals = [1.0]
    var = np.dot(x, x)
    for lag in range(1, nlags+1):
        cov = np.dot(x[:-lag], x[lag:])
        acf_vals.append(cov / var)
    return np.array(acf_vals)

acf_vals = acf(residuals, nlags=30)
plt.figure(figsize=(12, 4))
lags = np.arange(0, len(acf_vals))
colors = ["#FF1493"if v > 0 else"#00CED1"for v in acf_vals]
plt.bar(lags, acf_vals, color=colors)
plt.title("EAR5-T2M：测试残差自相关（ACF）")
plt.xlabel("滞后阶数")
plt.ylabel("自相关")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.tight_layout()
plt.show()
