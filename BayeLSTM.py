import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成虚拟时间序列
np.random.seed(42)
T = 2000
t = np.arange(T)
trend = 0.0015 * t
season1 = 1.2 * np.sin(2 * np.pi * t / 200)
season2 = 0.6 * np.sin(2 * np.pi * t / 30 + 0.7)
spikes = np.zeros_like(t, dtype=float)
for _ in range(18):
    pos = np.random.randint(50, T-50)
    spikes += np.exp(-0.5*((t-pos)/5)**2) * (np.random.rand()*3 + 1)
noise = 0.4 * np.random.randn(T)
y = 5.0 + trend + season1 + season2 + spikes + noise

# 2. 构造滑窗数据集（避免数据泄露）
def make_windows(series, p, H):
    X, Y = [], []
    for i in range(len(series) - p - H + 1):
        X.append(series[i:i+p])
        Y.append(series[i+p:i+p+H])
    return np.array(X), np.array(Y)

p = 60   # history length
H = 10   # forecast horizon (direct multi-output)
X, Y = make_windows(y, p, H)
n = X.shape[0]

train_end = int(n * 0.7)
val_end = int(n * 0.85)
X_train, Y_train = X[:train_end], Y[:train_end]
X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
X_test, Y_test = X[val_end:], Y[val_end:]

# 标准化：仅用训练集统计量（关键防止泄露）
scaler_x = StandardScaler().fit(X_train.reshape(-1,1))
scaler_y = StandardScaler().fit(Y_train.reshape(-1,1))
def scale_xy(X_raw, Y_raw, sx=scaler_x, sy=scaler_y):
    Xs = sx.transform(X_raw.reshape(-1,1)).reshape(X_raw.shape)
    Ys = sy.transform(Y_raw.reshape(-1,1)).reshape(Y_raw.shape)
    return Xs, Ys
X_train_s, Y_train_s = scale_xy(X_train, Y_train)
X_val_s, Y_val_s = scale_xy(X_val, Y_val)
X_test_s, Y_test_s = scale_xy(X_test, Y_test)

# 3. PyTorch Dataset & Model
class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LSTMDirect(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, H=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Linear(hidden_size, H)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.unsqueeze(-1)  # batch x seq_len x 1
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        h_last = self.dropout(h_last)
        y = self.head(h_last)
        return y

device = torch.device('cuda'if torch.cuda.is_available() else'cpu')

# 4. 训练 / 评估 
def train_one(model, opt, loss_fn, loader):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loss_fn, loader):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze()
            total_loss += loss_fn(pred, yb).item() * xb.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds) if preds else np.zeros((0,H))
    trues = np.vstack(trues) if trues else np.zeros((0,H))
    return total_loss / len(loader.dataset), preds, trues

def fit_model(hparams, epochs=50, verbose=False, return_history=False):
    model = LSTMDirect(input_size=1, hidden_size=hparams['hidden_size'],
                       num_layers=hparams['num_layers'], dropout=hparams['dropout'], H=H).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams.get('weight_decay',0.0))
    train_loader = DataLoader(TimeSeriesDataset(X_train_s, Y_train_s), batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val_s, Y_val_s), batch_size=hparams['batch_size'], shuffle=False)
    history = {'train_loss':[], 'val_loss':[]}
    best_val = float('inf')
    best_state = None
    patience=6
    wait=0
    for ep in range(epochs):
        trl = train_one(model, opt, loss_fn, train_loader)
        vall, _, _ = evaluate(model, loss_fn, val_loader)
        history['train_loss'].append(trl)
        history['val_loss'].append(vall)
        if vall < best_val - 1e-8:
            best_val = vall
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    if return_history:
        return model, best_val, history
    return model, best_val

# 5. 简化贝叶斯优化（GP + EI）
space = {
    'hidden_size': (16, 128),
    'num_layers': (1, 3),
    'lr': (1e-4, 5e-2),
    'dropout': (0.0, 0.5),
    'batch_size': (16, 128)
}

def sample_random_point():
    hs = int(np.round(np.random.uniform(space['hidden_size'][0], space['hidden_size'][1])))
    nl = int(np.round(np.random.uniform(space['num_layers'][0], space['num_layers'][1])))
    lr = 10**np.random.uniform(np.log10(space['lr'][0]), np.log10(space['lr'][1]))
    dr = np.random.uniform(space['dropout'][0], space['dropout'][1])
    bs = int(np.round(np.random.uniform(space['batch_size'][0], space['batch_size'][1])))
    bs = max(8, int(2**np.round(np.log2(bs))))
    return {'hidden_size':hs, 'num_layers':nl, 'lr':lr, 'dropout':dr, 'batch_size':bs}

def vectorize_point(pt):
    return np.array([pt['hidden_size'], pt['num_layers'], np.log10(pt['lr']), pt['dropout'], pt['batch_size']])

# 初始随机样本
np.random.seed(1)
init_n = 6
D_X = []
D_y = []
all_params = []

for i in range(init_n):
    p0 = sample_random_point()
    all_params.append(p0)
    val_loss = fit_model(p0, epochs=30, verbose=False)
    D_X.append(vectorize_point(p0))
    D_y.append(val_loss)
    print("Init", i, "val_loss", val_loss, p0)

D_X = np.vstack(D_X)
D_y = np.array(D_y)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(D_X.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-6)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
gp.fit(D_X, D_y)
y_min = D_y.min()

def expected_improvement(Xcand, gp, y_min, xi=0.01):
    mu, sigma = gp.predict(Xcand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z = (y_min - mu - xi) / sigma
    ei = (y_min - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

# BO 主循环：用随机候选采样来近似最大化获取函数
bo_iters = 18# 可根据算力调小
for it in range(bo_iters):
    n_cand = 1200
    cand_pts = []
    cand_vecs = []
    for _ in range(n_cand):
        pt = sample_random_point()
        cand_pts.append(pt)
        cand_vecs.append(vectorize_point(pt))
    cand_vecs = np.vstack(cand_vecs)
    ei = expected_improvement(cand_vecs, gp, y_min, xi=0.01)
    best_idx = np.argmax(ei)
    next_pt = cand_pts[int(best_idx)]
    all_params.append(next_pt)
    _, val_loss = fit_model(next_pt, epochs=40, verbose=False)
    D_X = np.vstack([D_X, vectorize_point(next_pt)])
    D_y = np.concatenate([D_y, [val_loss]])
    gp.fit(D_X, D_y)
    y_min = D_y.min()
    print(f"BO iter {it+1}/{bo_iters}, val_loss={val_loss:.4f}, y_min={y_min:.4f}")

# 取最优
best_idx = np.argmin(D_y)
best_vec = D_X[best_idx]
best_params = {
    'hidden_size': int(np.round(best_vec[0])),
    'num_layers': int(np.round(best_vec[1])),
    'lr': 10**best_vec[2],
    'dropout': float(best_vec[3]),
    'batch_size': int(np.round(best_vec[4]))
}
best_params['hidden_size'] = int(np.clip(best_params['hidden_size'], space['hidden_size'][0], space['hidden_size'][1]))
best_params['num_layers'] = int(np.clip(best_params['num_layers'], space['num_layers'][0], space['num_layers'][1]))
bs = max(8, int(2**np.round(np.log2(max(8, best_params['batch_size'])))))
best_params['batch_size'] = bs

# 6. 用 train+val 重训最终模型
X_trainval = np.vstack([X_train, X_val])
Y_trainval = np.vstack([Y_train, Y_val])
scaler_x2 = StandardScaler().fit(X_trainval.reshape(-1,1))
scaler_y2 = StandardScaler().fit(Y_trainval.reshape(-1,1))
X_trainval_s = scaler_x2.transform(X_trainval.reshape(-1,1)).reshape(X_trainval.shape)
Y_trainval_s = scaler_y2.transform(Y_trainval.reshape(-1,1)).reshape(Y_trainval.shape)
X_test_s2 = scaler_x2.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
Y_test_s2 = scaler_y2.transform(Y_test.reshape(-1,1)).reshape(Y_test.shape)

def fit_final_model(hparams, epochs=120):
    model = LSTMDirect(input_size=1, hidden_size=hparams['hidden_size'],
                       num_layers=hparams['num_layers'], dropout=hparams['dropout'], H=H).to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams.get('weight_decay',0.0))
    train_loader = DataLoader(TimeSeriesDataset(X_trainval_s, Y_trainval_s), batch_size=hparams['batch_size'], shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test_s2, Y_test_s2), batch_size=256, shuffle=False)
    history = {'train_loss':[]}
    for ep in range(epochs):
        trl = train_one(model, opt, loss_fn, train_loader)
        history['train_loss'].append(trl)
    test_loss, test_preds, test_trues = evaluate(model, loss_fn, test_loader)
    test_preds_unscaled = scaler_y2.inverse_transform(test_preds.reshape(-1,1)).reshape(test_preds.shape)
    test_trues_unscaled = scaler_y2.inverse_transform(test_trues.reshape(-1,1)).reshape(test_trues.shape)
    return model, test_loss, test_preds_unscaled, test_trues_unscaled, history

final_model, test_loss, test_preds_unscaled, test_trues_unscaled, final_history = fit_final_model(best_params, epochs=120)

# 7. MC Dropout 不确定性估计
def mc_dropout_preds(model, X_s, mc_runs=80):
    model.train()
    loader = DataLoader(TimeSeriesDataset(X_s, np.zeros((len(X_s), H))), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for _ in range(mc_runs):
            batch_preds = []
            for xb, _ in loader:
                xb = xb.to(device)
                p = model(xb).cpu().numpy()
                batch_preds.append(p)
            preds.append(np.vstack(batch_preds))
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

mc_mean_s, mc_std_s = mc_dropout_preds(final_model, X_test_s2, mc_runs=60)
mc_mean = scaler_y2.inverse_transform(mc_mean_s.reshape(-1,1)).reshape(mc_mean_s.shape)
mc_std = scaler_y2.scale_[0] * mc_std_s

# 8. 绘图
plt.rcParams['figure.figsize'] = (12,5)
# 图 A: 合成序列，强调 train/val/test 区域
fig, ax = plt.subplots(figsize=(14,4))
ax.plot(t, y, linewidth=1.0, label='Series', color='#ff6f61')
ax.axvspan(0, (train_end+p+H-1), alpha=0.12, color='#7fc97f', label='Train region')
ax.axvspan((train_end+p+H-1), (val_end+p+H-1), alpha=0.12, color='#fdc086', label='Val region')
ax.axvspan((val_end+p+H-1), T, alpha=0.12, color='#beaed4', label='Test region')
ax.set_title('Synthetic Time Series (trend + seasonality + spikes + noise)', fontsize=14)
ax.legend()
plt.tight_layout()

# 图 B: BO 搜索过程（每次评估的 validation MSE）
fig, ax = plt.subplots(figsize=(12,4))
it_nums = np.arange(1, len(D_y)+1)
ax.plot(it_nums, D_y, marker='o', linewidth=1.5, label='Validation MSE per eval', color='#e41a1c')
ax.scatter(it_nums, D_y, c=D_y, cmap='viridis', s=60, edgecolor='k')
ax.set_xlabel('Evaluation #')
ax.set_ylabel('Validation MSE')
ax.set_title('BO: validation loss per eval', fontsize=14)
ax.grid(alpha=0.25)
ax.set_yscale('log')
plt.tight_layout()

# 图 C: 超参数采样散点（hidden_size vs lr），色彩映射为 val loss
fig, ax = plt.subplots(figsize=(10,6))
vecs = D_X
vals_plot = D_y
sc = ax.scatter(vecs[:,0], 10**vecs[:,2], c=vals_plot, s=90, cmap='plasma', edgecolor='k')
ax.set_yscale('log')
ax.set_xlabel('hidden_size')
ax.set_ylabel('learning_rate (log)')
ax.set_title('Hyperparameter samples colored by val MSE', fontsize=14)
cb = plt.colorbar(sc, ax=ax)
cb.set_label('Validation MSE')
plt.tight_layout()

# 图 D: 单个测试窗口的 history + 真值 + 预测与 95% CI
sel_idx = 30
true_hist = X_test[sel_idx]
true_future = Y_test[sel_idx]
pred_mean = mc_mean[sel_idx]
pred_std = mc_std[sel_idx]
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(np.arange(p), true_hist, label='History', linewidth=2, color='#377eb8')
ax.plot(np.arange(p, p+H), true_future, label='True future', linewidth=2, color='#4daf4a')
ax.plot(np.arange(p, p+H), pred_mean, label='Pred mean (MC Dropout)', linewidth=2, color='#ff7f00')
ax.fill_between(np.arange(p, p+H), pred_mean - 1.96*pred_std, pred_mean + 1.96*pred_std, alpha=0.35, color='#ff7f00', label='95% CI')
ax.scatter(np.arange(p, p+H), pred_mean, s=60, color='#ff7f00', edgecolor='k')
ax.set_title('One test-window: history, true future and predicted (95% CI)', fontsize=14)
ax.legend()
plt.tight_layout()

# 图 E: 最终重训的训练 loss 曲线
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(final_history['train_loss'], label='Train loss (final retrain)', color='#984ea3', linewidth=1.6)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.set_title('Final model training loss (retrain on train+val)', fontsize=14)
ax.grid(alpha=0.2)
plt.tight_layout()

# 图 F: 测试集中一段窗口的 first-step 对比（多窗口并列）
fig, ax = plt.subplots(figsize=(14,5))
nplot = 60
x_axis = np.arange(nplot)
ax.plot(x_axis, test_trues_unscaled[:nplot,0], label='True t+1', color='#1b9e77', linewidth=2)
ax.plot(x_axis, test_preds_unscaled[:nplot,0], label='Pred t+1', color='#d95f02', linewidth=2, linestyle='--')
ax.fill_between(x_axis, (test_preds_unscaled[:nplot,0]-1.96*mc_std[:nplot,0]), (test_preds_unscaled[:nplot,0]+1.96*mc_std[:nplot,0]), color='#d95f02', alpha=0.18)
ax.set_title('Test set: first-step predictions vs truth for many windows', fontsize=14)
ax.legend()
plt.tight_layout()

plt.show()

# 最后打印 summary
print("Best params (rounded):", best_params)
print("Best validation MSE (GP observed):", float(D_y.min()))
print("Final test MSE (scaled-space):", float(test_loss))
