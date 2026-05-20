# *** coding: utf-8 ***
# *** by kumuyu on 2026-05-13 ***

from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

filepath = Path('F:/TS_PY/KAZ_T2M_2000_2024_daily/clip_nc/')
filelist = list(filepath.glob('era5_20*_clip.nc'))

start_time = '2000-01-01'       # 起始时间 (YYYY-MM-DD)
end_time   = '2024-12-31'       # 结束时间 (YYYY-MM-DD)

for lat in np.arange(56,44.9,-0.25):    # 纬度56~45，EAR5数据分辨率是0.25°*0.25°
    for lon in np.arange(46,84.2,0.25):     # 经度46~84
        T2M_tot = []    # 初始化T2M空列表
        for file in filelist:   # 遍历文件列表
            with xr.open_dataset(file) as ds:
                f = ds.sel(valid_time=slice(start_time, end_time)).sel(latitude=lat, longitude=lon, method='nearest')
                t = f['t2m'].data  
                t = t - 273.15     # 转换为摄氏度
                T2M_tot.extend(t)  # T2M列表末尾一次性追加新序列
            ts_sel_season = T2M_tot[:]   # 可指定时间切片索引
            df = pd.DataFrame({'ts_sel_season': ts_sel_season})
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
            df.to_csv(f"{filepath}/ts_T2M_2000_2024_grid_R{(56-lat)/0.25:.0f}_C{(lon-46)/0.25:.0f}.csv", index=False, sep=',')
