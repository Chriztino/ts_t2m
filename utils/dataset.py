import pandas as pd
# from Ipython.display import display

def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols,names = list(),list()

    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names += [f'var{j}(t-{i})' for j in range(n_vars)]
    
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0 :
            names += [f'var{j}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j}(t+{i})' for j in range(n_vars)]

    dataset = pd.concat(cols,axis=1)
    dataset.columns = names

    # 删除空值行
    if dropnan:
        dataset.dropna(inplace=True)

    dataset.reset_index(inplace=True,drop=True)

    return dataset