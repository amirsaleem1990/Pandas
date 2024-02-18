#!/usr/bin/python3

d = {}
n = []
def get_cols_names(min_val, max_val, dtype):
    global d
    global n
    cols = list(n[(n.min_ >= min_val) & (n.max_ <= max_val)].index)
    n = n[~n.index.isin(cols)]
    for col in cols:
        d[col] = dtype

def pandas_type_casting(df):
    import numpy as np
    import pandas as pd
    global n
    # df = pd.read_csv("users-isprep.zip")
    old = df.memory_usage() / 1024/1024
    
    #numeric cols
    number_cols = list(df.select_dtypes("number").columns)
    n = df[number_cols].fillna(0).agg([min,max]).T.add_suffix("_")
            
    get_cols_names(0, 255, pd.UInt8Dtype())
    get_cols_names(256, 65535, pd.UInt16Dtype())
    get_cols_names(65536, 4294967295, pd.UInt32Dtype())

    get_cols_names(-128, 127, pd.Int8Dtype())
    get_cols_names(-32768, 32767, pd.Int16Dtype())
    get_cols_names(-2147483648, 2147483647, pd.Int32Dtype())

    # date and catagorical datacols
    catagoriacal_cols = list(df.select_dtypes("O").columns)
    date_cols = []
    for i in catagoriacal_cols:
        x = df[i][~df[i].isna()].head()
        try:
            pd.to_datetime(x)
            date_cols.append(i)
        except:
            pass
    catagoriacal_cols = [i for i in catagoriacal_cols if not i in date_cols]

    c = df[catagoriacal_cols].apply(lambda x:x.nunique()/len(df)*100)
    for i in c[c<5].index:
        d[i] = "category"
        
    # del df
    # df = pd.read_csv("users-isprep.zip", parse_dates=date_cols, dtype=d)
    for i in d:
        df[i] = df[i].astype(d[i])
    new = df.memory_usage() / 1024/1024

    m = pd.DataFrame({"new" : new,
                      "old" : old,
                      "Imporovement" : old - new})
    m['Dtype'] = [None] + list(df[list(new.index.drop("Index"))].dtypes.astype(str).values)

    c = df[catagoriacal_cols].apply(lambda x:x.nunique()/len(df)*100)

    m["nunique"] = None
    m.loc[c.index, "nunique"] = list(df[c.index].apply(lambda x:x.nunique() / len(df) * 100).values)

    print("Before :", round(m.old.sum()))
    print("After  :", round(m.new.sum()))
    print("Diff   :", round(m.Imporovement.sum()))
    print("Diff % :", round(m.Imporovement.sum()/m.old.sum(), 2))

    print("\n\nImprovement:")
    print(m.groupby("Dtype").Imporovement.agg([min, max, sum, np.mean, np.median, "count"]))

    print("\n\nDetailed Summary:")
    print(m.to_string())
    return df