import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_df(df, scale_col, no_scale_col):
    ss = StandardScaler()
    df_ = ss.fit_transform(df[scale_col])
    ret_df = pd.DataFrame(df_, index=df.index, columns=scale_col)
    for col in no_scale_col:
        ret_df[col] = df[col]
    return ret_df


def prep_meta(df, scale_col, transformer):
    df_ = transformer.transform(df[scale_col])
    ret_df = pd.DataFrame(df_, index=df.index, columns=scale_col)
    return ret_df


def prep_series2(df, scale_col, transformer):

    df_ = transformer.transform(df[scale_col])
    ret_df = pd.DataFrame(df_, index=df.index, columns=scale_col)
    ret_df["object_id"] = df["object_id"]
    ret_df["passband"] = df["passband"]

    # ohe_df = pd.get_dummies(df["passband"], prefix="pass")
    # ret_df = pd.concat([ret_df, ohe_df], axis=1)

    # print(ret_df.query("object_id == 74256178").head())
    # print(df.query("object_id == 74256178").head())

    return ret_df
