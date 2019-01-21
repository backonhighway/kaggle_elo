def drop_col(df, drop_col_list):
    df_cols = df.columns
    actual_drop_cols = [c for c in drop_col_list if c in df_cols]
    return df.drop(columns=actual_drop_cols)


def drop_col_like(df, drop_col_like_list):
    df_cols = df.columns
    actual_drop_cols = [before for c in drop_col_like_list for before in df_cols if c in before]
    print(actual_drop_cols)
    return df.drop(columns=actual_drop_cols)

