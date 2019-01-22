def get_col_like(from_col, col_like):
    return [fc for c in col_like for fc in from_col if c in fc]

