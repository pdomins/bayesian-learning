from   typing import Any
import pandas as pd

def get_column_value_dict(df : pd.DataFrame, column : str) -> dict[Any, Any] :
    return df[[column]].to_dict()[column]