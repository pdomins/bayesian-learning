from   typing     import Any, Callable
from   data_split import k_fold_n_splits
from   df_utils   import get_column_value_dict
import pandas     as pd
import numpy      as np


def compute_classification_error(df_test : pd.DataFrame, expected_labels : dict[Any, Any], predict : Callable[[pd.Series, Any], Any], model : Any = None) -> int:
    classification_error = 0
    for test_sample_idx, test_sample_series in df_test.iterrows():
        expected_label  = expected_labels[test_sample_idx]
        predicted_label = predict(test_sample_series, model)
        if predicted_label != expected_label : classification_error += 1
    return classification_error

def compute_classification_error_from_df(df : pd.DataFrame, out_label_col : str, predict : Callable[[pd.Series, Any], Any], model : Any = None) -> int:
    df_out_label_dict = get_column_value_dict(df, out_label_col)
    df_input          = df.drop(columns=[out_label_col], inplace=False)
    df_err            = compute_classification_error(df_input, df_out_label_dict, predict, model)
    return df_err

def k_fold_cross_validation(df : pd.DataFrame, possible_out_labels : np.ndarray, out_label_col : str, train : Callable[[pd.DataFrame, np.ndarray], Any], predict : Callable[[pd.Series, Any], Any], k : int, random_state : np.random.Generator = None) -> list[dict[str, Any]]:
    k_splits = k_fold_n_splits(df, k, random_state=random_state)
    k_splits_dict_list = []
    for train_df, test_df in k_splits:
        model     = train(train_df, possible_out_labels)
        train_err = compute_classification_error_from_df(train_df, out_label_col, predict, model)
        test_err  = compute_classification_error_from_df(test_df, out_label_col, predict, model)
        curr_split_dict = {
            "train": {
                "df"  : train_df,
                "err" : train_err
            },
            "test" : {
                "df"   : test_df,
                "err"  : test_err
            }
        }
        k_splits_dict_list.append(curr_split_dict)
    return k_splits_dict_list

