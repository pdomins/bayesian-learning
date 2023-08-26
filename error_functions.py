from   typing import Any, Callable
import pandas as pd

def compute_classification_error(df_test : pd.DataFrame, expected_labels : dict[Any, Any], predict : Callable[[pd.Series], Any]):
    classification_error = 0
    for test_sample_idx, test_sample_series in df_test.iterrows():
        expected_label  = expected_labels[test_sample_idx]
        predicted_label = predict(test_sample_series)
        if predicted_label != expected_label : classification_error += 1
    return classification_error