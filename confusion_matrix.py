import pandas as pd
import numpy as np
from typing import Any

def __init_confusion_matrix_rows__(out_labels : np.ndarray) -> dict[Any, np.ndarray] :
    out_labels_size = out_labels.shape[0]
    return dict(map(lambda x : (x, np.zeros(out_labels_size)), out_labels))

def get_confusion_matrix_row(conf_mat : pd.DataFrame, real : Any) -> pd.DataFrame:
    return conf_mat.loc[real].to_frame().T

def get_confusion_matrix_value(conf_mat : pd.DataFrame, real : Any, predicted : Any) -> float:
    return conf_mat[predicted][real]

def calculate_confusion_matrix(possible_out_labels : np.ndarray, predicted : dict[Any, Any], expected : dict[Any, Any]) -> pd.DataFrame:
    data        = __init_confusion_matrix_rows__(possible_out_labels)
    conf_mat_df = pd.DataFrame(columns=possible_out_labels, index=possible_out_labels, data=data)
    for expected_idx, expected_label in expected.items():
        predicted_label = predicted[expected_idx]
        conf_mat_df[predicted_label][expected_label] += 1
    return conf_mat_df

def calculate_relative_confusion_matrix(possible_out_labels : np.ndarray, predicted : dict[Any, Any], expected : dict[Any, Any]) -> pd.DataFrame:
    conf_mat_df     = calculate_confusion_matrix(possible_out_labels, predicted, expected)
    real_count      = conf_mat_df.sum(axis='columns')
    real_count      = real_count.replace(0, 1)
    rel_conf_mat_df = conf_mat_df.div(real_count, axis='index')
    return rel_conf_mat_df

def calculate_true_positives_from_confusion_matrix(confusion_matrix : pd.DataFrame, positive_label : Any) -> float:
    return confusion_matrix.loc[positive_label][positive_label]

def calculate_false_positives_from_confusion_matrix(confusion_matrix : pd.DataFrame, positive_label : Any) -> float:
    return confusion_matrix.drop([positive_label], axis="index")[[positive_label]].values.sum()

def calculate_false_negatives_from_confusion_matrix(confusion_matrix : pd.DataFrame, positive_label : Any) -> float:
    return confusion_matrix.drop([positive_label], axis="columns").loc[positive_label].values.sum()

def calculate_true_negatives_from_confusion_matrix(confusion_matrix : pd.DataFrame, positive_label : Any) -> float:
    return confusion_matrix.drop([positive_label], axis="index").drop([positive_label], axis="columns").values.sum()

def __init_per_label_confusion_matrix__() -> pd.DataFrame:
    return pd.DataFrame(data={ "P" : np.zeros(2),
                               "F" : np.zeros(2)
                        }, index=["P", "F"])

def calculate_per_label_confusion_matrix(possible_out_labels : np.ndarray, predicted : dict[Any, Any], expected : dict[Any, Any]) -> pd.DataFrame:
    conf_mat                    = calculate_confusion_matrix(possible_out_labels, predicted, expected)
    per_label_conf_mats         = dict()
    for possible_out_label in possible_out_labels:
        curr_label_conf_mat = __init_per_label_confusion_matrix__()
        curr_label_conf_mat.loc["P"]["P"] = calculate_true_positives_from_confusion_matrix(conf_mat, possible_out_label)
        curr_label_conf_mat.loc["F"]["P"] = calculate_false_positives_from_confusion_matrix(conf_mat, possible_out_label)
        curr_label_conf_mat.loc["P"]["F"] = calculate_false_negatives_from_confusion_matrix(conf_mat, possible_out_label)
        curr_label_conf_mat.loc["F"]["F"] = calculate_true_negatives_from_confusion_matrix(conf_mat, possible_out_label)
        per_label_conf_mats[possible_out_label] = curr_label_conf_mat
    return per_label_conf_mats
