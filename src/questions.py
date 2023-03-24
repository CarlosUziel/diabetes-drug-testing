import os
from copy import deepcopy

import pandas as pd


# Question 3
def reduce_dimension_ndc(data: pd.DataFrame, ndc: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new field with generic drug names.

    Args:
        data: Input dataset of EHR data at line level.
        ndc: Mapping of NDC codes to generic drug names.

    Returns:
        Original dataset with a new field indicating generic drug names.
    """
    ndc_drug_map = ndc.set_index("NDC_Code")["Non-proprietary Name"].to_dict()
    df = deepcopy(data)
    df["generic_drug_name"] = df["ndc_code"].map(ndc_drug_map)
    return df


# Question 4
def select_first_encounter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Select the first encounter for each patient in the dataset.

    Args:
        data: Input dataset of EHR data at line level.

    Returns:
        A Dataframe with only the first encounter for a given patient
    """
    return deepcopy(
        data.sort_values("encounter_id", ascending=True)
        .groupby("patient_nbr")
        .first()
        .reset_index()
    )


# Question 6
def patient_dataset_splitter(df, patient_key="patient_nbr"):
    """
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    """
    return train, validation, test


# Question 7


def create_tf_categorical_feature_cols(
    categorical_col_list, vocab_dir="./diabetes_vocab/"
):
    """
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    """
    output_tf_list = []
    for c in categorical_col_list:
        os.path.join(vocab_dir, c + "_vocab.txt")
        """
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        """
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list


# Question 8
def normalize_numeric_with_zscore(col, mean, std):
    """
    This function can be used in conjunction with the tf feature column for normalization
    """
    return (col - mean) / std


def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    """
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    """
    return tf_numeric_feature


# Question 9
def get_mean_std_from_preds(diabetes_yhat):
    """
    diabetes_yhat: TF Probability prediction object
    """
    m = "?"
    s = "?"
    return m, s


# Question 10
def get_student_binary_prediction(df, col):
    """
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    """
    return student_binary_prediction
