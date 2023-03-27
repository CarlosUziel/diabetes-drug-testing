from copy import deepcopy
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


def aggregate_dataset(
    data: pd.DataFrame, grouping_field_list: Iterable[str], array_field: str
) -> Tuple[pd.DataFrame, Iterable[str]]:
    """
    Aggregate dataset to the right level for modeling

    Args:
        data: Input dataset of EHR data at line level.
        grouping_field_list: Field to group data by.
        array_field: Field used to create dummy variables.

    Returns:
        A dataframe ready for modeling.
    """
    df = (
        deepcopy(data)
        .groupby(grouping_field_list)[["encounter_id", array_field]]
        .apply(lambda x: x[array_field].values.tolist())
        .reset_index()
        .rename(columns={0: array_field + "_array"})
    )

    dummy_df = (
        pd.get_dummies(df[array_field + "_array"].apply(pd.Series).stack())
        .groupby(level=0)
        .sum()
    )

    dummy_col_list = [col.lower().replace(" ", "_") for col in list(dummy_df.columns)]
    concat_df = pd.concat([df, dummy_df], axis=1)
    concat_df.columns = [
        col.lower().replace(" ", "_") for col in list(df.columns)
    ] + dummy_col_list

    return concat_df, dummy_col_list


def preprocess_df(
    df: pd.DataFrame,
    cat_fields: Iterable[str],
    num_fields: Iterable[str],
    target_field: str,
) -> pd.DataFrame:
    """
    Preprocess dataframe by casting to appropriate types and imputing missing values.

    Args:
        df: Dataframe to be processed.
        cat_fields: Fields indicating categorical values.
        num_fields: Fields indicating numerical values.
        target_field: Field with the target variable.
    """
    df = df.copy()

    df[target_field] = df[target_field].astype(float)
    for c in cat_fields:
        df[c] = df[c].astype(str)
    for numerical_column in num_fields:
        df[numerical_column] = df[numerical_column].fillna(df[numerical_column].mean())
    return df


def df_to_dataset(
    df: pd.DataFrame, target_field: str, batch_size: int = 32
) -> tf.data.Dataset:
    """Dataframe to tensorflow dataset

    *adapted from https://www.tensorflow.org/tutorials/structured_data/feature_columns

    Args:
        df: Dataframe to be processed.
        target_field: Target field name.
        batch_size: Batch size for the dataset.

    Returns:
        Tensorflow dataset
    """
    df = df.copy()
    labels = df.pop(target_field)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


# build vocab for categorical features
def write_vocabulary_file(
    vocab_list: np.array,
    vocab_file: Path,
    default_value: str = "00",
):
    """Generate vocabulary file

    Args:
        vocab_list: List of unique categories (vocabulary entries).
        vocab_file: File to save the vocabulary to.
        default_value: Value to include in the first row.
    """
    pd.DataFrame(np.insert(vocab_list, 0, default_value, axis=0)).to_csv(
        vocab_file, index=None, header=None
    )


def build_vocab_files(
    df: pd.DataFrame, cat_fields: Iterable[str], save_dir: Path, **kwargs
) -> Iterable[Path]:
    """Build vocabulary files for each categorical field.

    Args:
        df: Dataframe to be processed.
        cat_fields: Fields indicating categorical values.
    """
    vocab_files_list = []
    for c in cat_fields:
        vocab_file = save_dir.joinpath(f"{c}_vocab.txt")
        write_vocabulary_file(df[c].unique(), vocab_file, **kwargs)
        vocab_files_list.append(vocab_file)

    return vocab_files_list


def show_group_stats_viz(df: pd.DataFrame, group_field: str):
    """Print group statistics.

    Args:
        df: Dataframe with group.
        group_field: Field by which to group the data.
    """
    print(df.groupby(group_field).size())
    print(df.groupby(group_field).size().plot(kind="barh"))


import functools
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


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


def patient_dataset_splitter(
    df: pd.DataFrame,
    patient_field: str = "patient_nbr",
    test_val_size: float = 0.2,
    random_seed: int = 8080,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split patient dataset into train, validation and test sets. To avoid data leakage,
    the same patient will only reside in one partition.

    Args:
        df: Input dataset that will be split
        patient_field: Column for patient IDs.
        test_val_size: Percentage of the total dataset to be in test and validation
            sets.

    Returns:
        Train dataframe.
        Validation dataframe.
        Test dataframe.
    """
    patients = df[patient_field].unique()
    train_patients, test_patients = train_test_split(
        patients, test_size=test_val_size, random_state=random_seed
    )
    train_patients, val_patients = train_test_split(
        train_patients, test_size=len(test_patients), random_state=random_seed
    )
    return (
        deepcopy(df[df[patient_field].isin(train_patients)]),
        deepcopy(df[df[patient_field].isin(val_patients)]),
        deepcopy(df[df[patient_field].isin(test_patients)]),
    )


def demo(feature_column, example_batch):
    feature_layer = tf.keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))
    return feature_layer(example_batch)


def create_tf_categorical_feature_cols(
    cat_fields: Iterable[str], vocab_files: Iterable[Path]
) -> Iterable[Any]:
    """
    Create TF categorical features.

    Args:
        cat_fields: categorical field list that will be transformed with TF feature
            column.
        vocab_files: string, the path where the vocabulary text files are located

    Returns:
        List of TF feature columns
    """
    return [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_file(
                key=cat_field,
                vocabulary_file=str(vocab_file),
                num_oov_buckets=1,
            )
        )
        for cat_field, vocab_file in zip(cat_fields, vocab_files)
    ]


def normalize_numeric_with_zscore(col: tf.Tensor, mean: float, std: float) -> tf.Tensor:
    """
    Performs z-score normalization.

    Args:
        col: Column values.
        mean: Mean value used for normalization.
        std: Standard deviation value used for normalization.

    Returns:
        Normalized column values.
    """
    return (tf.cast(col, dtype=tf.float64) - mean) / std


def create_tf_numeric_feature(
    col: str, mean: float, std: float, default_value: float = 0.0
) -> Any:
    """
    Create TF numerical feature.

    Args:
        col: Numerical column name.
        mean: the mean for the column in the training data.
        std: the standard deviation for the column in the training data.
        default_value: the value that will be used for imputing the field.

    Returns:
        TF feature column representation of the input field.
    """
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=mean, std=std)
    return tf.feature_column.numeric_column(
        key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64
    )


def create_tf_numerical_feature_cols(
    data: pd.DataFrame, num_fields: Iterable[str]
) -> Iterable[Any]:
    """Compute TF numerical features with z-score normalization and pre-calculation of
        mean and standard deviation.

    Args:
        data: Dataframe with numerical fields.
        num_fields: Numerical features names.

    Returns:
        List of TF numeric features.
    """
    return [
        create_tf_numeric_feature(
            num_field, **data[num_field].describe()[["mean", "std"]].to_dict()
        )
        for num_field in num_fields
    ]


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    """Adapted from TF Probability"""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(
                        loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    """Adapted from TF Probability"""
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=t, scale=1),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def build_sequential_model(feature_layer: Any) -> tf.keras.Sequential:
    """Build regression model

    Args:
        feature_layer: Dense features layer.

    Returns:
        Regression model.
    """
    return tf.keras.Sequential(
        [
            feature_layer,
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable),
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Normal(
                    loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:])
                )
            ),
        ]
    )


def build_train_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    feature_layer: Any,
    epochs: int = 5,
    loss_metric: str = "mse",
) -> Tuple[tf.keras.Sequential, Any]:
    """Build and train a regression model.

    Args:
        train_ds: Train dataset.
        val_ds: Validation dataset.
        feature_layer: Layer of dense features.
        epochs: Number of epochs to train.
        loss_metric: Loss function name.

    Returns:
        Trained model
        Training history
    """
    model = build_sequential_model(feature_layer)
    model.compile(optimizer="adam", loss=loss_metric, metrics=[loss_metric])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=loss_metric, patience=50)
    history = model.fit(
        train_ds, validation_data=val_ds, callbacks=[early_stop], epochs=epochs
    )
    return model, history


def get_student_binary_prediction(
    df: pd.DataFrame, pred_mean_col: str, time_th: int = 5
) -> pd.Series:
    """
    Convert regression output to binary by checking whether the patient is predicted to
        stay in the hospital longer than is required to be part of the drug clinical
        trial.

    Args:
        df: pandas dataframe prediction output dataframe
        pred_mean_col: Probability mean prediction field
        time_th: Minimum number of predicted days in the hospital for a patient to be
            part of the clinical trial.

    Returns:
        Dataframe with a new column for binary labels.
    """
    df = deepcopy(df)
    df["pred_binary"] = (df[pred_mean_col] > time_th).astype(int)
    return df
