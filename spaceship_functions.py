from polars import DataFrame

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# import plotly.io as pio
import polars as pl
from numpy.typing import NDArray

# from plotly.graph_objs._figure import Figure
# from plotly.subplots import make_subplots
from polars import DataFrame
from polars.dataframe.frame import DataFrame

# from pyod.models.knn import KNN
from scipy.stats import chi2_contingency  # , ks_2samp

# import tensorflow_decision_forests as tfdf
# from sklearn.decomposition import PCA
# from skimpy import skim
from sklearn.impute import SimpleImputer

# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

# from sklearn.semi_supervised import LabelPropagation
# import streamlit
from tpot import TPOTClassifier

# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from mlxtend.classifier import StackingClassifier
from tpot import TPOTClassifier

# from sklearn.model_selection import train_test_split
from sklearn.base import clone

# from catboost import CatBoostClassifier
from skopt import BayesSearchCV

# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
# from sklearn.pipeline import Pipeline
from numpy.random import randint

# from sklearn.semi_supervised import LabelSpreading
import plotly.express as px
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# from plotly.subplots import make_subplots


# import train and test dataframes
train: DataFrame = pl.read_csv("train.csv")
test: DataFrame = pl.read_csv("test.csv")

# x and y arrays for modeling
#x_train = np.loadtxt("x_train.csv", delimiter=",", dtype=float)
#y_train = np.loadtxt("y_train.csv", delimiter=",", dtype=float)
#training_x = np.loadtxt("training_y.csv", delimiter=",", dtype=float)
#training_y = np.loadtxt("training_y.csv", delimiter=",", dtype=float)
#validation_x = np.loadtxt("validation_x.csv", delimiter=",", dtype=float)
#validation_y = np.loadtxt("validation_y.csv", delimiter=",", dtype=float)



def make_subplot(figure, df, feature, position, color_series="green") -> None:
    """Makes bar subplot for row and column of figure specified

    Args:
        figure (plotly go): Plotly graph_objects figure
        feature (string): the name of the column within pandas df to make plot of
        position (list): list of integers in [row,column] format for specifying where in figure to plot graph
        labels (list, optional): Title, xlabel, and ylabel for subplots. Defaults to ['',None,None].
    """
    df: DataFrame = df.to_pandas()

    if color_series == "green":
        color: list[str] = [
            "rgb(191,237,204)",
            "rgb(76,145,151)",
            "rgb(33,92,113)",
            "rgb(22,70,96)",
        ]
    elif color_series == "purple":
        color = [
            "rgb(224, 194, 239)",
            "rgb(168, 138, 211)",
            "rgb(108, 95, 167)",
            "rgb(108, 95, 167)",
        ]
    else:
        color = [
            "rgb(117,180,216)",
            "rgb(7, 132, 204)",
            "rgb(35, 114, 181)",
            "rgb(11, 88, 161)",
        ]

    tallies: DataFrame = df[feature].sort_values(ascending=True).value_counts()
    figure.add_trace(
        go.Bar(
            x=tallies.index,
            y=tallies.values,
            name="",
            marker=dict(color=color),
            hovertemplate="%{x} : %{y}",
            text=tallies.values,
        ),
        row=position[0],
        col=position[1],
    )
    figure.update_layout(bargap=0.2)


def create_encoder_mapping(df, feature) -> dict[str, int]:
    """Creates dictionary for mapping to encode categorical features

    Args:
        df (polars dataframe): dataframe of features
        feature (string): name of feature of interest

    Returns:
        encoding_key: dictionary of feature values and numbers for encoding
    """
    df: DataFrame = (
        df.group_by(feature)
        .agg(pl.len().alias("values"))
        .sort("values", descending=True)
    )

    options: List = df[feature].to_list()

    numbers_to_encode = list(range(0, len(options)))
    encoding_key = {options[i]: numbers_to_encode[i] for i in range(len(options))}

    if df[feature].str.contains("Yes").to_list()[0] == True:
        encoding_key: dict[str, int] = {"Yes": 1, "No": 0}

    return encoding_key


def encode_feature(df, feature, encoding_key) -> DataFrame:
    """Encode features using supplied encoding key

    Args:
        df (polars): Dataframe to be modified
        feature (string): feature to be encoded
        encoding_key (dict): dictionary of values and numerical codes

    Returns:
        df: input dataframe with feature replaced by numerical values
    """
    df: DataFrame = df.with_columns(
        df.select(pl.col(feature).replace(encoding_key)).cast({feature: pl.Int64})
    )

    return df


def mark_outliers(df, outlier_indices) -> DataFrame:
    outlier_series: ndarray[int] = np.zeros(len(df), dtype=int)
    for i in outlier_indices:
        outlier_series[i] = 1

    df: DataFrame = df.with_columns(outliers=outlier_series)

    return df


def impute_missing_values(df, feature, method, format) -> DataFrame:
    """Impute missing values with sklearn simple imputer

    Args:
        df (polars): dataframe
        feature (string): feature to be imputed
        method (string): specified strategy parameter for SimpleImputer
        format (string): specifies value used for missing value in df

    Returns:
        imputed_df (polars dataframe): dataframe with imputed values for feature given
    """
    columns: List = df.columns
    array: NDArray = df.fill_null(strategy="zero").to_numpy()

    imputer = SimpleImputer(strategy=method, missing_values=format)
    imputed_array: NDArray = imputer.fit_transform(array)
    imputed_df = pl.DataFrame(imputed_array)

    imputed_df.columns = columns
    return imputed_df


def iterative_duplicate_check(df) -> None:
    column_list = list()

    for col in list(df.columns[1:]):
        column_list.append(col)
        print(
            df.to_pandas().duplicated(subset=column_list).sum(),
            "duplicates",
            column_list,
        )


def polars_crosstab(df, col_a, col_b):
    crosstab = df.pivot(
        values=col_a, index=col_b, columns=col_a, aggregate_function="len"
    ).fill_null(0)
    return crosstab


def calculate_chi2(df, col_a, col_b) -> float:
    crosstab: DataFrame = polars_crosstab(df, col_a, col_b)
    stats, p_value, dof, array = chi2_contingency(crosstab)
    return p_value


def int_range(start, end) -> np.ndarray[int, np.dtype[int]]:
    """Generate np.linspace range for limits given, such that all inclusive consecutive integers are included

    Args:
        start (int): lower limit of range
        end (int): upper limit of range

    Returns:
        numpy array of range: integer range of values
    """
    return np.linspace(start, end, len(range(start, end + 1)), dtype=int)


def skopt_bayesian_search(classifier,x_train,y_train, params, np=False):
    if np:
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
        search = BayesSearchCV(
            estimator=classifier, search_spaces=params, n_jobs=-1, cv=cv
        )
        search.fit(x_train, y_train)
        return search.best_params_ 
    else:
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
        search = BayesSearchCV(
            estimator=classifier, search_spaces=params, n_jobs=-1, cv=cv
        )
        search.fit(x_train, y_train)
        return search.best_params_


def write_predictions_to_csv(predictions, test, csv_name) -> None:
    submissions_df = pd.DataFrame(test["passengerid"], columns=["passengerid"])
    submissions_df["Transported"] = predictions

    # Convert 1s and 0s to boolean
    submissions_df["Transported"] = submissions_df.Transported.apply(
        lambda x: False if x == 0 else True
    )

    submissions_df
    submissions_df.to_csv(csv_name, index=False)


def restrict_columns(array):
    array = array[:, :5]
    return array


def conduct_PCA(x):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=10)
    x = pca.fit_transform(x)
    return x


def int_variable_series(df, variable, n):  # -> Anyndarray[Any, dtype]:
    variable_series = df[variable]
    min: int = variable_series.min()
    max: int = variable_series.max()

    array: NDArray = np.array([round(np.random.uniform(min, max)) for i in range(0, n)])

    return array


def multiclass_variable_series(df, variable, n):
    variable_series = df[variable]
    min: int = variable_series.min()
    max: int = variable_series.max()
    array: NDArray = randint(min, max + 1, dtype=int, size=n)

    return array

def split_cabin_column(df):
    df = df.with_columns(
        pl.col("cabin").str.split("/").list.get(0).alias("deck"),
        pl.col("cabin").str.split("/").list.get(1).alias("cabin_num"),
        pl.col("cabin").str.split("/").list.get(2).alias("side"),
    )
    df = df.drop("cabin")
    return df

def check_grid_permutations(hyperparameter_grid) -> int:
    """Quick function to calculate the number of model permutations created by a hyperparameter grid, to help ring-fence runtimes in GridSearchCV

    Args:
        hyperparameter_grid (dictionary): dictionary of arrays or lists specifying hyperparameter configurations

    Returns:
        combinations: integer representing number of distinct model specifications in the hyperparameter grid
    """
    combinations = list()
    for i in hyperparameter_grid.keys():
        combinations.append(len(hyperparameter_grid[i]))
    return np.prod(combinations, dtype=int)

def calculate_model_statistics(
    y_true, y_predict, beta=0, title="statistics"
) -> DataFrame:
    """Uses actual y and predicted y values to return a dataframe of accuracy, precision, recall, and f-beta values as well as false negative and false posititive rates for a given classifier

    Args:
        y_true (numpy array or data series): dependent variable values from the dataset
        y_predict (_type_): dependent variable values arising from model
        beta (float, optional): Beta value to determine weighting between precision and recall in the f-beta score.Defaults to beta value set in global scope of this notebook.
        title (str, optional): _description_. Defaults to "statistics".

    Returns:
        model_statistics: pandas dataframe of statistics
    """
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
        fbeta_score,
    )

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()

    # calculate statistics from confusion matrix
    accuracy: float = accuracy_score(y_true, y_predict)
    precision: float = precision_score(y_true, y_predict)
    recall: float = recall_score(y_true, y_predict)
    f_score: float = fbeta_score(y_true, y_predict, beta=beta)
    false_negative_rate: float = fn / (tn + fp + fn + tp)
    false_positive_rate: float = fp / (tn + fp + fn + tp)

    return pd.DataFrame(
        data={
            title: [
                accuracy,
                precision,
                recall,
                f_score,
                false_negative_rate,
                false_positive_rate,
            ]
        },
        index=[
            "accuracy",
            "precision",
            "recall",
            "f_score",
            "false_negative_rate",
            "false_positive_rate",
        ],
    )