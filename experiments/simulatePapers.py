import os
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import trange

#from ..sitepackages.eloRating import EloSystem


random.seed(24)
datasets = ["./data/eccd/eccd.pkl", "./data/wdbc/wdbc.pkl"]


# define models
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

models = {"lr" : LogisticRegression(),
          "svm": SVC(),
          "knn": KNeighborsClassifier(),
          "nb": GaussianNB(),
          "dt": DecisionTreeClassifier(),
          "rf": RandomForestClassifier(),
          "adab": AdaBoostClassifier(),
          "mlp": MLPClassifier()}


# define metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score, auc, precision_recall_curve

def precision_recall_auc_score(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

metrics = {"precision": precision_score,
           "recall": recall_score,
           "f1": f1_score,
           "roc_auc": roc_auc_score,
           "pr_rc_auc": precision_recall_auc_score,
           "fbeta": fbeta_score,
           "accuracy": accuracy_score}


#define pre-processing
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.svm import LinearSVC
from imblearn import FunctionSampler
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import PolynomialFeatures

def CustomSampler_IQR(X, y):
    df = pd.DataFrame(X)
    features = df.columns
    df['Target'] = y
    indices = [x for x in df.index]
    out_indexlist = []
    for col in features:
        # Using nanpercentile instead of percentile because of nan values
        Q1 = np.nanpercentile(df[col], 25.)
        Q3 = np.nanpercentile(df[col], 75.)
        cut_off = (Q3 - Q1) * 1.5
        upper, lower = Q3 + cut_off, Q1 - cut_off
        outliers_index = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
        out_indexlist.extend(outliers_index)
    # using set to remove duplicates
    out_indexlist = list(set(out_indexlist))
    clean_data = np.setdiff1d(indices, out_indexlist)
    return X[clean_data], y[clean_data]


preprocessing = {"normalize": Normalizer(),
                 "standardize": StandardScaler(),
                 "feature selection low var": VarianceThreshold(threshold=(.8 * (1 - .8))),
                 "feature selection l1": SelectFromModel(LinearSVC(dual="auto", penalty="l1")),
                 "feature aggregation pca": PCA(n_components=6),
                 "feature creation": PolynomialFeatures(2)
                }

# define resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

resampling = {"smote": SMOTE(sampling_strategy= 0.1),
              "outlier removal iqr": FunctionSampler(func=CustomSampler_IQR, validate = False),
              "random under sampling": RandomUnderSampler(sampling_strategy= 0.1),
              "random over sampling": RandomOverSampler(sampling_strategy= 0.1)
              }



for i in trange(0, 50, desc= "Simulating Papers"):
    if not os.path.exists(f"./output/papers/researcher_{i}.json"):
        run_models = [list(models.keys())[i] for i in random.sample(range(0, len(models.keys())), random.randint(1, 5))]
        run_metrics = [list(metrics.keys())[i] for i in random.sample(range(0, len(metrics.keys())), random.randint(1, 4))]
        run_preprocessing = [list(preprocessing.keys())[i] for i in random.sample(range(0, len(preprocessing.keys())), math.ceil((random.randint(0, 4)/4)))]
        run_resampling = [list(resampling.keys())[i] for i in random.sample(range(0, len(resampling.keys())), math.ceil((random.randint(0, 4)/4)))]

        results_dict = {"config": {"models": run_models, "metrics": run_metrics, "preprocessing": run_preprocessing, "resampling": run_resampling}, "results": {}}

        for dataset in datasets:
            print("Dataset: ", dataset)
            results_dict["results"][Path(dataset).stem] = {}
            df = pd.read_pickle(dataset)

            # allow variation in datasets
            sample_size = random.randint(50, 100)/100
            df = df.sample(frac=sample_size).reset_index(drop=True)
            print("Sample size: ", sample_size)

            X = df.drop(columns=["Target"]).values
            y = df["Target"].values

            # allow for different preprocessing steps
            for run_preprocess in run_preprocessing:
                print("Preprocessing: ", run_preprocess)
                if run_preprocess == "feature selection l1":
                    X = preprocessing[run_preprocess].fit_transform(X, y)
                else:
                    X = preprocessing[run_preprocess].fit_transform(X)

            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=random.randint(10, 50)/100, random_state=42)

            # allow for different resampling steps
            for run_resample in run_resampling:
                print("Resampling: ", run_resample)
                train_X, train_y = resampling[run_resample].fit_resample(train_X, train_y)

            # allow for different models and metrics
            for run_model in run_models:
                results_dict["results"][Path(dataset).stem][run_model] = {}
                print("Model: ", run_model)
                run_model_init = models[run_model]
                run_model_init.fit(train_X, train_y)
                predictions = run_model_init.predict(test_X)

                for run_metric in run_metrics:
                    print("Metric: ", run_metric)
                    if not run_metric in ["roc_auc", "pr_rc_auc"]:
                        predictions = (pd.Series(predictions) > 0.5).astype(int).values

                    if run_metric == "fbeta":
                        score = metrics[run_metric](test_y, predictions, beta= 0.5)
                    else:
                        score = metrics[run_metric](test_y, predictions)

                    print(score)
                    results_dict["results"][Path(dataset).stem][run_model][run_metric] = score

        with open(f"./output/papers/researcher_{i}.json", "w") as f:
            f.write(str(results_dict))