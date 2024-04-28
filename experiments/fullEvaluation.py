import os.path
import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

#from ..sitepackages.eloRating import EloSystem

from tqdm import tqdm

random.seed(24)
datasets = ["./data/eccd/eccd.pkl", "./data/wdbc/wdbc.pkl"]

def flatten(xss):
    return [x for xs in xss for x in xs]

# define models
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

models = {"lr" : LogisticRegression(),
          "svm": SVC(),
          "knn": KNeighborsClassifier(),
          "nb": GaussianNB(),
          "dt": DecisionTreeClassifier(),
          "rf": RandomForestClassifier(),
          "adab": AdaBoostClassifier(),
          "mlp": MLPClassifier(),
          "lda": LinearDiscriminantAnalysis(),
          "lasso": Lasso(),
          "sgd": SGDClassifier()
          }


# define metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, fbeta_score, auc, precision_recall_curve, balanced_accuracy_score, average_precision_score

def precision_recall_auc_score(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

metrics = {"precision": precision_score,
           "avg precision": average_precision_score,
           "recall": recall_score,
           "f1": f1_score,
           "roc_auc": roc_auc_score,
           "pr_rc_auc": precision_recall_auc_score,
           "fbeta": fbeta_score,
           "accuracy": accuracy_score,
           "blcd accuracy": balanced_accuracy_score}


#define resampling and pre-processing
from sklearn.preprocessing import Normalizer, StandardScaler
from imblearn.over_sampling import SMOTE

resampling = {"wdbc": {"smote": SMOTE(sampling_strategy=0.5)},
              "eccd":{"smote": SMOTE(sampling_strategy=0.1)}}
preprocessing = {"standardize": StandardScaler()}

pbar = tqdm(total=len(list(set(flatten([[key2 for key2 in resampling[key1].keys()] for key1 in resampling.keys()])))) * len(preprocessing) * len(datasets) * len(models) * 5, desc= "Full Evaluation")

for resample in list(set(flatten([[key2 for key2 in resampling[key1].keys()] for key1 in resampling.keys()]))):
    for preprocess in preprocessing:
        if not os.path.exists(f"./output/full/{resample}_{preprocess}.json"):
            results_dict = {"config": {"metrics": list(metrics.keys()), "preprocessing": [preprocess], "resampling": [resample]}, "results": {}}
            pbar.set_description("Full Evaluation - " + resample + " " + preprocess)
            results_dict_temp = {}
            for dataset in datasets:
                print("Dataset: ", dataset)
                if not dataset in results_dict_temp.keys():
                    results_dict_temp[Path(dataset).stem] = {}

                df = pd.read_pickle(dataset)
                X = df.drop(columns=["Target"]).values
                y = df["Target"].values

                if preprocess == "feature selection l1":
                    X = preprocessing[preprocess].fit_transform(X, y)
                else:
                    X = preprocessing[preprocess].fit_transform(X)

                # do 5-fold cross validation
                for i in range(5):
                    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=i)

                    if not resample == "outlier removal iqr":
                        if train_y.sum() / (len(train_y) - train_y.sum()) < resampling[Path(dataset).stem][resample].sampling_strategy:
                            train_X, train_y = resampling[Path(dataset).stem][resample].fit_resample(train_X, train_y)
                    else:
                        train_X, train_y = resampling[Path(dataset).stem][resample].fit_resample(train_X, train_y)

                    for model in models:
                        if not model in results_dict_temp[Path(dataset).stem].keys():
                            results_dict_temp[Path(dataset).stem][model] = {}

                        model_init = models[model]
                        model_init.fit(train_X, train_y)
                        predictions = model_init.predict(test_X)

                        for metric in metrics:
                            if not metric in ["roc_auc", "pr_rc_auc"]:
                                predictions = (pd.Series(predictions) > 0.5).astype(int).values

                            if metric == "fbeta":
                                score = metrics[metric](test_y, predictions, beta=0.5)
                            else:
                                score = metrics[metric](test_y, predictions)

                            if not metric in results_dict_temp[Path(dataset).stem][model].keys():
                                results_dict_temp[Path(dataset).stem][model][metric] = {}

                            results_dict_temp[Path(dataset).stem][model][metric][f"cv_{i}"] = score
                        pbar.update(1)

            for dataset in datasets:
                for model in models:
                    for metric in metrics:
                        results_dict_temp[Path(dataset).stem][model][metric]["cv_score"] = np.mean([results_dict_temp[Path(dataset).stem][model][metric][f"cv_{i}"] for i in range(5)])

            results_dict["results"] = results_dict_temp

            with open(f"./output/full/{resample}_{preprocess}.json", "w") as f:
                f.write(str(results_dict))
        else:
            pbar.update(len(datasets) * len(models) * 5)