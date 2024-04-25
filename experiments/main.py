import json
import pandas as pd

# run dataPreperation.py, fullEvaluation.py, and simulatePapers.py first

experiment_path = "./data/papers/"
full_evaluation_path = "./output/papers/"

metrics_ordered = ["f1", "pr_rc_auc", "fbeta", "roc_auc", "recall", "accuracy", "precision"]

# baseline 1: use only papers with relevant metrics (f1 score)

# baseline 2: use top 3 algo from each paper

# test 1: use the elo ranking to select the best models

# gold standard: use the full evaluation to select the best models
with open("./output/full/smote_standardize.json", "r") as f:
    gold_standard_raw = f.read()
gold_standard_raw = json.loads(gold_standard_raw.replace("\'", "\""))

gold_standard = {}
for dataset in gold_standard_raw["results"].keys():
    gold_standard[dataset] = pd.DataFrame([[gold_standard_raw["results"][dataset][algo][metric]["cv_score"] for metric in metrics_ordered] for algo in gold_standard_raw["results"][dataset]],
                                          columns= metrics_ordered, index= gold_standard_raw["results"][dataset].keys())
    gold_standard[dataset]["rank"] = gold_standard[dataset].sort_values(metrics_ordered[0], ascending=False).rank(ascending=False)[metrics_ordered[0]]
