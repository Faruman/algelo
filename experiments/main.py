import json
import random
from tqdm import trange

import numpy as np
import pandas as pd

from glob import glob

import itertools

from sklearn.metrics import ndcg_score
import seaborn as sns

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

import sys
import os
sys.path.append(os.path.abspath('../sitepackages'))
from sitepackages.eloRating import EloSystem

import scipy

# run dataPreperation.py, fullEvaluation.py, and simulatePapers.py first

experiment_paths = ["output/papers/wdbc/", "output/papers/eccd/"]
full_evaluation_path = "./output/full/"

metrics_ordered = {"eccd": ["f1", "pr_rc_auc", "fbeta", "roc_auc", "recall", "blcd accuracy", "accuracy", "avg precision", "precision"],
                   "wdbc": ["recall", "roc_auc", "f1", "pr_rc_auc", "blcd accuracy", "fbeta", "accuracy", "avg precision", "precision"]}

time_intervals = list(range(2, 50))

rankings = dict(zip(time_intervals, [{} for _ in time_intervals]))
datasets = ["wdbc", "eccd"]

def get_performanceTable(paper_raw, id):
    performance_tables = pd.DataFrame([], columns= paper_raw["config"]["metrics"] + ["model", "dataset", "rank"])
    for dataset in paper_raw["results"].keys():
        temp = pd.DataFrame([[paper_raw["results"][dataset][algo][metric] for metric in paper_raw["config"]["metrics"]] for algo in paper_raw["results"][dataset]], columns= paper_raw["config"]["metrics"], index= paper_raw["config"]["models"])
        temp = temp.reset_index().rename(columns={"index": "model"})
        temp["dataset"] = paper_raw["config"]["target_dataset"]
        temp["useCase_confidence"] = int(dataset == paper_raw["config"]["target_dataset"])
        temp["paper_confidence"] = paper_raw["config"]["sample_pct"]
        temp["rank"] = temp.sort_values(paper_raw["config"]["metrics"][0], ascending=False).rank(ascending=False)[paper_raw["config"]["metrics"][0]]
        performance_tables = pd.concat((performance_tables, temp))
    performance_tables["id"] = id
    performance_tables = performance_tables.melt(id_vars= ["id","dataset", "useCase_confidence", "paper_confidence", "model", "rank"], value_vars= paper_raw["config"]["metrics"], var_name= "metric", value_name= "score")
    return performance_tables

def flatten(l):
    if not isinstance(l, list):
        return [l]
    flat = []
    for sublist in l:
        flat.extend(flatten(sublist))
    return flat

# load papers
combined_paper_performance_table = pd.DataFrame([], columns= ["id", "dataset", "model", "rank", "metric", "score"])
for i, file in enumerate(flatten([list(glob(f"{experiment_path}researcher_*.json")) for experiment_path in experiment_paths])):
    print(file)
    with open(file, "r") as f:
        paper = f.read()
    paper = json.loads(paper.replace("\'", "\"").replace("nan", '""'))
    combined_paper_performance_table = pd.concat((combined_paper_performance_table, get_performanceTable(paper, i)))
combined_paper_performance_table = combined_paper_performance_table.reset_index(drop=True)
combined_paper_performance_table["score"] = combined_paper_performance_table["score"].apply(lambda x: 0 if x == "" else float(x))
combined_paper_performance_table["id"] = combined_paper_performance_table["id"] - combined_paper_performance_table["id"].min()

#load gold standard
gold_standard_raw_results = []
gold_standard_raw_configs = {}
for i, file in enumerate(glob(f"{full_evaluation_path}/*.json")):
    with open(file, "r") as f:
        paper = f.read()
    temp_results = json.loads(paper.replace("\'", "\"").replace("nan", '""'))["results"]
    temp_results = flatten([[[{"index": i, "dataset": ds, "model": mdl, "metric": me, "score": temp_results[ds][mdl][me]["cv_score"]} for me in temp_results[ds][mdl]] for mdl in temp_results[ds]] for ds in temp_results.keys()])
    gold_standard_raw_results.append(temp_results)
    temp_config = json.loads(paper.replace("\'", "\"").replace("nan", '""'))["config"]
    gold_standard_raw_configs[i] = temp_config
gold_standard_raw_results = pd.DataFrame(flatten(gold_standard_raw_results))
gold_standard_raw_results["score"] = gold_standard_raw_results["score"].apply(lambda x: 0 if x == "" else float(x))

results = pd.DataFrame()

for type in ["default", "sparse", "dense"]:
    print("Type: {}".format(type))

    if type == "sparse":
        value_counts = combined_paper_performance_table["id"].value_counts()
        lower_q_upper_bound = value_counts.quantile(0.25)
        value_counts = value_counts[value_counts <= lower_q_upper_bound]
        value_counts = value_counts.index
        combined_paper_performance_table_temp = combined_paper_performance_table.loc[combined_paper_performance_table["id"].isin(value_counts)]
    elif type == "dense":
        value_counts = combined_paper_performance_table["id"].value_counts()
        higher_q_lower_bound = value_counts.quantile(0.75)
        value_counts = value_counts[value_counts >= higher_q_lower_bound]
        value_counts = value_counts.index
        combined_paper_performance_table_temp = combined_paper_performance_table.loc[combined_paper_performance_table["id"].isin(value_counts)]
    else:
        combined_paper_performance_table_temp = combined_paper_performance_table.copy()

    # validate using 10 round monte carlo simulation
    results_master_df = pd.DataFrame([], columns= ["Observed Papers", "Dataset", "Selection Method", "NDGC"])
    for rst in trange(10, desc= "CV Validation ({})".format(type)):
        # shuffle time intervals
        new_idx = pd.Series(combined_paper_performance_table_temp["id"].unique()).sample(frac=1, random_state= rst)
        id_shuffle_dict = dict(zip(combined_paper_performance_table_temp["id"].unique(), new_idx.values))
        combined_paper_performance_table_temp["id"] = combined_paper_performance_table_temp["id"].map(id_shuffle_dict)
        # load full evaluation
        for time_interval in time_intervals:
            if not time_interval > combined_paper_performance_table_temp["id"].nunique():
                time_interval_ids = np.sort(combined_paper_performance_table_temp["id"].unique())[:time_interval]
            else:
                raise ValueError("Not enough papers to evaluate full intervall")

            if time_interval not in rankings.keys():
                rankings[time_interval] = {}
            for dataset in datasets:
                if dataset not in rankings[time_interval].keys():
                    rankings[time_interval][dataset] = {}
                # baseline 1: use only papers with relevant metrics (f1 score)
                baseline1 = combined_paper_performance_table_temp.loc[(combined_paper_performance_table_temp["metric"] == metrics_ordered[dataset][0]) & (combined_paper_performance_table_temp["dataset"] == dataset) & (combined_paper_performance_table_temp["id"].isin(time_interval_ids))].groupby(["model", "dataset", "metric"])["score"].mean().reset_index()
                baseline1["rank"] = baseline1.sort_values("score", ascending=False).rank(ascending=False)["score"]
                rankings[time_interval][dataset]["baseline_1"] = baseline1

                # baseline 2: use top 3 algo from each paper
                baseline2 = combined_paper_performance_table_temp.loc[(combined_paper_performance_table_temp["rank"] <= 3) & (combined_paper_performance_table_temp["dataset"] == dataset) & (combined_paper_performance_table_temp["id"].isin(time_interval_ids))].groupby(["model", "dataset"])["score"].count().reset_index()
                baseline2["rank"] = baseline2.sort_values("score", ascending=False).rank(ascending=False)["score"]
                rankings[time_interval][dataset]["baseline_2"] = baseline2

                # test 1: use the elo ranking to select the best models
                cv_eloranking = pd.DataFrame([])
                for fold in range(5):
                    elo = EloSystem(use_mov=True, mov_delta=2, mov_alpha=2)
                    for algorithm in combined_paper_performance_table_temp["model"].unique():
                        elo.add_player(algorithm)
                    for i, performance_df in combined_paper_performance_table_temp.loc[
                        combined_paper_performance_table_temp["id"].isin(time_interval_ids)].groupby("id"):
                        for metric in metrics_ordered[dataset]:
                            if metric in performance_df["metric"].unique():
                                chosen_metric = metric
                                idx_chosen_metric = metrics_ordered[dataset].index(chosen_metric)
                                break
                        temp_df_dict = performance_df.loc[
                            (performance_df["metric"] == chosen_metric) & (performance_df["dataset"] == dataset)][
                            ["model", "score"]].set_index("model")["score"].to_dict()
                        temp_df_dict = itertools.combinations([(key, temp_df_dict[key]) for key in temp_df_dict], r=2)
                        temp_df_dict = [(comp[0][0], comp[1][0], comp[0][0]) if comp[0][1] > comp[1][1] else (
                        comp[0][0], comp[1][0], None) if comp[0][1] >= comp[1][1] else (
                        comp[0][0], comp[1][0], comp[1][0]) for comp in temp_df_dict]

                        for comp in temp_df_dict:
                            elo.record_match(*comp)

                        cv_eloranking = pd.concat((cv_eloranking, pd.DataFrame(elo.get_overall_list())))
                cv_eloranking = cv_eloranking.rename(columns={"player": "model", "elo": "score"})
                cv_eloranking["dataset"] = dataset
                cv_eloranking = cv_eloranking.groupby(["model", "dataset"]).mean().reset_index()
                cv_eloranking["rank"] = cv_eloranking.sort_values("score", ascending=False).rank(ascending=False)["score"]
                rankings[time_interval][dataset]["eloranking_woconf"] = cv_eloranking

                # test 2: use the elo ranking to select the best models
                cv_eloranking = pd.DataFrame([])
                for fold in range(5):
                    elo = EloSystem(use_mov= True, mov_delta= 2, mov_alpha= 2)
                    for algorithm in combined_paper_performance_table_temp["model"].unique():
                        elo.add_player(algorithm)
                    for i, performance_df in combined_paper_performance_table_temp.loc[combined_paper_performance_table_temp["id"].isin(time_interval_ids)].groupby("id"):
                        for metric in metrics_ordered[dataset]:
                            if metric in performance_df["metric"].unique():
                                chosen_metric = metric
                                idx_chosen_metric = metrics_ordered[dataset].index(chosen_metric)
                                break
                        usecase_cofindence = performance_df["useCase_confidence"].value_counts().idxmax()+0.5
                        paper_confidence = performance_df["paper_confidence"].value_counts().idxmax()*2
                        if idx_chosen_metric == 0:
                            usecase_cofindence *= 2
                        elif idx_chosen_metric > int(performance_df["metric"].nunique()/2):
                            usecase_cofindence *= 0.5

                        temp_df_dict= performance_df.loc[(performance_df["metric"] == chosen_metric) & (performance_df["dataset"] == dataset)][["model", "score"]].set_index("model")["score"].to_dict()
                        temp_df_dict = itertools.combinations([(key, temp_df_dict[key]) for key in temp_df_dict], r=2)
                        temp_df_dict = [(comp[0][0], comp[1][0], comp[0][0]) if comp[0][1] > comp[1][1] else (comp[0][0], comp[1][0], None) if comp[0][1] >= comp[1][1] else (comp[0][0], comp[1][0], comp[1][0]) for comp in temp_df_dict]

                        for comp in temp_df_dict:
                            elo.record_match(*comp, mov=(paper_confidence + usecase_cofindence) / 2)

                        cv_eloranking = pd.concat((cv_eloranking, pd.DataFrame(elo.get_overall_list())))
                cv_eloranking = cv_eloranking.rename(columns= {"player": "model", "elo": "score"})
                cv_eloranking["dataset"] = dataset
                cv_eloranking = cv_eloranking.groupby(["model", "dataset"]).mean().reset_index()
                cv_eloranking["rank"] = cv_eloranking.sort_values("score", ascending=False).rank(ascending=False)["score"]
                rankings[time_interval][dataset]["eloranking"] = cv_eloranking

        # gold standard: use the models generated by the fullEvaluation.py to select the best models as our gold standard
        gold_standard = {}
        for dataset in gold_standard_raw_results["dataset"].unique():
            gold_standard[dataset] = pd.DataFrame(gold_standard_raw_results.loc[(gold_standard_raw_results["dataset"] == dataset) & (gold_standard_raw_results["metric"] == metrics_ordered[dataset][0])].groupby(["model", "dataset", "metric"])["score"].max().reset_index(level= [1,2], drop= True))
            gold_standard[dataset]["rank"] = gold_standard[dataset].sort_values("score", ascending=False).rank(ascending=False)["score"]

        # evaluate and plot performance
        results_df = pd.DataFrame([], columns= ["Observed Papers", "Dataset", "Selection Method", "NDGC"])
        for time_interval in time_intervals:
            for dataset in datasets:
                for ranking in rankings[time_interval][dataset].keys():
                    temp_ranking = rankings[time_interval][dataset][ranking]
                    temp_ranking = temp_ranking.join(gold_standard[dataset]["rank"], on= "model", how="right", rsuffix="_gold")
                    temp_ranking["dataset"] = dataset
                    temp_ranking["rank"] = temp_ranking["rank"].fillna(0)
                    ndcg = ndcg_score([temp_ranking["rank_gold"].tolist()], [temp_ranking["rank"].tolist()])
                    results_df = pd.concat((results_df, pd.DataFrame([[time_interval, dataset, ranking, ndcg]], columns= ["Observed Papers", "Dataset", "Selection Method", "NDGC"])))
        results_df = results_df.reset_index(drop=True)

        results_master_df = pd.concat((results_master_df, results_df))

    # plot
    results_master_df_grouped = results_master_df.groupby(["Observed Papers", "Dataset", "Selection Method"])["NDGC"].mean().reset_index()
    selection_methods_dict = {"baseline_1": "Average Performance\nMetric", "baseline_2": "Top 3 Count", "eloranking": "Elo Ranking with Confidence (Ours)", "eloranking_woconf": "Elo Ranking"}
    results_master_df_grouped["Selection Method"] = results_master_df_grouped["Selection Method"].map(selection_methods_dict)
    g = sns.FacetGrid(results_master_df_grouped, col="Dataset", hue= "Selection Method")
    g.map(sns.scatterplot, "Observed Papers", "NDGC")
    g.add_legend()
    #g.map(plt.axhline, y=1, ls='--', color='black')
    plt.ylim(0.8, 0.95)
    if type == "default":
        plt.suptitle("Performance by Ranking Method")
    else:
        plt.suptitle("Performance by Ranking Method ({})".format(type))
    plt.subplots_adjust(top=0.80)
    plt.savefig("./plots/{}_performance_by_ranking_method_scatter_{}_{}.png".format(type, time_intervals[0], time_intervals[-1]))
    #plt.show()

    results_master_df_grouped["Type"] = type
    results = pd.concat((results, results_master_df_grouped))

    # test difference in mean between the methods
    for dataset in datasets:
        for selection_method_baseline in ["baseline_1", "baseline_2", "eloranking_woconf"]:
            rel_ttest = scipy.stats.ttest_rel(results_master_df.loc[(results_master_df["Dataset"] == dataset) & (results_master_df["Selection Method"] == "eloranking"), "NDGC"],
                                  results_master_df.loc[(results_master_df["Dataset"] == dataset) & (results_master_df["Selection Method"] == selection_method_baseline), "NDGC"],
                                  alternative="greater")
            print(f"Dataset: {dataset}, Type: {type}, Selection Method: {selection_methods_dict[selection_method_baseline]}, p-value: {rel_ttest.pvalue}")


results.to_csv("./output/results_{}_{}.csv".format(time_intervals[0], time_intervals[-1]), index= False)

for selection_method in selection_methods_dict.keys():
    g = sns.FacetGrid(results.loc[results["Selection Method"] == selection_methods_dict[selection_method]], col="Dataset", hue="Type")
    g.map(sns.scatterplot, "Observed Papers", "NDGC")
    g.add_legend()
    plt.ylim(0.8, 0.95)
    plt.suptitle("{} Performance Robustness Check".format(selection_methods_dict[selection_method]))
    plt.subplots_adjust(top=0.80)
    plt.savefig("./plots/{}_performance_robustness_check_{}_{}.png".format(selection_method, time_intervals[0], time_intervals[-1]))