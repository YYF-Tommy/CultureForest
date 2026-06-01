import json
from typing import Dict, Any, Set, Union
import argparse
import os
import jsonlines
from collections import defaultdict
import numpy as np
# import scipy.stats as stats
    

def get_golden(model, folder, file):
    goldens = []
    with jsonlines.open(f"{gold_path}/{folder}/{file[:-5]}") as f:
        for line in f:
            goldens.append(line)
    return goldens


def map_to_pm100(score):
    return -1 + 2 * (score - lower) / (upper - lower)


def evaluate(model, folder, file):

    acceptability = []
    goldens = get_golden(model, folder, file)
    for gold in goldens:
        bundle = gold["Norm_Bundle"]["Norms"]
        norm_1_id = bundle["Norm_1"]["norm_id"]
        norm_2_id = bundle["Norm_2"]["norm_id"]
        norm_3_id = bundle["Norm_3"]["norm_id"]
        acceptability.append(
            {
                "norm_1": acceptability_dict[norm_1_id]["acceptability"], 
                "norm_2": acceptability_dict[norm_2_id]["acceptability"], 
                "norm_3": acceptability_dict[norm_3_id]["acceptability"]
            }
        )



    open_results = defaultdict(list)
    with jsonlines.open(f"{read_path}/{model}/{folder}/{file}") as f:
        for line in f:
            open_results["sample_1"].append(line["sample_1"])
            open_results["sample_2"].append(line["sample_2"])
            open_results["sample_3"].append(line["sample_3"])

    
    # -1: format error
    # 0: wrong
    # 1: correct
    open_scores = defaultdict(list)
    for i in range(len(goldens)):
        for norm in ["norm_1", "norm_2", "norm_3"]:
            if acceptability[i][norm] == "Sometimes":
                coeff = coefficient_sometimes
            elif acceptability[i][norm] == "No":
                coeff = coefficient_no
            score = 0
            for k in ["Satisfy", "Neutral", "Violate"]:
                score += open_results["sample_1"][i][norm][k] * coeff[k]
            open_scores["sample_1"].append(map_to_pm100(score))

    for i in range(len(goldens)):
        for norm in ["norm_1", "norm_2", "norm_3"]:
            if acceptability[i][norm] == "Sometimes":
                coeff = coefficient_sometimes
            elif acceptability[i][norm] == "No":
                coeff = coefficient_no
            score = 0
            for k in ["Satisfy", "Neutral", "Violate"]:
                score += open_results["sample_2"][i][norm][k] * coeff[k]
            open_scores["sample_2"].append(map_to_pm100(score))



    for i in range(len(goldens)):
        for norm in ["norm_1", "norm_2", "norm_3"]:
            if acceptability[i][norm] == "Sometimes":
                coeff = coefficient_sometimes
            elif acceptability[i][norm] == "No":
                coeff = coefficient_no
            score = 0
            for k in ["Satisfy", "Neutral", "Violate"]:
                score += open_results["sample_3"][i][norm][k] * coeff[k]
            open_scores["sample_3"].append(map_to_pm100(score))


    mean_value = (sum(open_scores["sample_1"]) + sum(open_scores["sample_2"]) + sum(open_scores["sample_3"])) / (len(open_scores["sample_1"]) + len(open_scores["sample_2"]) + len(open_scores["sample_3"]))
    pass_list = []
    labels = []
    count_label = 0
    for a, b, c in zip(open_scores["sample_1"], open_scores["sample_2"], open_scores["sample_3"]):
        pass_list.append(max(a,b,c))
        labels.append(f"{folder}_{file}_{count_label}")

        count_label += 1
    pass_value = sum(pass_list) / len(pass_list)
    stdev_value = np.std([sum(open_scores["sample_1"]) / len(open_scores["sample_1"]), sum(open_scores["sample_2"]) / len(open_scores["sample_2"]), sum(open_scores["sample_3"]) / len(open_scores["sample_3"])])

    return {
        "Mean@3": mean_value,
        "Pass@3": pass_value,
        "Stdev@3": stdev_value,
        "sample_1": open_scores["sample_1"],
        "sample_2": open_scores["sample_2"],
        "sample_3": open_scores["sample_3"]
    }


# demo
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='test')
    parser.add_argument('-r', '--region', type=str, default='All')
    parser.add_argument('-c', '--category', type=str, default='All')
    args = parser.parse_args()
    model = args.model
    region = args.region
    category = args.category

    d = defaultdict(list)

    key = "content"

    short_1 = set()
    short_2 = set()
    short_3 = set()
    short_4 = set()

    read_path = "XXX/Eval_Open/results"

    gold_path = "XXX/CultureForest"

    if region == 'All':
        countryORregion_folders = os.listdir(f"{read_path}/{model}")
    else:
        countryORregion_folders = [region]
    print(countryORregion_folders)

    upper = 11345.5 / 16134
    lower = - upper

    sample_1_all = []
    sample_2_all = []
    sample_3_all = []

    sample_1_country = defaultdict(list)
    sample_2_country = defaultdict(list)
    sample_3_country = defaultdict(list)

    acceptability_dict = json.load(open("norms_both_acc.json"))

    coefficient_sometimes = {"Satisfy": 0.5, "Neutral": -0.25, "Violate": -0.5}
    coefficient_no = {"Satisfy": 1, "Neutral": -0.5, "Violate": -1}

    count = 0
    for folder in countryORregion_folders:
        count += 1
        if category == 'All':
            files = os.listdir(f"{read_path}/{model}/{folder}")
        else:
            files = [category]
        files = [file for file in files if file.endswith(".json")]
        for file in files:
            print("*" * 20)
            print(folder, file)
            results = evaluate(model, folder, file)

            sample_1_all.extend(results["sample_1"])
            sample_2_all.extend(results["sample_2"])
            sample_3_all.extend(results["sample_3"])

            sample_1_country[folder].extend(results["sample_1"])
            sample_2_country[folder].extend(results["sample_2"])
            sample_3_country[folder].extend(results["sample_3"])
    
    mean_value_all = (sum(sample_1_all) + sum(sample_2_all) + sum(sample_3_all)) / (len(sample_1_all) + len(sample_2_all) + len(sample_3_all))
    pass_list_all = []
    for a, b, c in zip(sample_1_all, sample_2_all, sample_3_all):
        pass_list_all.append(max(a,b,c))
    pass_value_all = sum(pass_list_all) / len(pass_list_all)
    mean_value_country = defaultdict(float)
    for k, v in sample_1_country.items():
        mean_value_country[k] = (sum(sample_1_country[k]) + sum(sample_2_country[k]) + sum(sample_3_country[k])) / (len(sample_1_country[k]) + len(sample_2_country[k]) + len(sample_3_country[k]))

    print("Overall", count)
    print("Mean: " + str(mean_value_all))
    print("Pass: " + str(pass_value_all))
    print("*" * 20)
    print("Disparity:")
    print("Std", round(np.std(list(mean_value_country.values()))*100,2))
    print("Gap", round(max(list(mean_value_country.values()))*100-min(list(mean_value_country.values()))*100, 2))
    print("Max", max(list(mean_value_country.values()))*100, "Min", min(list(mean_value_country.values()))*100)
    print("Mean\tPass\tStd\tGap")
    print(format(mean_value_all*100, ".2f") + "\t" + format(pass_value_all*100, ".2f") + "\t" + format(np.std(list(mean_value_country.values()))*100, ".2f") + "\t" + format(max(list(mean_value_country.values()))*100-min(list(mean_value_country.values()))*100, ".2f"))