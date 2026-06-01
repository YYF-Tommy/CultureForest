import json
from typing import Dict, Any, Set, Union
import argparse
import os
import jsonlines
from collections import defaultdict
import numpy as np

def extract(response):
    """Extract the first JSON-like object from model response with redundant text."""
    if not response or not isinstance(response, str):
        return None

    try:
        first_open = response.find('{')
        if first_open == -1:
            return None

        first_close = response.find('}', first_open)
        if first_close == -1 or first_close <= first_open:
            return None

        if "\\" in  response[first_open:first_close + 1]:
            return response[first_open:first_close + 1].replace("\\", "")

        return response[first_open:first_close + 1]
    except:
        return None
    

def get_golden(model, folder, file):
    goldens = []
    with jsonlines.open(f"{gold_path}/{folder}/{file}") as f:
        for line in f:
            goldens.append(line["Easy"]["Answer_Key_Shuffle"])
    return goldens

def evaluate(model, folder, file):
    
    goldens = get_golden(model, folder, file)
    judge_1 = defaultdict(list)
    judge_2 = defaultdict(list)
    judge_3 = defaultdict(list)
    judge_4 = defaultdict(list)
    with jsonlines.open(f"{read_1_path}/{model}/output/{folder}/{file}") as f:
        for line in f:
            if len(line["samples"]["responses"]) == 3:
                judge_1["sample_1"].append(line["samples"]["responses"][0][key])
                judge_1["sample_2"].append(line["samples"]["responses"][1][key])
                judge_1["sample_3"].append(line["samples"]["responses"][2][key])
            elif len(line["samples"]["responses"]) == 2:
                judge_1["sample_1"].append(line["samples"]["responses"][0][key])
                judge_1["sample_2"].append(line["samples"]["responses"][1][key])
                judge_1["sample_3"].append("Num Error")
            elif len(line["samples"]["responses"]) == 1:
                judge_1["sample_1"].append(line["samples"]["responses"][0][key])
                judge_1["sample_2"].append("Num Error")
                judge_1["sample_3"].append("Num Error")
            elif len(line["samples"]["responses"]) == 0:
                judge_1["sample_1"].append("Num Error")
                judge_1["sample_2"].append("Num Error")
                judge_1["sample_3"].append("Num Error")
    with jsonlines.open(f"{read_2_path}/{model}/output/{folder}/{file}") as f:
        for line in f:
            if len(line["samples"]["responses"]) == 3:
                judge_2["sample_1"].append(line["samples"]["responses"][0][key])
                judge_2["sample_2"].append(line["samples"]["responses"][1][key])
                judge_2["sample_3"].append(line["samples"]["responses"][2][key])
            elif len(line["samples"]["responses"]) == 2:
                judge_2["sample_1"].append(line["samples"]["responses"][0][key])
                judge_2["sample_2"].append(line["samples"]["responses"][1][key])
                judge_2["sample_3"].append("Num Error")
            elif len(line["samples"]["responses"]) == 1:
                judge_2["sample_1"].append(line["samples"]["responses"][0][key])
                judge_2["sample_2"].append("Num Error")
                judge_2["sample_3"].append("Num Error")
            elif len(line["samples"]["responses"]) == 0:
                judge_2["sample_1"].append("Num Error")
                judge_2["sample_2"].append("Num Error")
                judge_2["sample_3"].append("Num Error")
    with jsonlines.open(f"{read_3_path}/{model}/output/{folder}/{file}") as f:
        for line in f:
            if len(line["samples"]["responses"]) == 3:
                judge_3["sample_1"].append(line["samples"]["responses"][0][key])
                judge_3["sample_2"].append(line["samples"]["responses"][1][key])
                judge_3["sample_3"].append(line["samples"]["responses"][2][key])
            elif len(line["samples"]["responses"]) == 2:
                judge_3["sample_1"].append(line["samples"]["responses"][0][key])
                judge_3["sample_2"].append(line["samples"]["responses"][1][key])
                judge_3["sample_3"].append("Num Error")
            elif len(line["samples"]["responses"]) == 1:
                judge_3["sample_1"].append(line["samples"]["responses"][0][key])
                judge_3["sample_2"].append("Num Error")
                judge_3["sample_3"].append("Num Error")
            elif len(line["samples"]["responses"]) == 0:
                judge_3["sample_1"].append("Num Error")
                judge_3["sample_2"].append("Num Error")
                judge_3["sample_3"].append("Num Error")
    with jsonlines.open(f"{read_4_path}/{model}/output/{folder}/{file}") as f:
        for line in f:
            if len(line["samples"]["responses"]) == 3:
                judge_4["sample_1"].append(line["samples"]["responses"][0][key])
                judge_4["sample_2"].append(line["samples"]["responses"][1][key])
                judge_4["sample_3"].append(line["samples"]["responses"][2][key])
            elif len(line["samples"]["responses"]) == 2:
                judge_4["sample_1"].append(line["samples"]["responses"][0][key])
                judge_4["sample_2"].append(line["samples"]["responses"][1][key])
                judge_4["sample_3"].append("Num Error")
            elif len(line["samples"]["responses"]) == 1:
                judge_4["sample_1"].append(line["samples"]["responses"][0][key])
                judge_4["sample_2"].append("Num Error")
                judge_4["sample_3"].append("Num Error")
            elif len(line["samples"]["responses"]) == 0:
                judge_4["sample_1"].append("Num Error")
                judge_4["sample_2"].append("Num Error")
                judge_4["sample_3"].append("Num Error")
    # -1: format error
    # 0: wrong
    # 1: correct
    judge = defaultdict(list)
    for i in range(len(goldens)):
        try:
            flag_1 = str(json.loads(extract(judge_1["sample_1"][i]))["Answer"]).lower() == "false"
            flag_2 = str(json.loads(extract(judge_2["sample_1"][i]))["Answer"]).lower() == "false"
            flag_3 = str(json.loads(extract(judge_3["sample_1"][i]))["Answer"]).lower() == "false"
            flag_4 = str(json.loads(extract(judge_4["sample_1"][i]))["Answer"]).lower() == "true"
            # print(flag_1, flag_2, flag_3, flag_4)
            if flag_1 and flag_2 and flag_3 and flag_4:
                judge["sample_1"].append(1)
            else:
                judge["sample_1"].append(0)
        except:
            judge["sample_1"].append(-1)

    for i in range(len(goldens)):
        try:
            # print("222", judge_1["sample_2"][i])
            flag_1 = str(json.loads(extract(judge_1["sample_2"][i]))["Answer"]).lower() == "false"
            flag_2 = str(json.loads(extract(judge_2["sample_2"][i]))["Answer"]).lower() == "false"
            flag_3 = str(json.loads(extract(judge_3["sample_2"][i]))["Answer"]).lower() == "false"
            flag_4 = str(json.loads(extract(judge_4["sample_2"][i]))["Answer"]).lower() == "true"
            if flag_1 and flag_2 and flag_3 and flag_4:
                judge["sample_2"].append(1)
            else:
                judge["sample_2"].append(0)
        except:
            judge["sample_2"].append(-1)


    for i in range(len(goldens)):
        try:
            # print("333", judge_1["sample_3"][i])
            flag_1 = str(json.loads(extract(judge_1["sample_3"][i]))["Answer"]).lower() == "false"
            flag_2 = str(json.loads(extract(judge_2["sample_3"][i]))["Answer"]).lower() == "false"
            flag_3 = str(json.loads(extract(judge_3["sample_3"][i]))["Answer"]).lower() == "false"
            flag_4 = str(json.loads(extract(judge_4["sample_3"][i]))["Answer"]).lower() == "true"
            if flag_1 and flag_2 and flag_3 and flag_4:
                judge["sample_3"].append(1)
            else:
                judge["sample_3"].append(0)
        except:
            judge["sample_3"].append(-1)
    
    mean_value = (judge["sample_1"].count(1) + judge["sample_2"].count(1) + judge["sample_3"].count(1)) / (len(judge["sample_1"]) * 3)
    pass_list = []
    labels = []
    count_label = 0
    for a, b, c in zip(judge["sample_1"], judge["sample_2"], judge["sample_3"]):
        if a == 1 or b == 1 or c == 1:
            pass_list.append(1)
            labels.append(f"{folder}_{file}_{count_label}")
        else:
            pass_list.append(0)
        count_label += 1
    pass_value = sum(pass_list) / len(pass_list)
    stdev_value = np.std([judge["sample_1"].count(1) / len(judge["sample_1"]), judge["sample_2"].count(1) / len(judge["sample_2"]), judge["sample_3"].count(1) / len(judge["sample_3"])])

    return {
        "Mean@3": mean_value,
        "Pass@3": pass_value,
        "Stdev@3": stdev_value,
        "sample_1": judge["sample_1"],
        "sample_2": judge["sample_2"],
        "sample_3": judge["sample_3"]
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

    read_1_path = "XXX/Judge_1"
    read_2_path = "XXX/Judge_2"
    read_3_path = "XXX/Judge_3"
    read_4_path = "XXX/Judge_4"

    gold_path = "XXX/CultureForest" # where you save CultureForest

    if region == 'All':
        countryORregion_folders = os.listdir(f"{read_1_path}/{model}/output")
    else:
        countryORregion_folders = [region]
    print(countryORregion_folders)

    sample_1_all = []
    sample_2_all = []
    sample_3_all = []

    sample_1_country = defaultdict(list)
    sample_2_country = defaultdict(list)
    sample_3_country = defaultdict(list)

    count = 0
    for folder in countryORregion_folders:
        count += 1
        if category == 'All':
            files = os.listdir(f"{read_1_path}/{model}/output/{folder}")
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

    mean_value_all = (sample_1_all.count(1) + sample_2_all.count(1) + sample_3_all.count(1)) / (len(sample_1_all) * 3)
    pass_list_all = []
    for a, b, c in zip(sample_1_all, sample_2_all, sample_3_all):
        if a == 1 or b == 1 or c == 1:
            pass_list_all.append(1)
        else:
            pass_list_all.append(0)
    pass_value_all = sum(pass_list_all) / len(pass_list_all)

    mean_value_country = defaultdict(float)
    for k, v in sample_1_country.items():
        mean_value_country[k] = (sample_1_country[k].count(1) + sample_2_country[k].count(1) + sample_3_country[k].count(1)) / (len(sample_1_country[k]) * 3)

    print("Overall", count)
    print("Mean: " + str(mean_value_all))
    print("Pass: " + str(pass_value_all))
    print("Format Error: " + str((sample_1_all.count(-1) / (sample_1_all.count(-1) + sample_1_all.count(0)) + sample_2_all.count(-1) / (sample_2_all.count(-1) + sample_2_all.count(0)) + sample_3_all.count(-1) / (sample_3_all.count(-1) + sample_3_all.count(0)))/3))
    print("Content Error: " + str((sample_1_all.count(0) / (sample_1_all.count(-1) + sample_1_all.count(0)) + sample_2_all.count(0) / (sample_2_all.count(-1) + sample_2_all.count(0)) + sample_3_all.count(0) / (sample_3_all.count(-1) + sample_3_all.count(0)))/3))
    print("*" * 20)
    print("Disparity:")
    print("Std", round(np.std(list(mean_value_country.values()))*100,2))
    print("Gap", round(max(list(mean_value_country.values()))*100-min(list(mean_value_country.values()))*100, 2))
    print("Max", max(list(mean_value_country.values()))*100, "Min", min(list(mean_value_country.values()))*100)
    print("Mean\tPass\tStd\tGap")
    print(format(mean_value_all*100, ".2f") + "\t" + format(pass_value_all*100, ".2f") + "\t" + format(np.std(list(mean_value_country.values()))*100, ".2f") + "\t" + format(max(list(mean_value_country.values()))*100-min(list(mean_value_country.values()))*100, ".2f"))