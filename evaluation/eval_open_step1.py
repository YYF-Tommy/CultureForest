import sys
import json
import jsonlines
from collections import defaultdict
import os
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import argparse


@torch.no_grad()
def single_label_logprob_A(llm, tokenizer, prefix_text, label_text):
    full_text = prefix_text + label_text

    enc_full = tokenizer(full_text, return_tensors="pt").to(llm.device)
    enc_pref = tokenizer(prefix_text, return_tensors="pt").to(llm.device)

    out = llm(**enc_full)
    logits = out.logits[0]  # [T, V]

    pref_len = int(enc_pref["attention_mask"].sum().item())
    full_len = int(enc_full["attention_mask"].sum().item())
    lab_len = full_len - pref_len

    logp = torch.zeros((), device=logits.device, dtype=logits.dtype)

    if lab_len <= 0:
        return logp

    # label tokens indices in input_ids: [pref_len, ..., pref_len+lab_len-1]
    # corresponding prediction logits positions: [pref_len-1, ..., pref_len+lab_len-2]
    start = pref_len - 1
    end = pref_len + lab_len - 2

    seg_logits = logits[start:end + 1, :]                     # [lab_len, V]
    target = enc_full["input_ids"][0, pref_len:pref_len + lab_len]  # [lab_len]

    seg_logprobs = F.log_softmax(seg_logits, dim=-1)
    logp = seg_logprobs.gather(1, target.unsqueeze(1)).sum()

    return logp

# @torch.no_grad()
# def batch_label_logprob_A(llm, tokenizer, prefix_texts, label_text):
#     B = len(prefix_texts)
#     full_texts = [p + label_text for p in prefix_texts]

#     enc_full = tokenizer(full_texts, padding=True, return_tensors="pt").to(llm.device)
#     enc_pref = tokenizer(prefix_texts, padding=True, return_tensors="pt").to(llm.device)

#     out = llm(**enc_full)
#     logits = out.logits  # [B, T, V]

#     pref_len = enc_pref["attention_mask"].sum(dim=1)     # [B]
#     full_len = enc_full["attention_mask"].sum(dim=1)     # [B]
#     lab_len = full_len - pref_len                        # [B]

#     logps = torch.zeros(B, dtype=logits.dtype)

#     for i in range(B):
#         pl = int(pref_len[i].item())
#         ll = int(lab_len[i].item())
#         if ll <= 0:
#             continue

#         # label tokens indices in input_ids: [pl, ..., pl+ll-1]
#         # corresponding prediction logits positions: [pl-1, ..., pl+ll-2]
#         start = pl - 1
#         end = pl + ll - 2

#         seg_logits = logits[i, start:end + 1, :]                 # [ll, V]
#         target = enc_full["input_ids"][i, pl:pl + ll]            # [ll]

#         seg_logprobs = F.log_softmax(seg_logits, dim=-1)
#         logps[i] = seg_logprobs.gather(1, target.unsqueeze(1)).sum()

#     return logps
@torch.no_grad()
def batch_label_logprob_A(llm, tokenizer, prefix_texts, label_text):
    outs = []
    for p in prefix_texts:
        outs.append(single_label_logprob_A(llm, tokenizer, p, label_text))
    return torch.stack(outs, dim=0)  # [B]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='test')
    parser.add_argument('-r', '--region', type=str, default='All')
    parser.add_argument('-c', '--category', type=str, default='All')
    args = parser.parse_args()

    model = args.model
    region = args.region
    category = args.category

    verifier = "XXX/C-Verifier" # where you save our C-Verifier
    template = open(
        "CulturalNLI.txt", "r"
    ).read()

    llm = AutoModelForCausalLM.from_pretrained(
        verifier,
        # device_map="auto",
        device_map="balanced",
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
    )
    llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(verifier)

    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_texts = {"Satisfy": "Satisfy", "Neutral": "Neutral", "Violate": "Violate"}
    label_keys = list(label_texts.keys())

    src_path = "XXX/Open" # where you save the inference results
    if region == 'All':
        countryORregion_folders = os.listdir(f"{src_path}/{model}/output")
    else:
        countryORregion_folders = [region]
    countryORregion_folders.sort()

    tgt_output_path = f"XXX/Eval_Open/results/{model}"


    with torch.inference_mode():
        for folder in countryORregion_folders:
            if category == 'All':
                files = os.listdir(f"{src_path}/{model}/output/{folder}")
            else:
                files = [category]
            files = [file for file in files if file.endswith(".json")]
            files.sort()

            for file in files:
                source_data = []
                with jsonlines.open(
                    f"XXX/CultureForest/{folder}/{file}"
                ) as f:
                    for line in f:
                        source_data.append(line)

                save = []
                with jsonlines.open(f"{src_path}/{model}/output/{folder}/{file}") as f:
                    for i, line in tqdm(enumerate(f)):
                        source = source_data[i]
                        bundle = source["Norm_Bundle"]["Norms"]
                        norm1, norm2, norm3 = bundle["Norm_1"], bundle["Norm_2"], bundle["Norm_3"]

                        assert source["uuid"] == line["uuid"]

                        sample_1 = line["samples"]["responses"][0]["content"]
                        sample_2 = line["samples"]["responses"][1]["content"]
                        sample_3 = line["samples"]["responses"][2]["content"]

                        tmp = {"uuid": line["uuid"], "sample_1": {}, "sample_2": {}, "sample_3": {}}
                        k_list = ["sample_1", "sample_2", "sample_3"]
                        l_list = ["norm_1", "norm_2", "norm_3"]

                        prefix_texts = []
                        slots = []  # (sample_key, norm_key)
                        for k, b in enumerate([sample_1, sample_2, sample_3]):
                            key_k = k_list[k]
                            for l, n in enumerate([norm1, norm2, norm3]):
                                key_l = l_list[l]
                                prompt = template.replace("{{context}}", source["Open_End"]) \
                                                 .replace("{{behavior}}", b) \
                                                 .replace("{{norm}}", n["norm"])
                                text = tokenizer.apply_chat_template(
                                    [{"role": "user", "content": prompt}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False
                                )
                                text += "{\"Answer\": \""
                                prefix_texts.append(text)
                                slots.append((key_k, key_l))

                        # 2) “ logprob”
                        logps = []
                        for lk in label_keys:
                            lp = batch_label_logprob_A(llm, tokenizer, prefix_texts, label_texts[lk])  # [9]
                            logps.append(lp)
                        lp_mat = torch.stack(logps, dim=1)  # [9, 3]

                        # 3) softmax 
                        probs = torch.softmax(lp_mat, dim=1)  # [9,3]

                        print(probs)

                        for j, (key_k, key_l) in enumerate(slots):
                            prob_dict = {label_keys[c]: probs[j, c].item() for c in range(3)}
                            tmp[key_k][key_l] = prob_dict

                        save.append(tmp)

                os.makedirs(f"{tgt_output_path}/{folder}", exist_ok=True)
                with jsonlines.open(f"{tgt_output_path}/{folder}/{file}.json", 'w') as wf:
                    for line in save:
                        wf.write(line)


if __name__ == "__main__":
    main()
