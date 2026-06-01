import sys
import json
import jsonlines
from collections import defaultdict
import os
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, GenerationConfig


import argparse

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--think', action='store_true')  # 默认 False
    parser.add_argument('--gpu', type=int)

    args = parser.parse_args()

    model = args.model
    think_mode = args.think
    gpu = args.gpu

    temperature = 0.7
    top_p = 0.95
    

    sampling_params = SamplingParams(
        n=3,
        temperature=temperature,
        top_p=top_p,
        max_tokens=65536,
    )

    llm = LLM(model, tensor_parallel_size=int(gpu), max_model_len=65536)

    tokenizer = AutoTokenizer.from_pretrained(model)

    src_path = "XXX/CultureForest"
    if think_mode:
        tgt_input_path = f"XXX/Open/{model.split('/')[-1]}/input"
        tgt_output_path = f"XXX/Open/{model.split('/')[-1]}/output"
    else:
        tgt_input_path = f"XXX/Open/{model.split('/')[-1]}_nonThink/input"
        tgt_output_path = f"XXX/Open/{model.split('/')[-1]}_nonThink/output"

    countryORregion_folders = os.listdir(src_path)

    input_path_all = []
    output_path_all = []

    for folder in countryORregion_folders:
        files = os.listdir(f"{src_path}/{folder}")
        for file in files:
            prompts_as_file = []
            input_path = []
            output_path = []
            data = []
            with jsonlines.open(f"{src_path}/{folder}/{file}", 'r') as f:
                for line in f:
                    data.append(line)
            uuid_list = []
            for line in data:
                uuid_list.append(line["uuid"])
                complete_question = line["Open"]["Question_Open"]
                messages = [
                    {"role": "user", "content": complete_question}
                ]
                if not think_mode:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
                    )
                else:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        # enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
                    )
                prompts_as_file.append(text)

            outputs = llm.generate(prompts_as_file, sampling_params)
            # print(outputs)
            save = []
            for i, output in enumerate(outputs):
                tmp = {
                    "uuid": uuid_list[i],
                    "samples": {
                        "responses": []
                    }
                }
                for idx in range(3):
                    if "Qwen3" in model:
                        tmp["samples"]["responses"].append(
                            {
                                "content": output.outputs[idx].text.split("</think>")[-1],
                                "reasoning_content": output.outputs[idx].text,
                                "params": {
                                    "model": model.split("/")[-1],
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "max_tokens": 65536,
                                },
                                "sample_index": idx,
                            }
                        )
                    else:
                        tmp["samples"]["responses"].append(
                            {
                                "content": output.outputs[idx].text,
                                "reasoning_content": "",
                                "params": {
                                    "model": model.split("/")[-1],
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "max_tokens": 65536,
                                },
                                "sample_index": idx,
                            }
                        )
                save.append(tmp)
                os.makedirs(f"{tgt_output_path}/{folder}", exist_ok=True)
                with jsonlines.open(f"{tgt_output_path}/{folder}/{file}", 'w') as f:
                    for line in save:
                        f.write(line)


if __name__ == "__main__":
    main()