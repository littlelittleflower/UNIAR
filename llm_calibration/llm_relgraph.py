import json
import datetime
import argparse
# import reop
import os
import torch
# from openai import OpenAI
# from llama import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--llm_name", help="yaml configuration file", default='llama-13b', type=str)  # llama-7b llama-13b mistral-7b, gpt-3.5, gpt-4
args, unparsed = parser.parse_known_args()
llm_name = args.llm_name  # "gpt-3.5" #llama-7b llama-13b mistral-7b, gpt-3.5, gpt-4
if llm_name == "llama-13b":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["WORLD_SIZE"] = '1'

openai_organization = "your_openai_organization"


class Chatgpt_Component(object):
    def __init__(self, model_name):
        self.model = OpenAI(organization=openai_organization)
        if model_name == "gpt-3.5":
            self.model_name = "gpt-3.5-turbo"
        elif model_name == "gpt-4":
            self.model_name = "gpt-4-turbo-preview"

    def call(self, messages):
        chunk = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages, )  # stream=True,
        output = chunk.choices[0].message.content
        return output


class Mistral_Component(object):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", trust_remote_code=True)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.half()
        self.device = "cuda"
        self.model = self.model.to(self.device)

    def call(self, messages):
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = model_inputs.to(self.device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        output = decoded[0].split("[/INST]")[-1]
        return output


class Llama_Component(object):
    def __init__(self, root, mode="7b"):
        self.root = root
        self.ckpt_dir = self.root + "llama-2-7b-chat/" if mode == "7b" else self.root + "llama-2-13b-chat/"
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_dir,
            device_map={"": 0},  #
            torch_dtype=torch.float16,
            trust_remote_code=True  #
        )

    def call(self, messages):

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)



        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



def get_taskdesc_bymode(mode):
    if mode == "fixed":
        question_template = """The dictionary rel_dict includes brief information of partial relationships in a knowledge graph. Please analyze the possible entity types of each relationship��s head and tail entities. The candidate entity types are strictly fixed in ("genre/type", "person", "animal", "location/place", "organization", "creative work", "time", "profession", "event", "actual item", "language"). rel_dict = """
    elif mode == "free":
        question_template = """The dictionary rel_dict includes brief information of partial relationships in a knowledge graph. Please analyze the possible entity types of each relationship��s head and tail entities.  rel_dict = """
    elif mode == "refer":
        question_template = """The dictionary rel_dict includes brief information of partial relationships in a knowledge graph. Please analyze the possible entity types of each relationship��s head and tail entities. The candidate entity types are not limited to ("genre/type", "person", "animal", "location/place", "organization", "creative work", "time", "profession", "event", "actual item", "language"). rel_dict = """
    return question_template


def get_input_message_with_description(input_dict, mode):
    example_dict = """{
                       "rel0": {"description": "soccer football team current roster. soccer football roster position position"},
                       "rel1": {"description": "music artist origin"},
                       }"""

    example_output = """Based on the given information, the results are as follows: 
    {
      "rel0": {"head": ["organization"], "tail": ["profession"]},
      "rel1": {"head": ["person"], "tail": ["location/place"]}
    }
    """

    question_template = get_taskdesc_bymode(mode)

    input_template = [
        {"role": "user", "content": question_template + example_dict},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": question_template + input_dict}
    ]
    return input_template


def get_input_message_with_examples(input_dict, mode):
    example_dict = """{
                         "rel0": {"description": "soccer football team current roster. soccer football roster position position", "head entity": "Maldives national football team", "tail entity": "Forward"}, 
                         "rel1": {"description": "music artist origin", "head entity": "Mike Watt", "tail entity": "San Pedro"}
                       }"""

    example_output = """Based on the given information, the results are as follows: 
    {
      "rel0": {"head": ["organization"], "tail": ["profession"]},
      "rel1": {"head": ["person"], "tail": ["location/place"]}
    }
    """

    question_template = get_taskdesc_bymode(mode)

    input_template = [
        {"role": "user", "content": question_template + example_dict},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": question_template + input_dict}
    ]
    return input_template


def get_input_message_with_twoinfo(input_dict, mode):
    example_dict = """ {
                         "rel0": {"head entity": "Maldives national football team", "tail entity": "Forward"}, 
                         "rel1": {"head entity": "Mike Watt", "tail entity": "San Pedro"}
                       }"""

    example_output = """Based on the given information, the results are as follows: 
    
    {
      "rel0": {"head": ["organization"], "tail": ["profession"]},
      "rel1": {"head": ["person"], "tail": ["location/place"]}
    }
    """
    question_template = get_taskdesc_bymode(mode)

    input_template = [
        {"role": "user", "content": question_template + example_dict},
        {"role": "assistant", "content": example_output},
        {"role": "user", "content": question_template + input_dict}
    ]
    return input_template


if llm_name == "llama-7b":
    llm = Llama_Component(root="../projects/llama/")
elif llm_name == "llama-13b":
    llm = Llama_Component(root="../projects/llama/", mode="13b")
elif llm_name == "mistral-7b":
    llm = Mistral_Component()
elif llm_name == "gpt-3.5":
    llm = Chatgpt_Component(model_name=llm_name)
elif llm_name == "gpt-4":
    llm = Chatgpt_Component(model_name=llm_name)

data_path = "../llmoutput/"
# input_modes = ["des", "exp_1", "d&e_1"]
# output_modes = ["fixed", "refer", "free"]
# file_names = ["FB15k237", "Wikidata", "NELL995","eventkg"]
input_modes = ["des"]
output_modes = ["fixed"]
# file_names = ["FB15k237"]
file_names = ["eventkg"]
# file_name = file_names[0]
for input_mode in input_modes:
    for output_mode in output_modes:
        for file_name in file_names:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("start:", file_name + "-r2t.txt")
            rel_text_dict = {}
            with open(data_path + file_name + "-r2t.txt", "r",encoding='utf-8') as f:
                for line in f.readlines():
                    rel, text = line.replace("\n", "").split("\t")
                    rel_text_dict[rel] = text

            print("total rel num:", len(rel_text_dict))
            rel_list = list(rel_text_dict.keys())
            relname2id_dict = {rel: "rel" + str(id) for id, rel in enumerate(rel_list)}
            relid2name_dict = {v: k for k, v in relname2id_dict.items()}
            # rel_example_dict = json.load(open(data_path + file_name + "-r2example.json", "r", encoding="utf8"))

            start, call_size = 0, 5
            total_results = {}
            while start < len(rel_list):
                end = start + call_size
                try:
                    if input_mode == "des":
                        sub_data = {relname2id_dict[k]:    [k] for k in rel_list[start:end]}
                        input_dict = json.dumps(sub_data)
                        print(input_mode, ">>>>>>>>>>>>input:", input_dict)
                        messages = get_input_message_with_description(input_dict, output_mode)
                    elif "exp" in input_mode:  # exp_1
                        example_num = int(input_mode.split("_")[-1])
                        sub_data = {}
                        for k in rel_list[start:end]:
                            rid = relname2id_dict[k]
                            head_texts = [item["head_name"][0] if len(item["head_name"]) == 1 else item["head_name"] for
                                          item in rel_example_dict[k][:example_num]]
                            tail_texts = [item["tail_name"][0] if len(item["tail_name"]) == 1 else item["tail_name"] for
                                          item in rel_example_dict[k][:example_num]]
                            if len(head_texts) == 1: head_texts = head_texts[0]
                            if len(tail_texts) == 1: tail_texts = tail_texts[0]
                            sub_data[rid] = {"head entity": head_texts, "tail entity": tail_texts}
                        input_dict = json.dumps(sub_data)
                        print(input_mode, ">>>>>>>>>>>>input:", rel_list[start:end], input_dict)
                        messages = get_input_message_with_examples(input_dict, output_mode)
                    elif "d&e" in input_mode:  # exp_1
                        example_num = int(input_mode.split("_")[-1])
                        sub_data = {}
                        for k in rel_list[start:end]:
                            rid = relname2id_dict[k]
                            rtext = rel_text_dict[k]
                            head_texts = [item["head_name"][0] if len(item["head_name"]) == 1 else item["head_name"] for
                                          item in rel_example_dict[k][:example_num]]
                            tail_texts = [item["tail_name"][0] if len(item["tail_name"]) == 1 else item["tail_name"] for
                                          item in rel_example_dict[k][:example_num]]
                            if len(head_texts) == 1: head_texts = head_texts[0]
                            if len(tail_texts) == 1: tail_texts = tail_texts[0]
                            sub_data[rid] = {"description": rtext, "head entity": head_texts, "tail entity": tail_texts}
                        input_dict = json.dumps(sub_data)
                        print(input_mode, ">>>>>>>>>>>>input:", rel_list[start:end], input_dict)
                        messages = get_input_message_with_twoinfo(input_dict, output_mode)
                    output = llm.call(messages)
                    dict_matches = re.findall(r'"(\w+)":\s*{([^}]+)}', output)
                    for k, v in dict_matches:
                        key = relid2name_dict[k]
                        v = v.replace("_", " ")
                        # print(key, v)
                        value = json.loads("{" + v + "}", strict=False)
                        # print(key, value, type(value))
                        total_results[key] = value
                    print(len(total_results), "/", len(rel_text_dict))
                except Exception as e:
                    if hasattr(e, 'message'):
                        print(e.message)
                    else:
                        print(e)
                    print("xxxxxxxxxxxxxxxxxx", file_name, start, end, input_dict)
                start = end
            startTimeSpan = datetime.datetime.now().strftime("%m%d%H%M%S")
            save_file = data_path + file_name + "-r2t-" + llm_name.replace("-",
                                                                           "") + "-" + input_mode + "-" + output_mode + ".json"  # +"-"+startTimeSpan
            json.dump(total_results, open(save_file, "w", encoding='utf-8'))
            try:
                with open(save_file, 'r') as f:
                    rel_dict = json.load(f)
                    print(save_file, rel_dict)
            except Exception as e:
                if hasattr(e, 'message'):
                    print(e.message)
