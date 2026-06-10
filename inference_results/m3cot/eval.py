ALPHA_MAP = ["A", "B", "C", "D", "E", "F"]
import re


def judge_answer(text, choices, answer):
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]

    pattern = re.compile(r'\(([A-Za-z])\)')
    res = pattern.findall(text)
    if len(res) >= 1:
        pred = res[-1].upper()  # 'A', 'B', ...
    else:
        res = []
        for i, choice in enumerate(choices):
            if choice.lower() in text.lower():
                res.append(ALPHA_MAP[i])
        if len(res) >= 1:
            pred = res[-1]
        else:
            for i, choice in enumerate(choices):
                text = re.sub(r'[\n.,!?]', ' ', text)
                if ALPHA_MAP[i] in text.split(" "):
                    res.append(ALPHA_MAP[i])
            if len(res) >= 1:
                pred = res[-1]
            else:
                for i, choice in enumerate(choices):
                    text = re.sub(r'[\n.,!?]', ' ', text)
                    if ALPHA_MAP[i].lower() in text.split(" "):
                        res.append(ALPHA_MAP[i])
                if len(res) >= 1:
                    pred = res[-1]
                else:
                    return "FAILED"

    if pred == answer:
        return True
    else:
        return False

def my_way(answer, pred):
    matches = re.findall(r'[Oo]ption\s+([A-Z])\b', pred)
    # matches = re.findall(r'[Aa]nswer:\s+([A-Z])\b', pred)

    if matches:
        selected = matches[-1]
        if answer == selected:
            return True
        else:
            return False
    return "FAILED"

def my_way1(answer, pred):
    matches = re.findall(r'[Aa]nswer:\s+([A-Z])\b', pred)
    # matches = re.findall(r'[Aa]nswer:\s+([A-Z])\b', pred)


    if matches:
        selected = matches[-1]
        if answer == selected:
            return True
        else:
            return False

    return "FAILED"

def my_way2(answer, pred):
    matches = re.findall(r'[Aa]nswer(.*?)([A-Z])\b', pred)
    if matches:
        selected = matches[-1]
        if answer == selected[-1]:
            return True
        else:
            return False

    return "FAILED"

def my_way3(choices, answer, text):
    res = []
    text = re.sub(r'[\n.,!?()]', ' ', text)
    text = text.split(" ")
    for item in text:
        if item in ALPHA_MAP:
            res.append(item)
    if len(res) >= 1:
        pred = res[-1]
        # if pred == answer:
            # print(text)
            # print(answer)
            # print("++++++++++++++++++++")
        return pred == answer
    else:
        return "FAILED"

def print_acc(c, all, n):
    print(f"acc = {c / float(all) * 100: 0.2f}")
    print(f"no answer: {n}")

def test_file(path):
    if 'icot' not in path:
        path = path.replace("results-sam-wofirst1", "results-sam-wofirst-test")
    c = 0
    all = 0
    n = 0
    with open(path) as f:
        for line in f:
            if all > 2302:
                break
            all += 1
            data = json.loads(line)
            choices = data["choices"]
            answer = data["answer"]
            pred = data['pred']

            flag_1 = my_way(answer, pred)
            flag_2 = my_way1(answer, pred)
            flag_3 = judge_answer(pred, choices, answer)
            flag_4 = my_way2(answer, pred)


            if flag_1 is True or flag_2 is True or flag_3 is True or flag_4 is True:
                c += 1

            if flag_1 == 'FAILED' and flag_2 == "FAILED" and flag_3 == "FAILED" and flag_4 == "FAILED":
                all -= 1
                n += 1

    print_acc(c, all, n)




import json
import re
import numpy as np

path_list_0shot = [

{   "name": "our-0shot-0.0",
        "path": "qwen_mcot_zero0.0.json",
    },
{   "name": "our-0shot-0.1",
        "path": "qwen_mcot_zero0.1.json",
    },
{   "name": "our-0shot-0.2",
        "path": "qwen_mcot_zero0.2.json",
    },
{   "name": "our-0shot-0.3",
        "path": "qwen_mcot_zero0.3.json",
    },
{   "name": "our-0shot-0.4",
        "path": "qwen_mcot_zero0.4.json",
    },
{   "name": "our-0shot-0.5",
        "path": "qwen_mcot_zero0.5.json",
    },
{   "name": "our-0shot-0.6",
        "path": "qwen_mcot_zero0.6.json",
    },
{   "name": "our-0shot-0.7",
        "path": "qwen_mcot_zero0.7.json",
    },
{   "name": "our-0shot-0.8",
        "path": "qwen_mcot_zero0.8.json",
    },
{   "name": "our-0shot-0.9",
        "path": "qwen_mcot_zero0.9.json",
    },
{   "name": "our-0shot-1.0",
        "path": "qwen_mcot_zero1.0.json",
    },
]



path_list = path_list_0shot



for item in path_list:
    try:
        print(f"name: {item['name']}")
        test_file(item['path'].split('/')[-1])
        print("="*200)
    except Exception as e:
        print(e)
        continue
