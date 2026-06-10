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
        if ALPHA_MAP[int(answer)] == selected:
            return True
        else:
            return False
    return "FAILED"

def my_way1(answer, pred):
    matches = re.findall(r'[Aa]nswer:\s+([A-Z])\b', pred)
    # matches = re.findall(r'[Aa]nswer:\s+([A-Z])\b', pred)


    if matches:
        selected = matches[-1]
        if ALPHA_MAP[int(answer)] == selected:
            return True
        else:
            return False

    return "FAILED"

def my_way2(answer, pred):
    matches = re.findall(r'[Aa]nswer(.*?)([A-Z])\b', pred)
    if matches:
        selected = matches[-1]
        if ALPHA_MAP[int(answer)] == selected[-1]:
            return True
        else:
            return False

    return "FAILED"

def my_way3(choices, answer, text):
    res = []
    for i, choice in enumerate(choices):
        text = re.sub(r'[\n.,!?]', ' ', text)
        if ALPHA_MAP[i].lower() in text.split(" "):
            res.append(ALPHA_MAP[i])
    if len(res) >= 1:
        pred = res[-1]
        return pred == answer
    else:
        return "FAILED"


def print_acc(c, all, n):
    print(f"acc: {c} / {all} = {c / float(all) * 100: 0.2f}")
    print(f"no answer: {n}")

def test_file(path):
    c = 0
    all = 0
    n = 0
    with open(path) as f:
        for line in f:
            # if all > 1233:
            #     break
            all += 1
            data = json.loads(line)
            choices = data["choices"]
            answer = data["answer"]
            pred = data["pred"]

            flag_1 = my_way(answer, pred)
            flag_2 = my_way1(answer, pred)
            flag_3 = judge_answer(pred, choices, answer)
            flag_4 = my_way2(answer, pred)
            flag_5 = my_way3(choices, ALPHA_MAP[answer], pred)

            if flag_1 is True or flag_2 is True or flag_3 is True or flag_4 is True :
            # if flag_3 is True:

                c += 1

            if flag_1 == 'FAILED' and flag_2 == "FAILED" and flag_3 == "FAILED" and flag_4 == "FAILED":
                all -= 1
                n += 1
            # flag = my_way(answer, pred)
            # if flag == 'FAILED':
            #     flag1 = judge_answer(pred, choices, answer)
            #     if flag1 is True:
            #         c += 1
            #     elif flag == "FAILED":
            #         n += 1
            # elif flag is True:
            #     c += 1
    print_acc(c, all, n)




import json
import re

path_list = [

    {"name": "our-zero-shot-0.2",
     "path": "qwen_mcot_zero0.2.json",
     },

]



for item in path_list:
    try:
        print(f"name: {item['name']}")
        test_file(item['path'].split('/')[-1])
        print("="*200)
    except Exception as e:
        print(e)
        continue
