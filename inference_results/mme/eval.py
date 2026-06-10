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


def print_acc(c, all, n):
    print(f"acc: {c} / {all} = {c / float(all) * 100: 0.2f}")
    print(f"no answer: {n}")

def test_file(path):
    c = 0
    all = 0
    n = 0
    with open(path) as f:
        for line in f:
            all += 1
            if all > 1064:
                break
            data = json.loads(line)
            choices = data["choices"]
            answer = data["answer"]
            pred = data["pred"]

            flag_1 = my_way(answer, pred)
            flag_2 = my_way1(answer, pred)
            flag_3 = judge_answer(pred, choices, answer)
            flag_4 = my_way2(answer, pred)

            if 'direct' in path:
                if flag_2 is True or flag_4 is True:
                    c += 1
            else:
                if flag_1 is True or flag_2 is True or flag_3 is True or flag_4 is True:
                    c += 1

            if flag_1 == 'FAILED' and flag_2 == "FAILED" and flag_3 == "FAILED" and flag_4 == "FAILED":
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


import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='./LaVIN', type=str)

eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork",
                   "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

        return

    import re

    def parse_pred_ans(self, pred_ans_raw: str) -> str:
        if not pred_ans_raw or not isinstance(pred_ans_raw, str):
            return "other"

        # --- Method 1: Parse after the last "Answer:" ---
        # Find the starting index of the last "Answer:" (case-insensitive)
        last_answer_marker_idx = -1
        # Search for "answer:" from the end of the string
        # We use re.finditer to find all occurrences and then take the last one,
        # or pred_ans_raw.lower().rfind("answer:") for a simpler approach.
        # Let's use rfind for simplicity for the marker itself.

        # Normalize pred_ans_raw to lowercase for searching the marker "answer:"
        pred_ans_lower = pred_ans_raw.lower()
        marker = "answer:"
        idx = pred_ans_lower.rfind(marker)

        if idx != -1:
            # Extract the part after "answer:" from the original string to preserve case if needed, though we lowercase next
            text_after_marker_raw = pred_ans_raw[idx + len(marker):]

            # Try to get the first word after "Answer:"
            # Remove leading/trailing whitespace, then get the first sequence of letters
            match = re.match(r"^\s*([a-zA-Z]+)", text_after_marker_raw)
            if match:
                first_word_after_marker = match.group(1).lower()
                if "yes" in first_word_after_marker:
                    return "yes"
                if "no" in first_word_after_marker:
                    return "no"
            # If "Answer:" was found but the word after it wasn't "yes" or "no",
            # we fall through to Method 2.

        # --- Method 2: Find the last standalone "yes" or "no" in the entire string ---
        # Find all occurrences of "yes" or "no" as whole words, case-insensitive

        # Store (position, label) for all "yes" and "no" occurrences
        found_options = []
        pred_ans_raw = pred_ans_lower


        # Find "yes"
        for m_yes in re.finditer(r"\b(yes)\b", pred_ans_raw, re.IGNORECASE):
            found_options.append({'label': 'yes', 'pos': m_yes.start()})

        # Find "no"
        for m_no in re.finditer(r"\b(no)\b", pred_ans_raw, re.IGNORECASE):
            found_options.append({'label': 'no', 'pos': m_no.start()})

        if found_options:
            # Sort by position to find the last one
            found_options.sort(key=lambda x: x['pos'])
            return found_options[0]['label']  # This will be 'yes' or 'no' in lowercase

        # --- Default ---
        # If neither method yielded "yes" or "no"
        return "other"



    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }

        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds)

        clean_gts = []
        clean_preds = []
        other_num = 0
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict

    def process_result(self, results_dir):

        model_score_dict = dict()
        all_score = 0
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")

            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:

                task_txt = os.path.join(results_dir, task_name + ".txt")

                lines = open(task_txt, 'r', encoding='utf-8').readlines()
                chunk_lines = list(self.divide_chunks(lines))  # one image corresponds to two questions

                img_num = len(chunk_lines)
                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                for img_items in chunk_lines:
                    try:
                        assert len(img_items) == 2
                        img_correct_num = 0

                        for img_item in img_items:
                            img_name, question, gt_ans, pred_ans = img_item.split("\t")

                            gt_ans = gt_ans.lower()
                            pred_ans = pred_ans.lower()

                            assert gt_ans in ["yes", "no"]  # gt can only be yes or no.

                            pred_ans = self.parse_pred_ans(pred_ans)
                            assert pred_ans in ["yes", "no", "other"]

                            gts.append(gt_ans)
                            preds.append(pred_ans)

                            if gt_ans == pred_ans:
                                img_correct_num += 1

                            if pred_ans not in ["yes", "no"]:
                                task_other_ans_num += 1


                        if img_correct_num == 2:
                            acc_plus_correct_num += 1


                    except Exception as e:
                        print(e)

                # cal TP precision acc, etc.
                metric_dict = self.compute_metric(gts, preds)
                acc_plus = acc_plus_correct_num / img_num
                metric_dict["acc_plus"] = acc_plus

                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v * 100

                task_score_dict[task_name] = task_score

                scores += task_score


            print(f"total score: {scores:.2f}\n")
            all_score += scores
            for task_name, score in task_score_dict.items():
                print("\t", task_name, " score:", score)
            print("\n")

        print(f"all score: {all_score:.2f}")

        return


path = [
    'Your_Results/qwen_mcot_zero0',
]

if __name__ == "__main__":
    for results_dir in path:
        cal = calculate_metrics()
        cal.process_result(results_dir)

import json
import re
