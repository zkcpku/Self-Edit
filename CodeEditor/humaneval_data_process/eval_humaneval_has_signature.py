import itertools
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
import numpy as np


from execute import check_correctness
import re
import json
from tqdm import tqdm

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def humaneval_reference(test_func, entry_point):
    return test_func + '\ncheck(' + entry_point + ')'

def get_eval_results(predictions, references, k=[1, 10, 100], num_workers=4, timeout=3.0):
    assert len(predictions) == len(references)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)
        pbar = tqdm(total=len(predictions))
        for task_id, (candidates, test_case) in enumerate(zip(predictions, references)):
            pbar.update(1)
            for candidate in candidates:
                test_program = candidate + "\n" + test_case
                args = (test_program, timeout, task_id, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
        pbar.close()
        pbar = tqdm(total=len(futures))
        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))
            pbar.update(1)
        pbar.close()

    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}

    return pass_at_k, results

def load_humaneval():
    # # load from huggingface
    # import os
    # import sys
    # from datasets import load_dataset
    # humaneval = load_dataset("openai_humaneval")
    # humaneval_dict = humaneval['test'].to_dict()

    # or load from local
    import json
    with open("humaneval_dataset_dict.json",'r') as f:
        humaneval_dict = json.load(f)
    task_ids = humaneval_dict['task_id']
    prompts = humaneval_dict['prompt']
    task_ids = [int(task_id.split('/')[-1]) for task_id in task_ids]
    test_funcs = humaneval_dict['test']
    entry_points = humaneval_dict['entry_point']

    assert len(task_ids) == len(test_funcs) == len(entry_points) == len(prompts), "length not match, len(task_ids)={}, len(test_funcs)={}, len(entry_points)={}, len(prompts)={}".format(len(task_ids), len(test_funcs), len(entry_points), len(prompts))
    
    eval_funcs_dict = {task_ids[i]: humaneval_reference(test_funcs[i], entry_points[i]) for i in range(len(task_ids))}

    prompt_dict = {task_ids[i]: prompts[i] for i in range(len(task_ids))}
    return eval_funcs_dict, prompt_dict



def eval_humaneval(inp_dict):
    """
    inp_dict: {q_id->int 0,1,2... : [candidate, candidate, ...]}
    """
    humaneval_eval_func_dict, prompt_dict = load_humaneval()
    assert len(inp_dict) == len(humaneval_eval_func_dict), "length not match, len(inp_dict)={}, len(humaneval_eval_func_dict)={}".format(len(inp_dict), len(humaneval_eval_func_dict))
    inp_dict_keys = set(list(inp_dict.keys()))
    humaneval_keys = set(list(humaneval_eval_func_dict.keys()))
    assert inp_dict_keys == humaneval_keys, "keys not match"
    humaneval_keys = sorted(list(humaneval_keys))
    
    # predictions = [[prompt_dict[q_id] + e for e in inp_dict[q_id]] for q_id in humaneval_keys]
    predictions = [[e for e in inp_dict[q_id]] for q_id in humaneval_keys]
    references = [humaneval_eval_func_dict[q_id] for q_id in humaneval_keys]

    pass_at_k, results = get_eval_results(predictions, references, k=[1, 10, 100], num_workers=4, timeout=3.0)
    return pass_at_k, results




def test_ground_truth():
    # # load from huggingface
    # import os
    # import sys
    # from datasets import load_dataset
    # humaneval = load_dataset("openai_humaneval")
    # humaneval_dict = humaneval['test'].to_dict()

    # or load from local
    import json
    with open("humaneval_dataset_dict.json",'r') as f:
        humaneval_dict = json.load(f)
    task_ids = humaneval_dict['task_id']
    task_ids = [int(task_id.split('/')[-1]) for task_id in task_ids]
    src_codes = humaneval_dict['canonical_solution']

    inp_dict = {task_ids[i]: [src_codes[i]] for i in range(len(task_ids))}
    inp_dict[0] += ["print('hello world')"]
    inp_dict[0] += ["\tprint('hello world')"]
    inp_dict[0] += ["print('"]
    inp_dict[0] += ["    return True"]
    pass_at_k, results = eval_humaneval(inp_dict)
    print(pass_at_k)

def extract_err_linenum(err_msg):
    # reg = re.compile('.*g_img={url: "(http.*?jpg)"', re.S)
    # reg = re.compile('.*g_img={url: "(http.*?jpg)"', re.S)
    # pattern = 'File "<string>", line 22, in check AssertionError'
    line_num_reg = re.compile('File "<string>", line (\d+), in check AssertionError', re.S)
    line_num = line_num_reg.findall(err_msg)
    if len(line_num) > 0:
        line_num = int(line_num[0])
    else:
        line_num = -1
    return line_num

def compute_humaneval_metrics(skeleton_generations_dict,  debug, save_path):
    pass_at_k, results = eval_humaneval(skeleton_generations_dict)
    output_results = {}
    for i in range(len(results)):
        output_results[str(i)] = []
        this_rst = results[i]
        for each_sample in this_rst:
            this_sample_rst = []
            if each_sample[1]['passed']:
                this_sample_rst = [True]
            else:
                this_msg = each_sample[1]['result'].split("\n")
                start_id = [i for i, e in enumerate(this_msg) if 'File "<string>"' in e]
                each_test_case_msg = " ".join(this_msg)
                if len(start_id) == 0:
                    if "unable to get function error" in each_test_case_msg:
                        this_msg = "no execution code"
                    elif "Error" in this_msg[-1]:
                        this_msg = this_msg[-1]
                    elif "TimeoutException" in this_msg[-1]:
                        this_msg = "Time limit exceeded"
                    elif "timed out" in each_test_case_msg:
                        this_msg = "Time limit exceeded"
                    elif "MemoryError" in each_test_case_msg:
                        this_msg = "Memory limit exceeded"
                    else:
                        # import ipdb; ipdb.set_trace()
                        this_msg = "Time limit exceeded"
                else:
                    start_id = start_id[0]
                    this_msg = this_msg[start_id:]
                    this_msg = " ".join(this_msg)
                    line_num = extract_err_linenum(this_msg)
                    if line_num > 0:
                        try:
                            this_msg += each_sample[1]['input_src'].split('\n')[line_num-1]
                        except Exception as e:
                            print(e)
                            # import ipdb;ipdb.set_trace()
                this_sample_rst = [this_msg]
            output_results[str(i)].append(this_sample_rst)
    
    with open(save_path, "w") as f:
        json.dump(output_results, f)
    # TODO
    # pass_at_k
    return "Done!"

def find_indentation(src_code):
    lines = src_code.split("\n")
    indentation = None
    for line in lines:
        if line.startswith("\t"):
            indentation = "\t"
            break
        elif line.startswith(" "):
            blank_num = len(line) - len(line.lstrip(" "))
            if indentation is None:
                indentation = blank_num * " "
            elif blank_num < len(indentation):
                indentation = blank_num * " "
    return indentation
    
def replace_indent(skeleton_generations):
    INDENT = "    "
    new_skeleton_generations = []
    for each_skeleton in skeleton_generations:
        new_skeleton = []
        for each_answer in each_skeleton:
            old_indent = find_indentation(each_answer)
            if old_indent is not None:
                each_answer = each_answer.replace(old_indent, INDENT)
            # if not each_answer.startswith(INDENT):
            #     each_answer = INDENT + each_answer
            # each_answer = split_func(each_answer)
            new_skeleton.append(each_answer)
        new_skeleton_generations.append(new_skeleton)
    return new_skeleton_generations

def split_func(src_code):
    lines = src_code.split("\n")
    end_line_num = -1
    for i, line in enumerate(lines):
        if line.startswith("def ") and i > 0:
            end_line_num = i
            break
    if end_line_num == -1:
        return "\n".join(lines)
    else:
        return "\n".join(lines[:end_line_num])

def eval_skeleton(inp_path, save_path):
    skeleton_input_path = inp_path
    with open(skeleton_input_path,'r') as f:
        skeleton_generations = f.readlines()
    skeleton_generations = [json.loads(e) for e in skeleton_generations]
    skeleton_generations = skeleton_generations[0]
    skeleton_generations = replace_indent(skeleton_generations)
    # import ipdb; ipdb.set_trace()

    with open(f"full_question_ids_humaneval.json",'r') as f:
        question_ids = json.load(f)
        
    print("len of question_ids", len(question_ids))
    print("len of skeleton_generations", len(skeleton_generations))

    assert len(skeleton_generations) % len(question_ids) == 0
    each_len = len(skeleton_generations) // len(question_ids)
    skeleton_generations_dict = {}
    for i in range(len(question_ids)):
        this_answers = skeleton_generations[i*each_len:(i+1)*each_len]
        if type(this_answers[0]) == list:
            this_answers = [e for ee in this_answers for e in ee]
        elif type(this_answers[0]) == str:
            this_answers = this_answers
        else:
            raise Exception("type error")
        skeleton_generations_dict[int(question_ids[i])] = this_answers
    generations = skeleton_generations_dict
    print("min max num of each question: (should same)")
    print(min([len(e) for e in generations.values()]))
    print(max([len(e) for e in generations.values()]))
    assert min([len(e) for e in generations.values()]) == max([len(e) for e in generations.values()])
    print("=====================================")

    metrics = compute_humaneval_metrics(skeleton_generations_dict, debug=False, save_path=save_path)
    print(metrics)


def main():
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--inp_path", type=str, default="/home/zhangkechi/workspace/apps_metric_edit/otherDataset/humaneval/humaneval_ground_truth.skeleton")
    args = args_parser.parse_args()
    setattr(args, "save_path", args.inp_path.replace("skeleton", "has_signature.humaneval_eval.json"))
    print(args)
    eval_skeleton(args.inp_path, args.save_path)

if __name__ == "__main__":
    main()
