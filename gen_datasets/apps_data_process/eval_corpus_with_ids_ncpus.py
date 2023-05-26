import sys
import os
import json
import multiprocessing
from datasets import load_dataset
from testing_util import run_test
from utils import check_correctness
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
import pickle

from concurrent.futures import as_completed, ProcessPoolExecutor
from collections import defaultdict
from tqdm import tqdm

from typing import Dict

DATASET = "codeparrot/apps"

def wrap_for_multiprocess(problem_generations, sample, index, timeout, debug):
    res = []
    # loop over the generations
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res = check_correctness(sample, o, timeout=timeout, debug=debug)
            if debug:
                print(f"\nSuccessful compilation of task {index}!")
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            # if not np.all(curr_res):
            #     if debug:
            #         print(f"Results were not True for all test cases")
        except Exception as e:
            if debug:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
    return dict(
        index=index,
        res=res,
    )

def compute_apps_metrics(generations, apps_dataset, k_list=[1,5], count_errors=True, debug=False,save_path = None):
    def evaluate_generations(generations: dict, apps_dataset, debug: bool = False):
        TIMEOUT = 10
        """We take the list of code generations and try to compile them
        and the run their corresponding unit tests which are retrieved from the APPS dataset.
        Args:
            generations: list of code generations (same order as samples in APPS dataset)
            level: difficulty level used in the generation, can be "all", "introductory", "interview" or "competition"
        Returns:
            results: dictionary of results, key is the problem index, value is a list of results for each generation
            [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
        """

        # generations are code generations in the same order of the dataset
        apps_eval = apps_dataset
        results = {}
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            results_list = []
            for index in generations.keys():
                index = int(index)
                problem_generations = generations[index]
                sample = apps_eval[index]
                futures.append(executor.submit(wrap_for_multiprocess, problem_generations, sample, index, TIMEOUT, debug))
            print(f"Waiting for {len(futures)} processes to finish...")
            pbar = tqdm(total=len(futures))
            for idx, future in enumerate(as_completed(futures)):
                result = future.result()
                results_list.append(result)
                pbar.update(1)
            pbar.close()
            print(f"Done {len(results_list)} processes")
            for result in results_list:
                results[result["index"]] = result["res"]
        # for index in tqdm(generations.keys()):
        #     index = int(index)
        # # for index in tqdm(range(len(generations))):
        #     # code generations for problem (index)
        #     problem_generations = generations[index]
        #     # get corresponding samples from APPS dataset
        #     sample = apps_eval[index]
        #     res = wrap_for_multiprocess(problem_generations, sample, index, timeout=TIMEOUT, debug=debug)
        #     # loop over the generations
        #     # for o_idx, o in enumerate(problem_generations):
        #     #     curr_res = [-2]
        #     #     try:
        #     #         curr_res = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
        #     #         if debug:
        #     #             print(f"\nSuccessful compilation of task {index}!")
        #     #         fixed = []
        #     #         for e in curr_res:
        #     #             if isinstance(e, np.ndarray):
        #     #                 e = e.item(0)
        #     #             if isinstance(e, np.bool_):
        #     #                 e = bool(e)
        #     #             fixed.append(e)
        #     #         curr_res = fixed
        #     #         # if not np.all(curr_res):
        #     #         #     if debug:
        #     #         #         print(f"Results were not True for all test cases")
        #     #     except Exception as e:
        #     #         if debug:
        #     #             print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
        #     #         break
        #     #     finally:
        #     #         assert isinstance(curr_res, list)
        #     #         res.append(curr_res)
        #     results[index] = res
        return results
    def get_results(results: Dict[int, list], count_errors: bool = False, k_list: list = [1, 5]):
        from utils import estimate_pass_at_k
        metrics = {"avg_accuracy": None, "strict_accuracy": None, "pass_at_k": None}
        for k in results:
            for each_e_i in range(len(results[k])):
                if type(results[k][each_e_i]) == list and len(results[k][each_e_i]) > 0 and type(results[k][each_e_i][0]) == str:
                    results[k][each_e_i] = [-2]
        # print(results)
        if len(results[0]) == 1:
            # for single generations we compute average accuracy and stric accuracy: original APPS metrics
            print("Computing accuracy metrics...")
            res = []
            per_prob_res = []
            all_correct = []
            for index in results:
                problem_results = np.asarray(results[index])
                res.extend(problem_results)
                per_prob_res.append(np.mean(problem_results > 0))
                all_correct.append(np.all(problem_results > 0))
            # we count campilation and runtime errors once per pronlem
            compile_errors = len([e for e in res if -2 in e])
            runtime_errors = len([e for e in res if -1 in e])
            total_testcases = len(res)
            if count_errors:
                print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}")
                print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
                print(f"number of problems evaluated = {total_testcases}")

            print(f"Average Accuracy : {np.mean(per_prob_res)}")
            print(f"Strict Accuracy : {np.mean(all_correct)}")
            metrics["avg_accuracy"] = np.mean(per_prob_res)
            metrics["strict_accuracy"] = np.mean(all_correct)

        else:
            # for multiple generations we use pass@k metric used in the HumanEval benchmark
            # we use strict accuracy, a generation is valid if it has to pass all the tests
            print("Computing pass@k metric for multiple generations...")
            # total is list with nb generations per task (task=index)
            # correct is number of generations that passed all tests per task
            total = []
            correct = [] 
            for index in results:
                all_correct = []
                for generation in results[index]:
                    gen = np.array(generation)
                    all_correct.append(np.all(gen>0))
                total.append(len(all_correct))
                correct.append(sum(all_correct))
            total = np.array(total)
            correct = np.array(correct)
            ks = k_list
            pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
            print(pass_at_k)
            metrics["pass_at_k"] = pass_at_k
        return metrics
    results = evaluate_generations(generations, apps_dataset, debug=debug)
    with open(save_path, "w") as f:
        json.dump(results, f)
    metrics = get_results(results, count_errors=count_errors, k_list=k_list)
    return metrics
def list2dict(generations):
    generations_dict = {}
    for i, gen in enumerate(generations):
        generations_dict[i] = gen
    return generations_dict

def eval_ground_truth(split = "train"):
    apps_eval = load_dataset(DATASET, split=split, difficulties=["all"])
    generations = []
    for e in apps_eval:
        try:
            generations.append(eval(e['solutions']))
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
    # generations = [eval(e['solutions']) for e in apps_eval]
    print(min([len(e) for e in generations]))
    print(max([len(e) for e in generations]))
    generations = list2dict(generations)
    # pass_k = 1
    # generations = [e if len(e) >= pass_k else e + e[0] * (pass_k - len(e)) for e in generations]

    metrics = compute_apps_metrics(generations, apps_eval, debug=False, save_path=f"save/ground_truth_train/ground_truth_{split}.json")
    print(metrics)

# def eval_gpt3():
#     apps_eval = load_dataset(DATASET, split="train", difficulties=["all"])
#     gpt3_input_path = '/home/zhangkechi/workspace/pycodegpt-edit/dataset/apps/model_out/GPT3_appstrain_5.json'
#     with open(gpt3_input_path,'r') as f:
#         gpt3_generations = f.readlines()
#     gpt3_generations = [json.loads(e) for e in gpt3_generations]
#     gpt3_generations = gpt3_generations[0]
#     gpt3_generations = [gpt3_generations[str(k)] for k in range(len(gpt3_generations))]

#     generations = gpt3_generations
#     print(min([len(e) for e in generations]))
#     print(max([len(e) for e in generations]))
#     metrics = compute_apps_metrics(generations, apps_eval, debug=False, save_path="save/gpt3_train/gpt3_train.json")
#     print(metrics)

def eval_edit(inp_path, save_path, eval_type = "train"):
    gpt_input_path = inp_path
    with open(gpt_input_path,'r') as f:
        gpt_generations = f.readlines()
    gpt_generations = [json.loads(e) for e in gpt_generations]
    gpt_generations = gpt_generations[0]

    with open(f"full_question_ids_apps_{eval_type}.json",'r') as f:
        question_ids = json.load(f)
    
    print("len of question_ids", len(question_ids))
    print("len of gpt_generations", len(gpt_generations))

    assert len(gpt_generations) % len(question_ids) == 0
    each_len = len(gpt_generations) // len(question_ids)
    gpt_generations_dict = {}
    for i in range(len(question_ids)):
        this_answers = gpt_generations[i*each_len:(i+1)*each_len]
        if type(this_answers[0]) == list:
            this_answers = [e for ee in this_answers for e in ee]
        elif type(this_answers[0]) == str:
            this_answers = this_answers
        else:
            raise Exception("type error")
        gpt_generations_dict[int(question_ids[i])] = this_answers
    generations = gpt_generations_dict
    print("min max num of each question: (should same)")
    print(min([len(e) for e in generations.values()]))
    print(max([len(e) for e in generations.values()]))
    assert min([len(e) for e in generations.values()]) == max([len(e) for e in generations.values()])
    print("=====================================")

    apps_eval = load_dataset(DATASET, split=eval_type, difficulties=["all"])
    metrics = compute_apps_metrics(gpt_generations_dict, apps_eval, debug=False, save_path=save_path)
    print(metrics)

def main():
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--inp_path", type=str, default="/home/zhangkechi/workspace/pycodegpt-edit/save/APPS_pycodegpt_e11_top_p:5/test_no_code-checkpoint-epoch14-767880.skeleton")
    # args_parser.add_argument("--save_path", type=str, default="save/pycodegpt-gold/train_question-checkpoint-epoch10-40293.json")
    args_parser.add_argument("--type", type=str, default="train")
    args = args_parser.parse_args()
    setattr(args, "save_path", args.inp_path.replace("skeleton", "eval_corpus_ncpus.json"))
    eval_edit(args.inp_path, args.save_path, args.type)

if __name__ == "__main__":
    main()
    # eval_ground_truth()
    
        