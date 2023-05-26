from collections import Counter
import json

import sys
eval_path = sys.argv[1]
# eval_path = '/home/zhangkechi/workspace/apps_metric_edit/save/ground_truth_train/ground_truth_train.json'

# eval_path = "/home/zhangkechi/workspace/apps_metric_edit/save/incoder-1b/train_question-incoder-1b.skeleton.json"

# eval_path = "/home/zhangkechi/workspace/apps_metric_edit/save/ground_truth_train/ground_truth_train.json"

# eval_path = "/home/zhangkechi/workspace/apps_metric_edit/save/pycodegpt-gold/test_question-checkpoint-epoch11-43956.skeleton.json"
# eval_path = "/home/zhangkechi/workspace/apps_metric_edit/save/[nouse]gpt3_train/gpt3_train.json"

# pycodegpt-gold on train
# eval_path = "/home/zhangkechi/workspace/apps_metric_edit/save/pycodegpt-gold/train_question-checkpoint-epoch11-43956.skeleton.json"
# eval_path = "/home/zhangkechi/workspace/pycodegpt-edit/save/APPS_pycodegpt_e11_top_p:5/test_no_code-checkpoint-epoch14-767880.eval_corpus_ncpus_fast.json"
# pycodegpt-gold on test
# eval_path = "/home/zhangkechi/workspace/apps_metric_edit/save/pycodegpt-gold/test_question-checkpoint-epoch11-43956.skeleton.json"
# eval_path ="/home/zhangkechi/workspace/pycodegpt-edit/save/APPS_pycodegpt_e11_top_p:5_onDatasetFortest/test_no_code-checkpoint-epoch14-767880.eval_corpus_ncpus_fast.json"
# #   eval_path = "/home/zhangkechi/workspace/pycodegpt-edit/accelerate/pycodegpt_output/pycodegptE11_e1420221022-223623_infer_test_eval_results.eval_corpus_ncpus_fast.json"
# #  eval_path = "/home/zhangkechi/workspace/pycodegpt-edit/save/APPS_pycodegpt_e11_greedy_onDatasetFortest/test_no_code-checkpoint-epoch14-767880.eval_corpus.json"


# gpt3 on train
# eval_path = "/home/zhangkechi/workspace/apps_metric_edit/model_output/gpt3/GPT3_appstrain_sample10_5000.eval_corpus_ncpus_fast.json"

# gpt3 on test
# eval_path = "/home/zhangkechi/workspace/apps_metric_edit/model_output/gpt3/GPT3_appstest_sample5_5000.eval_corpus_ncpus_fast.json"


# eval_path = "/home/zhangkechi/workspace/pycodegpt-edit/save-code_accelerate/APPS_pycodegpt_e11_checkpoint-epoch14-767880_top_p:5_onDatasetFortrain/test_no_code-checkpoint-epoch14-767880.eval_corpus_ncpus_fast.json"

# SAMPLE_NUM = 50 # 100
SAMPLE_NUM = "decided by the first sample"


with open(eval_path,'r') as f:
    train_data = f.readlines()
train_data = [json.loads(e) for e in train_data]
train_data = train_data[0]


train_data_keys = list(train_data.keys())
print("Sample_num may be:",len(train_data[train_data_keys[0]]),"; Be sure to set it.")
SAMPLE_NUM = len(train_data[train_data_keys[0]])
print("Sample_num is:",SAMPLE_NUM)
for k in train_data_keys:
    if len(train_data[k]) != SAMPLE_NUM:
        del train_data[k]
print(len(train_data))



def extract_passed(error_msgs):
    passed = []
    for e in error_msgs:
        if type(e) == str:
            passed.append(False)
        elif type(e) == bool:
            passed.append(e)
        elif e == -1:
            passed.append(False)
        else:
            # import ipdb; ipdb.set_trace()
            # passed.append(False)
            raise ValueError
    return all(passed)


def pass_at_k(eval_data, ks = [1,5,10,50,100]):
    eval_data_is_passed = {k:[extract_passed(e) for e in eval_data[k]] for k in eval_data}
    for k in ks:
        passed_examples = [sum(e[:k]) for e in eval_data_is_passed.values()]
        print(f"Pass@{k}: ", len([e for e in passed_examples if e > 0])/len(passed_examples), "\tCorrect Solutions:", sum(passed_examples))
        # print(len(passed_examples))
        # print(passed_examples)
pass_at_k(train_data)


