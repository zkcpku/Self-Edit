import sys
import os
import json
from tqdm import tqdm

model_output_path = "TODO"
eval_results_path = "TODO"
save_path = f'gen_datasets/{model_type}/{checkpoint}.error_code_with_msg.jsonl'
save_correct_solutions_path = f'gen_datasets/{model_type}/{checkpoint}.correct_solutions.json'

SAMPLE_NUM = "decided by the first sample"

def not5000_to_5000(train_data, dataset_type):
    with open(f"full_question_ids_apps_{dataset_type}.json",'r') as f:
        full_question_ids = json.load(f)
    length_q_ids = len(full_question_ids)
    assert len(train_data) % length_q_ids == 0
    each_q_num = len(train_data) // length_q_ids
    rtn_train_data = []
    now_train_data_id = 0
    for i in range(5000):
        this_data = []
        if str(i) not in full_question_ids:    
            rtn_train_data.append(this_data)
            continue
        this_data = train_data[now_train_data_id:now_train_data_id+each_q_num]
        now_train_data_id += each_q_num
        if type(this_data[0]) == list:
            this_data = [e for ee in this_data for e in ee]
        rtn_train_data.append(this_data)
    assert len(rtn_train_data) == 5000
    assert now_train_data_id == len(train_data)  
    return rtn_train_data
        
with open(model_output_path,'r') as f:
    train_data = json.load(f)
# print(len(train_data))
if len(train_data) != 5000:
    train_data = not5000_to_5000(train_data, dataset_type=dataset_type)
    # import ipdb; ipdb.set_trace()

with open(eval_results_path,'r') as f:
    pycodegpt_generations = json.load(f)

question_ids = list(pycodegpt_generations.keys())
not_in_5000_ids = [e for e in range(5000) if str(e) not in question_ids]
print("not_in_5000_ids",not_in_5000_ids)
SAMPLE_NUM = len(pycodegpt_generations[question_ids[0]])
print(f"sample num: {SAMPLE_NUM}")
for k in question_ids:
    if len(pycodegpt_generations[k]) != SAMPLE_NUM:
        del pycodegpt_generations[k]
print("len(pycodegpt_generations)",len(pycodegpt_generations))
question_ids = list(pycodegpt_generations.keys())
# print(question_ids)

def extract_error_msg(err_msg):
    rtn_msg = []
    for each_test_case_msg in err_msg:
        if type(each_test_case_msg) == str:
            this_msg = each_test_case_msg.split("\n")
            this_msg = [e for e in this_msg if e != ""]
            start_id = [i for i, e in enumerate(this_msg) if 'File "<string>"' in e]
            if len(start_id) == 0:
                if "unable to get function error" in each_test_case_msg:
                    this_msg = "no execution code"
                elif "Error" in this_msg[-1]:
                    this_msg = this_msg[-1]
                elif "TimeoutException" in this_msg[-1]:
                    this_msg = "Time limit exceeded"
                else:
                    import ipdb; ipdb.set_trace()
            else:
                start_id = start_id[0]
                this_msg = this_msg[start_id:]
                this_msg = " ".join(this_msg)
        elif type(each_test_case_msg) == bool:
            if each_test_case_msg:
                this_msg = ""
            else:
                this_msg = "Wrong answers"
        elif each_test_case_msg == -1:
            this_msg = "Time limit exceeded"
        else:
            raise ValueError
        rtn_msg.append(this_msg)
    return "\n".join(rtn_msg)
# print(extract_error_msg(pycodegpt_generations["0"][2]))


error_code_with_msg = []
correct_solutions = {}
for qid in tqdm(question_ids):
    error_msg = pycodegpt_generations[qid]
    gen_code = train_data[int(qid)]
    assert len(error_msg) == len(gen_code), f"{qid} {len(error_msg)} {len(gen_code)}"
    for i in range(len(error_msg)):
        this_gen_code = gen_code[i]
        try:
            this_error_msg = extract_error_msg(error_msg[i])
            this_error_msg = this_error_msg.strip()
        except Exception as e:
            import ipdb; ipdb.set_trace()
        # 保存正确的样本
        if this_error_msg == "":
            if int(qid) not in correct_solutions:
                correct_solutions[int(qid)] = []
            correct_solutions[int(qid)].append(this_gen_code)
        error_code_with_msg.append(
            {"question_id": int(qid), "error_msg": this_error_msg, "gen_code": this_gen_code}
        )

print("len(error_code_with_msg):", len(error_code_with_msg))
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
with open(save_path,'w') as f:
    for e in error_code_with_msg:
        f.write(json.dumps(e) + "\n")

print("len(correct_solutions): [question num]",len(correct_solutions))
print("sum of correct_solutions: [total answer num]", sum([len(e) for e in correct_solutions.values()]))
with open(save_correct_solutions_path,'w') as f:
    json.dump(correct_solutions,f)