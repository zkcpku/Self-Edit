import json
import multiprocessing
from datasets import load_dataset
from testing_util import run_test
from utils import check_correctness
from tqdm import tqdm

# def pool_evaluate_apps_modelout(inp_path, out_path, eval_type='train'):
#     DATASET = "codeparrot/apps"
#     apps_eval = load_dataset(DATASET, split=eval_type, difficulties=["all"])
#     APPS_rst = []
#     print("inp: ", inp_path)
#     print("out: ",out_path)
#     print("eval_type: ",eval_type)
#     with open(inp_path,'r') as f:
#         gpt3_output = json.load(f)
#     print(len(gpt3_output))
#     with open(APPS_PATH[eval_type],'r') as f:
#         items = f.readlines()
#     print(len(items))
#     items = [json.loads(e) for e in items]
#     input_items = []
#     with ProcessPoolExecutor() as executor:
#         futures = []
#         results_list = []
#         for item in items:
#             question_id = item['question_id']
#             src_code = item['src_code']
#             test_cases = item['test_cases']
#             future = executor.submit(check_all_test_cases, question_id, src_code, test_cases)
#             futures.append(future)

#         print(f'{len(futures)} execution requests are submitted')

#         pbar = tqdm(total=len(futures))
#         for idx, future in enumerate(as_completed(futures)):
#             # print('[{}/{}] execution completed'.format(idx+1, len(futures)))
#             result = future.result()
#             results_list.append(result)
#             pbar.update(1)
#         pbar.close()

#         for ind in tqdm(range(len(items))):
#             each_item = items[ind]
#             question_id = each_item['id']
#             sample = apps_eval[int(question_id)]
#             # src_code = eval(each_item['solutions'])[-1]
#             src_codes = gpt3_output[str(question_id)]
#             for src_code in src_codes:
#                 correct, test_rst = test_one(sample, src_code)
#                 APPS_rst.append(dict(
#                     question_id = question_id,
#                     src_code = src_code,
#                     results = test_rst,
#                     correct = correct
#                 ))
#     with open(out_path,'w') as f:
#         json.dump(APPS_rst, f)


def evaluate_apps_modelout(inp_path, out_path, eval_type='train'):
    DATASET = "codeparrot/apps"
    apps_eval = load_dataset(DATASET, split=eval_type, difficulties=["all"])
    APPS_rst = []
    print("inp: ", inp_path)
    print("out: ",out_path)
    print("eval_type: ",eval_type)
    with open(inp_path,'r') as f:
        gpt3_output = json.load(f)
    print(len(gpt3_output))
    # with open(APPS_PATH[eval_type],'r') as f:
    #     items = f.readlines()
    # print(len(items))
    # items = [json.loads(e) for e in items]
    input_items = []
    for ind in tqdm(range(5000)):
        question_id = ind
        sample = apps_eval[int(question_id)]
        # src_code = eval(each_item['solutions'])[-1]
        src_codes = gpt3_output[str(question_id)]
        for src_code in src_codes:
            correct, test_rst = test_one(sample, src_code)
            APPS_rst.append(dict(
                question_id = question_id,
                src_code = src_code,
                results = test_rst,
                correct = correct
            ))
    with open(out_path,'w') as f:
        json.dump(APPS_rst, f)

def test_one(sample, src_code):
    test_rst = run_test(sample, src_code, False)
    correct = False
    for e in test_rst:
        if type(e) != bool or e == False:
            correct = False
            break
    return correct, test_rst


if __name__ == '__main__':
    # unit_test_check_all_test_cases_call_func()
    gpt3_input_path = '/home/zhangkechi/workspace/pycodegpt-edit/dataset/apps/model_out/GPT3_appstrain_5.json'
    save_gpt3 = "eval_apps/GPT3_appstrain_5_evalrst.json"

    ground_truth_path = '/home/zhangkechi/workspace/pycodegpt-edit/dataset/apps/model_out/groundtruth_appstrain.json'
    save_ground_truth = "eval_apps/groundtruth_appstrain_evalrst.json"

    # inp_path, out_path = gpt3_input_path, save_gpt3
    inp_path, out_path = ground_truth_path, save_ground_truth
    
    evaluate_apps_modelout(inp_path, out_path)