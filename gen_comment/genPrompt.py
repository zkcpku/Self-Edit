# %%
import json
import re
from tqdm import tqdm
import sys
inp_path = sys.argv[1]
# inp_path = '/home/zhangkechi/workspace/gentestedit/generation/text002/eval_APPS-firstcase_text002/ground_truth_exec_result.jsonl'
out_path = inp_path.replace(".jsonl", ".faultPrompt.jsonl")
# %%
def read_jsonl(inp_path):
    with open(inp_path, 'r') as f:
        for each in f:
            yield json.loads(each)
def split_line(s):
    return re.split(r"(?<!\\)\n", s)
template = {
    'pass': "Pass the example test case.", 
    'wa': "Wrong Answer. {execution}Rewrite the code.",
    'error': "Line {lineno}, {line_content}, {error_msg}. Fix the bug.",
    "no_lineno": "{error_msg}. Fix the bug.",
    }
def get_prompt(parse_result):
    if parse_result['passed']:
        return template['pass']
    elif "AssertionError: Expected" in parse_result['error']:
        return template['wa'].format(execution=parse_result['error'].replace("AssertionError: ",""))
    # if 'completion_error_line_no' not in parse_result:
    #     import ipdb; ipdb.set_trace()
    elif 'completion_error_line_no' in parse_result and parse_result['completion_error_line_no'] and type(parse_result['completion_error_line_no']) == int and parse_result['completion_error_line_no'] >= 1 and parse_result['completion_error_line']:
        return template['error'].format(lineno=parse_result['completion_error_line_no'], line_content=parse_result['completion_error_line'], error_msg=parse_result['error'])
    else:
        if 'completion_error_line_no' in parse_result and type(parse_result['completion_error_line_no']) == str and parse_result['completion_error_line_no'].isdigit():
            print(parse_result)
            import ipdb; ipdb.set_trace()
        error = parse_result['error']
        if """msg: Traceback (most recent call last):\n  File "/home/zhangkechi/workspace/gentestedit/work/eval/GenEditRefine_for_code/CodeT/src/_execution.py", line 278, in unsafe_execute\n    exec(check_program, exec_globals)\n""" in error:
            error = error.replace("""msg: Traceback (most recent call last):\n  File "/home/zhangkechi/workspace/gentestedit/work/eval/GenEditRefine_for_code/CodeT/src/_execution.py", line 278, in unsafe_execute\n    exec(check_program, exec_globals)\n""","")
        return template['no_lineno'].format(error_msg=parse_result['error'])

# %%

text003_apps = read_jsonl(inp_path)

# %%
with open(out_path, 'w') as f:
    for each_item in tqdm(text003_apps):
        each_item['fault_prompt'] = get_prompt(each_item['parse_result'])
        # dict_keys(['task_id', 'prompt', 'test', 'entry_point', 'completion', 'result', 'passed', 'parse_result', 'fault_prompt'])
        del each_item['test'],  each_item['result'], each_item['passed'], each_item['parse_result']
        f.write(json.dumps(each_item) + '\n')
# %%
# first_text003_apps = next(text003_apps)
# fault_prompt = get_prompt(first_text003_apps['parse_result'])
# first_text003_apps, fault_prompt

# %%
# completions = first_text003_apps['completion']
# completions = split_line(completions)
# print(completions[2-1])

# %%


# %%



