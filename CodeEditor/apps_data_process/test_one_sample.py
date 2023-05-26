import json
import multiprocessing
from datasets import load_dataset
from testing_util import run_test
from utils import check_correctness

DATASET = "codeparrot/apps"

apps_eval = load_dataset(DATASET, split="train", difficulties=["all"])
test_code = '''def
'''
# generation = "\nx = int(input())\n\nl = list(range(x+1))\n\nm = next(l)\n\ns = sum(list([int(i) for i in str(m)]))\n\nif s > sum(list([int(i) for i in str(m)])) :\n\tm = next(l)\n\t\nprint(m)\n"
sample = apps_eval[1999]

print(sample.keys())
print(sample['input_output'])
# print(len(eval(sample['input_output'])['inputs']))

test_code = eval(sample['solutions'])[0]


# print(test_code)

test_rst = run_test(sample, test_code, False)
print("=================")
print(test_rst)

# print(sample.keys())
# print(sample['input_output'])
# print(eval(sample['solutions'])[0])

# test_rst = check_correctness(sample, test_code, timeout=10, debug=False)
# print("======================\n".join(test_rst))
# print("finished")