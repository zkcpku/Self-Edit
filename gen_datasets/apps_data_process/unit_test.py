import json
import multiprocessing
from datasets import load_dataset
from testing_util import run_test,custom_compare_
from utils import check_correctness



output = ['1 3 5 2 4 6', '2 3 1 4', '1 3 5 6 2 4', '1 3 2 4', '1 2 3', '1']
ground_truth = ['1 6 4 2 5 3', '4 2 3 1', '1 4 2 6 5 3', '3 4 2 1', '1 3 2', '1']

print(custom_compare_(output, ground_truth))