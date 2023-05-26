import sys
import time
import itertools
from itertools import accumulate, product, permutations, combinations
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache
import math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions
from typing import List, Tuple
import numpy as np
import random
import heapq
from heapq import *
import sys
stdin = sys.stdin
stdout = sys.stdout
def code():
	
	def main():
	    t = int(input())
	    x = list(map(int, input().split()))
	    tower = [0] * (t + 1)
	    for i in range(t):
	        tower[i] = list(map(int, input().split()))
	    for i in range(t):
	        for j in range(t):
	            if tower[j] == 0:
	                tower[j] = tower[j] + 1
	                if tower[j] > 1:
	                    tower[j] = tower[j] - 1
	    if tower[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t[t

if __name__ == '__main__':
    code()