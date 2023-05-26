import dataset
import pickle
with open('/home/zhangkechi/workspace/pycodegpt/save/APPS/train_blocksize_2048_wordsize_1_rank_0','rb') as f:
    data = pickle.load(f)
print(len(data))

inputs_len = [len([e for e in data['inputs'][i] if e != 32004]) for i in range(len(data['inputs']))]
# token_labels_len = [len([e for e in data['token_labels'][i] if e != 32004]) for i in range(len(data['token_labels']))]

total_len = inputs_len

# len([e for e in inputs_len if e<=1024])/len(inputs_len)
# 0.9298996860925345

# import ipdb; ipdb.set_trace()