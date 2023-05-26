import json
with open("humaneval_dataset_dict.json",'r') as f:
        humaneval_dict = json.load(f)
task_ids = humaneval_dict['task_id']
task_ids = [int(task_id.split('/')[-1]) for task_id in task_ids]

task_ids = [str(e) for e in task_ids]

print(task_ids)

with open("full_question_ids_humaneval.json","w") as f:
    json.dump(task_ids, f)