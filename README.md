# Self-Edit: Fault-Aware Code Editor for Code Generation

Accepted by ACL 2023

Link to the paper: https://arxiv.org/abs/2305.04087

Link to the poster:  [poster.pdf](poster.pdf) 


This is a official implementation of Fault-aware Code Editor

- We include the dataset process tool, the model code, the metric tool here.
- We remove some source code files and config files containing personal information. So be careful! *There might be some small adjustments to the file structure on Github that could cause certain bugs. I will organize it when I have some free time later. If you find any issues, feel free to create an issue or send me an email.*
- We remove the edit dataset because it is too large to upload for review, but we provide the data process method. You can easily follow the steps.

## Benchmark Split

We list the question ids in `benchmark_split/`

We follow *Fault-Aware Neural Code Rankers* (*NeurIPS 2022*)

The authors claims their APPS validation dataset contains 600 tasks in their paper. However, they release a version of dataset with 598 tasks on GitHub. We follow the latter version.

https://github.com/microsoft/CodeRanker

## Data Process and Edit Dataset Construction

After getting the output from LLMs, we pack the outputs as json files, the format is:

```json
// the key number represents the question id, with its value means the list of all outputs for this question
{
	"1": ["xxx","xxx","xxx",...],
	"2": ["xxx","xxx","xxx",...],
	...
}
```

### Get the error message

For APPS dataset:

```bash
python apps_data_process/eval_corpus_with_ids_ncpus_fast.py --inp_path output_path --type train/dev/test
```

For humanEval dataset:

```bash
python humaneval_data_process/eval_humaneval_has_signature_onlyonefunc.py --inp_path output_path
or
python humaneval_data_process/eval_humaneval_only_one_func.py --inp_path output_path
```

- You can replace the data file in the corresponding python file to *get the error_message for **example test cases** or **hidden test cases**.*

### Edit Dataset Construction

The scripts are in `generate_datasets/`

For APPS dataset:

```
python generate_datasets_for_edit_humaneval.py
```

For humanEval dataset:

```
python generate_datasets_for_edit_humaneval.py
```

- You can replace the file path in the corresponding python file to generate edit datasets for different LLMs.



## Editor Model

The scripts are in `editor_model/`

Our GPU platform uses the *hfai* tool to control, so we provide the scripts in *hfai-style*. It is similar to *Pytorch*.

### Train

change the config file in `trainConfig/`, we provide an example script. All scripts will release after review period.

Then run the bash:

```
hfai python multi_trainedit_hfai.py --config_file trainConfig/pycodegpt.train.ngpus.json
```

### Inference

change the config file in `inferConfig/`, we provide an example script. All scripts will release after review period.

Then run the bash:

```
hfai python multi_inferedit_hfai.py --config_file inferConfig/pycodegpt_e11.infer.train.json
```

We also provide a `.sh` file: `hfai_sh.sh`



## Metric Tool

We use the official tool to evaluate the metric.

https://huggingface.co/spaces/codeparrot/apps_metric/tree/main

https://huggingface.co/spaces/evaluate-metric/code_eval/tree/main

For APPS dataset, you can follow the file: `apps_metric/README.md`

For humanEval dataset, you can use the script:

```
python humaneval_data_process/eval_humaneval_has_signature_onlyonefunc.py --inp_path output_path
or
python humaneval_data_process/eval_humaneval_only_one_func.py --inp_path output_path
```

, which we are developed based on the official tool.
