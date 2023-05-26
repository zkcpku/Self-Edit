# hfai workspace diff --no_checksum
hfai workspace push --no_checksum --force
# hfai workspace remove $workspace_name -f $path_to_delete
# hfai workspace download save/infer_ngpu/ --no_checksum

# # train
# hfai python multi_trainedit_hfai.py --config_file trainConfig/pycodegpt.train.ngpus.json -- --name pycodegpt.train.ngpus.json -n 1 -p 10 --no_checksum --force > ../logs/code_edit/pycodegpt.train.ngpus.json.log 2>&1 &


# # inference
# hfai python multi_inferedit_hfai.py --config_file inferConfig/pycodegpt_e11.infer.train.json -- --name pycodegpt_e11.infer.train.json_ngpu -n 1 -p 10 --no_checksum --force > ../logs/code_edit/pycodegpt_e11.infer.train.json.log 2>&1 &