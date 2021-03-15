PLEASE READ BEFORE YOU PERFORM EDA
===================================================

Baseline: Eval F1: 47.96   EM: 32.20


Baseline + OOD : Eval F1: 49.43, EM: 36.39

NOTE EVERYTIME YOU PERFORM EDA :  Be sure to repeat following steps otherwise the results are inaccurate!!!!!

1. change parameters in xuran_perform_eda.py
2. rm datasets/oodomain_train/_duorc_race_relation_extraction_encodings.pt


The .pt file is created when you finetune on OOD train sets. However, if you don't delete this file, it will keep using old OOD train sets to train the model, instead of the newly augmented OOD sets. So be sure to remove this file everytime you try new parameters, so that our code will have to create new .pt file, based on new EDA parameters. Otherwise, the results are not accurate, because you are not using the right augmented OOD train sets.


===================================================

PARAMETER RECOMMENDATION (Please change them in xuran_perform_eda.py ):

alpha_sr = 0.3
alpha_ri = 0.01
alpha_rs = 0.01
p_rd = 0.2
num_aug = 8 (or 2)

You can also try the below parameters which yields the current best model:

alpha_sr = 0.3
alpha_ri = 0
alpha_rs = 0
p_rd = 0
num_aug = 4  (Note: Based on eda.py, you have to set num_aug = 4 instead of 2, to get 2 augumented sentences per original sentence)

===================================================

Currently the BEST model is obtained by finetune baseline with EDA-enhanced OOD train sets, where alpha_SR = 0.3, n_aug = 2 per original sentences. No RI, RS, RD is applied. 

Path: baseline_finetune_outdomain_eda_SR_0_3-01/checkpoint
Uploaded to: https://drive.google.com/drive/u/2/folders/15TRvz22_Zlj4DFGXiVP1mvAlSYpaxFL7


Eval F1: 51.10, EM: 37.43

===================================================

What files are changed?

1. eda.py  -> Brand new file for EDA
2. xuran_perform_eda.py -> Brand new file for EDA
3. train_finetune_outdomain.py  -> This file is based on the original train.py file with a little modification. 

