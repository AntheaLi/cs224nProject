# cs224nProject

Baseline: Eval F1: 47.96   EM: 32.20

Baseline + OOD : Eval F1: 49.43, EM: 36.39

Currently the BEST model is obtained by finetune baseline with EDA-enhanced OOD train sets, where alpha_SR = 0.3, n_aug = 2 per original sentences. No RI, RS, RD is applied.

Path: baseline_finetune_outdomain_eda_SR_0_3-01/checkpoint

Uploaded to: https://drive.google.com/drive/u/2/folders/15TRvz22_Zlj4DFGXiVP1mvAlSYpaxFL7

Eval F1: 51.10, EM: 37.43


MRQA: https://github.com/seanie12/mrqa

DAST: https://github.com/cookielee77/DAST

MultinomialDA: https://github.com/ccsasuke/man

pivotDA:https://github.com/eyalbd2/PERL

wasserstein_regularization:https://github.com/RUC-WSM/WD-Match

domain_clustering:https://github.com/roeeaharoni/unsupervised-domain-clusters

domain_tuning:https://github.com/trangvu/mlm4uda

multi_source:https://github.com/copenlu/xformer-multi-source-domain-adaptation


https://arxiv.org/pdf/2003.13003.pdf
https://arxiv.org/pdf/2009.11538.pdf
https://arxiv.org/pdf/2010.06028.pdf
