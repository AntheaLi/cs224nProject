python train.py \
    --do-train \
    --num-epochs 10 \
    --eval-every 2000 \
    --batch-size 48 \
    --run-name mrqa \
    --load_weights 'save/mrqa-03' \
    --train-datasets 'race,relation_extraction,duorc' \
    --train-dir '/media/liyichen/scratch/data/nlpdomaindatasets/oodomain_train' \
    --val-dir '/media/liyichen/scratch/data/nlpdomaindatasets/oodomain_val'
