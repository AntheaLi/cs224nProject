python train_multisourceDA.py \
    --do-train \
    --eval-every 200 \
    --batch-size 32 \
    --num_classes 6 \
    --lr 1e-4 \
    --load_distilbert_weights "/media/liyichen/scratch/courses/224n/project/robustqa/save/baseline-01" \
    --run-name MSDA
