python train_multisourceDA.py \
    --do-train \
    --eval-every 2000 \
    --batch-size 32 \
    --num_classes 6 \
    --load_distilbert_weights "/media/liyichen/scratch/courses/224n/project/robustqa/save/baseline-01" \
    --run-name MSDA
