python train_binaryDA.py \
    --do-train \
    --lr 5e-5 \
    --eval-every 2000 \
    --batch-size 32 \
    --num_classes 2 \
    --load_distilbert_weights "/media/liyichen/scratch/courses/224n/project/robustqa/save/baseline-01" \
    --run-name SDA
