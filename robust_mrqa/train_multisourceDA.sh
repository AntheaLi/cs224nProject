python train_multisourceDA.py \
    --do-train \
    --eval-every 2000 \
    --lr 1e-4 \
    --batch-size 32 \
    --num_classes 6 \
    --load_distilbert_weights "/home/hehe/cs224nProject/" \
    --run-name MSDA
