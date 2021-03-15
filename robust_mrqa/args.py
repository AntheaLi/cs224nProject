import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--recompute-features', action='store_true')
    parser.add_argument('--train-dir', type=str, default='/media/liyichen/scratch/data/nlpdomaindatasets/indomain_train')
    parser.add_argument('--val-dir', type=str, default='/media/liyichen/scratch/data/nlpdomaindatasets/indomain_val')
    parser.add_argument('--eval-dir', type=str, default='/media/liyichen/scratch/data/nlpdomaindatasets/oodomain_val')
    parser.add_argument('--eval-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--load_weights', type=str, default='')
    parser.add_argument('--load_distilbert_weights', type=str, default='')
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument("--dis_lambda", default=0.01, type=float, help="Importance of adversarial loss")
    parser.add_argument("--hidden_size", default=768, type=int, help="Hidden size for discriminator")
    parser.add_argument("--num_layers", default=3, type=int, help="Number of layers for discriminator")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for discriminator")
    parser.add_argument("--anneal", action="store_true")
    parser.add_argument("--concat", action="store_true", help="Whether to use both cls and sep embedding")
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--source-train-datasets', type=str, default='squad,nat_questions,newsqa')
    parser.add_argument('--target-train-datasets', type=str, default='race,relation_extraction,duorc')
    parser.add_argument('--source-train-dir', type=str,default='/media/liyichen/scratch/data/nlpdomaindatasets/indomain_train')
    parser.add_argument('--target-train-dir', type=str, default='/media/liyichen/scratch/data/nlpdomaindatasets/oodomain_train')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--wasserstein', action='store_true')

    args = parser.parse_args()
    return args
