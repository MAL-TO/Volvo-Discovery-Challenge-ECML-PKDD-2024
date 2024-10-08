import argparse
from models.TcnTrainer import TcnTrainer

def main(args):
    t = TcnTrainer(args)
    if not args.test_only:
        t.train()
    t.test()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### Training arguments
    parser.add_argument("--seed", type=int, default=13_04_2000, help="Size of the batch for training")
    parser.add_argument("--test_only", action="store_true", help="Flag to avoid training")
    parser.add_argument("--load_model", type=str, default="", help="Weights model name (stored in 'weights/' directory)")

    parser.add_argument("--batch_size", type=int, default=128, help="Size of the batch for training")

    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train the model")
    parser.add_argument("--skip_phase_1", type=bool, default=True, help="Skip or not phase 1 of training")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Starting learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for the optimizer")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.8, help="Muliply the learning rate by the gamma factor every {args.lr_cheduler_step} steps in phase 1")
    parser.add_argument("--lr_scheduler_gamma_2", type=float, default=0.9, help="Muliply the learning rate by the gamma factor every {args.lr_cheduler_step} steps in phase 2")
    parser.add_argument("--lr_scheduler_step", type=int, default=1, help="Every how many epochs apply the gamma to the learning rate in phase 1")
    parser.add_argument("--lr_scheduler_step_2", type=int, default=3, help="Every how many epochs apply the gamma to the learning rate in phase 2")
    parser.add_argument("--patience_epochs", type=int, default=7, help="After how many epochs of not improving the validation score stop the training")

    parser.add_argument("--disable_cuda", action="store_true", help="Even if cuda is available, dont use it")
    
    parser.add_argument("--data_path", default=r"data", help="absolute path to data file")
    parser.add_argument("--train_csv", default=r"train_gen1.csv", help="absolute path to data file")
    parser.add_argument("--test_csv", default=r"public_X_test.csv", help="absolute path to data file")
    parser.add_argument("--variants_csv", default=r"variants.csv", help="absolute path to data file")
    parser.add_argument("--tcnn_weights", default=r"./", help="absolute path to data file")

    args = parser.parse_args()
    main(args)