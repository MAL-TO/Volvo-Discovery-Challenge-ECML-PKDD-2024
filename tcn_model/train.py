import argparse
from tcn_model.TcnTrainer import TcnTrainer

def main(args):
    t = TcnTrainer(args)
    if not args.test_only:
        t.train()
    t.test()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ### Training arguments
    parser.add_argument("--test_only", action="store_true", help="Flag to avoid training")
    parser.add_argument("--load_model", type=str, default="", help="Weights model name (stored in 'weights/' directory)")

    parser.add_argument("--batch_size", type=int, default=16, help="Size of the batch for training")

    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Starting learning rate")
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.3, help="Muliply the learning rate by the gamma factor every \{args.lr_cheduler_step\} steps")
    parser.add_argument("--lr_scheduler_step", type=int, default=2, help="Every how many epochs apply the gamma to the learning rate")
    parser.add_argument("--patience_epochs", type=int, default=4, help="After how many epochs of not improving the validation score stop the training")

    parser.add_argument("--disable_cuda", action="store_true", help="Even if cuda is available, dont use it")
    
    parser.add_argument("--train_data_path", action="store_true", default=r"/data1/malto/volvo_ecml_2024/train_gen1.csv", help="absolute path to data file")
    parser.add_argument("--tcnn_weights", action="store_true", default=r"/data1/malto/volvo_ecml_2024/tcnn_weights", help="absolute path to data file")

    # Wandb arguments    

    args = parser.parse_args()
    main(args)