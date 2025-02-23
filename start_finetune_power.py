import torch
from torch import multiprocessing as mp
from random import randrange
import argparse
from pangu_power.finetune.finetune_power import main, test_best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_net", type=str, default="fformula2")
    parser.add_argument(
        "--gpu_list",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPUs to use for finetuning",
    )
    parser.add_argument("--dist", action="store_true", help="Enable distributed mode")
    parser.add_argument(
        "--start_epoch", type=int, default=1, help="Starting epoch for training"
    )

    args = parser.parse_args()

    world_size = len(args.gpu_list)
    print(f"World size: {world_size if args.dist else 1}")

    # Pick a (somewhat) random port number for the master node, to run multiple instances on the same machine
    master_port = str(12357 + randrange(-20, 20, 1))
    print(f"Master port: {master_port}")

    # Spawn processes for distributed training
    if args.dist and torch.cuda.is_available():
        mp.spawn(main, args=(args, world_size, master_port), nprocs=world_size)  # type: ignore
    else:
        main(0, args, 1, master_port)

    # Test the best model on the test dataset
    test_best_model(args)

    # # Baseline tests (run separately)
    # test_baselines(args, 'formula')
