import argparse

from tiur_tricks import run_set1_quick


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="./tiur_out")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    run_set1_quick(device=args.device, out_dir=args.out_dir, data_dir=args.data_dir, fast=args.fast)


if __name__ == "__main__":
    main()
