import argparse
import json
import os

import matplotlib.pyplot as plt


def collect_series(log_history):
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for record in log_history:
        if "loss" in record:
            train_steps.append(record["step"])
            train_losses.append(record["loss"])
        if "eval_loss" in record:
            eval_steps.append(record["step"])
            eval_losses.append(record["eval_loss"])

    return train_steps, train_losses, eval_steps, eval_losses


def main():
    parser = argparse.ArgumentParser(description="Plot train/eval loss from Hugging Face trainer_state.json.")
    parser.add_argument(
        "--state",
        default="output_dir/gpt2-1b-russian/trainer_state.json",
        help="Path to trainer_state.json saved by Trainer.save_state().",
    )
    parser.add_argument(
        "--output",
        default="output_dir/gpt2-1b-russian/loss_curve.png",
        help="Where to save the loss plot.",
    )
    args = parser.parse_args()

    with open(args.state, "r", encoding="utf-8") as fp:
        state = json.load(fp)

    train_steps, train_losses, eval_steps, eval_losses = collect_series(state["log_history"])
    if not train_losses and not eval_losses:
        raise ValueError(f"No loss values found in {args.state}")

    plt.figure(figsize=(10, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, label="train loss", linewidth=1.5)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval loss", marker="o", linewidth=1.5)

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Mini-pretrain loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(f"Saved loss plot to {args.output}")


if __name__ == "__main__":
    main()
