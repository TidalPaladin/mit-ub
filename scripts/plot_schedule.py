from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from deep_helpers.optim.rsqrt import get_lr, get_momentum
from matplotlib import pyplot as plt


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="plot-schedule", description="Plot a reciprocal square root schedule")
    parser.add_argument(
        "-p", "--peak", type=float, required=True, help="Peak value (learning rate or momentum) to reach after warmup"
    )
    parser.add_argument("-i", "--initial", type=float, default=0.0, help="Initial value before warmup")
    parser.add_argument("-t", "--total-steps", type=int, default=50000, help="Total number of training steps")
    parser.add_argument(
        "-w", "--warmup-steps", type=int, default=1000, help="Number of warmup steps to reach peak value"
    )
    parser.add_argument(
        "-c", "--cooldown-steps", type=int, default=10000, help="Number of cooldown steps at end of training"
    )
    parser.add_argument(
        "-ti", "--timescale", type=int, default=10000, help="Timescale parameter controlling schedule shape"
    )
    parser.add_argument(
        "-s", "--stopped-steps", type=int, default=0, help="Number of steps to hold initial value at end of training"
    )
    parser.add_argument("-o", "--output", type=Path, default="schedule.png")
    parser.add_argument(
        "-m", "--momentum", default=False, action="store_true", help="Plot a momentum schedule (cycled inversely)"
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    x = np.arange(args.total_steps)
    if args.momentum:
        y = np.array(
            [
                get_momentum(
                    step=min(x, args.total_steps - args.stopped_steps),
                    base_momentum=args.peak,
                    warmup_steps=args.warmup_steps,
                    cooldown_steps=args.cooldown_steps,
                    total_steps=args.total_steps - args.stopped_steps,
                    timescale=args.timescale,
                    initial_momentum=args.initial,
                )
                for x in np.arange(args.total_steps)
            ]
        )
    else:
        y = np.array(
            [
                get_lr(
                    step=min(x, args.total_steps - args.stopped_steps),
                    base_lr=args.peak,
                    warmup_steps=args.warmup_steps,
                    cooldown_steps=args.cooldown_steps,
                    total_steps=args.total_steps - args.stopped_steps,
                    timescale=args.timescale,
                    initial_lr=args.initial,
                )
                for x in np.arange(args.total_steps)
            ]
        )

    plt.plot(x, y)
    plt.xlabel("Step")
    plt.ylabel("Momentum" if args.momentum else "Learning Rate")
    plt.title(f"{'Momentum' if args.momentum else 'Learning Rate'} Schedule")
    plt.savefig(args.output)


def entrypoint() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
