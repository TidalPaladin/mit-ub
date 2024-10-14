from argparse import ArgumentParser, Namespace
from typing import cast

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mit_ub.model import BACKBONES, AdaptiveViT, ViT


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="profile", description="Profile MiT-UB")
    parser.add_argument("model", type=str, help="Name of the model to profile")
    parser.add_argument("size", type=int, nargs="+", help="Input size")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("-c", "--channels", type=int, default=3, help="Number of channels")
    parser.add_argument("-d", "--device", default="cpu", help="Device to use")
    parser.add_argument("-t", "--training", default=False, action="store_true", help="Use training mode")
    return parser.parse_args()


def main(args: Namespace):
    model = cast(ViT | AdaptiveViT, BACKBONES.get(args.model).instantiate_with_metadata().fn)
    model = model.to(args.device)
    model.train(args.training)
    device = torch.device(args.device)

    x = torch.randn(args.batch_size, args.channels, *args.size, device=device)

    # Run the model once to trigger torch.compile
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        with torch.no_grad():
            model(x)

    activities = [ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU]
    with torch.no_grad():
        with profile(activities=activities, record_shapes=True) as prof:
            with record_function("model_inference"):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    model(x)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
