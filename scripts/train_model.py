#!/usr/bin/python3 python

"""Train the MMM model."""

if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction
    from inspect import signature

    from transformers import Seq2SeqTrainingArguments
    from utils.baselines import baselines
    from utils.training import whole_training_process

    # Parse arguments for training params / model size
    parser = ArgumentParser(description="Model training script")
    parser.add_argument("--model", type=str, default="MMM_gpt2")
    parser.add_argument("--left_padding", action="store_true")
    for param in signature(Seq2SeqTrainingArguments).parameters.values():
        key = param.name.replace("_", "-")
        if param.annotation is bool:
            parser.add_argument(f"--{key}", action=BooleanOptionalAction, default=None)
        else:
            kwargs = {"default": None}
            if param.annotation in (int, float, str):
                kwargs["type"] = param.annotation
            parser.add_argument(f"--{key}", **kwargs)
    args = vars(parser.parse_args())

    # Identify model to train and tweak its training configuration
    baseline = baselines[args.pop("model")]
    left_padding = args.pop("left_padding")
    for arg, value in args.items():
        if value is not None:
            baseline.training_config_kwargs[arg] = value

    # Training the model
    whole_training_process(baseline, do_test=False, pad_on_left=left_padding)
