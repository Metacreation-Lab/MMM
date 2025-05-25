#!/usr/bin/python3 python

"""Test the MMM model."""

if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction
    from inspect import signature

    print("Importing")
    from transformers import Seq2SeqTrainingArguments
    print(' Seq2Seq imported')
    from utils.baselines import baselines
    print(' Baselines imported')
    from utils.testing import test_forward
    print("Done Importing")

    # Parse arguments for training params / model size
    parser = ArgumentParser(description="Model training script")
    parser.add_argument("--model", type=str, default="MMM_mistral")
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
    print("Parsed")

    # Identify model to train and tweak its training configuration
    baseline = baselines[args.pop("model")]
    for arg, value in args.items():
        if value is not None:
            baseline.training_config_kwargs[arg] = value

    # Training the model
    test_forward(baseline)
