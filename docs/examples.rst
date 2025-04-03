.. _mmm-examples-label:

===================================
Code Examples
===================================

Imports
------------------------
..  code-block:: python

    from transformers import GenerationConfig, MistralForCausalLM
    from mmm import InferenceConfig, generate

Create the InferenceConfig
-----------------------------

In this case we are infilling the first track from the 14th to the 18th bar and the
third track from bar 40 to bar 44. We are also generating a new track with program 43.
Attribute controls have been included to control the generation,

..  code-block:: python

    INFERENCE_CONFIG = InferenceConfig(
        {
            0: [(14, 18, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
            2: [(40, 44, ["ACTrackRepetition_0.67", "ACBarNoteDurationEight_1"])],
        },
        [
            (43, ["ACBarPitchClass_3"]),
        ],
    )

Create the model
------------------------

Use the model of your choice.

..  code-block:: python

    model = MistralForCausalLM.from_pretrained(
            "path/to/model",
            use_safetensors=True,
        )

[Optional] Create the GenerationConfig
-----------------------------------------

This is used by huggingface transformer models to control the generation
(see https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/generation/configuration_utils.py#L94).

..  code-block:: python

    gen_config = GenerationConfig(
            num_beams=1,
            temperature=1.2,
            repetition_penalty=1.2,
            top_k=20,
            top_p=0.95,
            max_new_tokens=300,
        )

Generate
------------------------

..  code-block:: python

    # Instantiate the tokenizer
    MMM(params="/path/to/tokenizer.json")

    output_scores, metadata = generate(
            model,
            tokenizer,
            inference_config,
            "path/to/MIDIfile",
            {"generation_config": gen_config},
        )

    # Create MIDI output
    output_scores.dump_midi("output.mid")
