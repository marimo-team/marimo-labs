from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import huggingface_hub  # type: ignore
import marimo as mo
import requests  # type: ignore

from marimo_labs.huggingface import _load_utils, _outputs
from marimo_labs.huggingface._processing_utils import (
    encode_to_base64,
    save_base64_to_cache,
)


@dataclass
class HFModel:
    title: str
    inputs: mo.ui.form
    examples: list[Any] | None
    inference_function: Callable[..., _outputs.Output]


def load(
    name: str,
    hf_token: str | None = None,
    **kwargs,
) -> HFModel:
    """Constructs a demo from a Hugging Face repo.

    Can accept model repos (if src is "models"). The input and output
    components are automatically loaded from the repo. Note that if a Space is
    loaded, certain high-level attributes of the Blocks (e.g. custom `css`,
    `js`, and `head` attributes) will not be loaded.

    Parameters:
        name: the name of the model (e.g. "gpt2" or "facebook/bart-base") or
            space (e.g. "flax-community/spanish-gpt2"), can include the `src`
            as prefix (e.g. "models/facebook/bart-base")

        src: the source of the model: `models` or `spaces` (or leave empty if
            source is provided as a prefix in `name`)

        hf_token: optional access token for loading private Hugging Face Hub
            models. Find your token here:
            https://huggingface.co/settings/tokens.

    Returns:
        an HFModel object

    Example:

    ```python
    import marimo_labs

    model = marimo_labs.load("models/runwayml/stable-diffusion-v1-5")
    inputs = model.inputs
    inputs
    ```

    ```python
    model.inference_function(inputs.value)
    ```
    """
    return load_model_from_repo(name=name, hf_token=hf_token, **kwargs)


def load_model_from_repo(
    name: str,
    hf_token: str | None = None,
    **kwargs,
) -> HFModel:
    """Creates and returns an HFModel"""
    # Separate the repo type (e.g. "model") from repo name (e.g.
    # "google/vit-base-patch16-224")
    tokens = name.split("/")
    if len(tokens) <= 1:
        raise ValueError(
            "Either `src` parameter must be provided, or "
            "`name` must be formatted as {src}/{repo name}"
        )
    src = tokens[0]
    name = "/".join(tokens[1:])

    factory_methods: dict[str, Callable] = {
        # for each repo type, we have a method that returns the Interface given
        # the model name & optionally an hf_token
        "huggingface": from_model,
        "models": from_model,
    }
    if src.lower() not in factory_methods:
        raise ValueError(
            f"parameter: src must be one of {factory_methods.keys()}"
        )

    return factory_methods[src](name, hf_token, **kwargs)


def from_model(model_name: str, hf_token: str | None, **kwargs) -> HFModel:
    del kwargs

    model_url = f"https://huggingface.co/{model_name}"
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"

    print(f"Fetching model from: {model_url}")

    headers = (
        {"Authorization": f"Bearer {hf_token}"} if hf_token is not None else {}
    )
    response = requests.request("GET", api_url, headers=headers)
    if response.status_code != 200:
        raise ValueError(
            f"Could not find model: {model_name}. "
            "If it is a private or gated model, please provide your "
            "Hugging Face access token "
            "(https://huggingface.co/settings/tokens) as the argument for the "
            "`hf_token` parameter."
        )
    p = response.json().get("pipeline_tag")

    headers["X-Wait-For-Model"] = "true"
    client = huggingface_hub.InferenceClient(
        model=model_name, headers=headers, token=hf_token
    )

    # For tasks that are not yet supported by the InferenceClient
    MARIMOLABS_CACHE = os.environ.get(
        "MARIMOLABS_TEMP_DIR"
    ) or str(  # noqa: N806
        Path(tempfile.gettempdir()) / "marimo_labs"
    )

    def custom_post_binary(data):
        response = requests.request(
            "POST", api_url, headers=headers, content=data
        )
        return save_base64_to_cache(
            encode_to_base64(response), cache_dir=MARIMOLABS_CACHE
        )

    inputs: mo.Html  # actually a UIElement, but not in public API ...
    preprocess = None
    postprocess: Callable[..., Any] | None = None
    examples: Any = None
    fn: Callable[..., _outputs.Output]

    # example model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    if p == "audio-classification":
        inputs = mo.ui.file(
            filetypes=["audio/*"], label="Upload an audio file", kind="area"
        )
        postprocess = _load_utils.postprocess_label
        examples = [
            "https://gradio-builds.s3.amazonaws.com/demo-files/audio_sample.wav"
        ]
        fn = _outputs.construct_output_function(
            inference_function=_load_utils.file_contents_wrapper(
                client.audio_classification
            )
        )
    # example model: facebook/xm_transformer_sm_all-en
    elif p == "audio-to-audio":
        inputs = mo.ui.file(
            filetypes=["audio/*"], label="Upload an audio file", kind="area"
        )
        # output_function = components.Audio(label="Output")
        examples = [
            "https://gradio-builds.s3.amazonaws.com/demo-files/audio_sample.wav"
        ]
        fn = _outputs.construct_output_function(
            lambda v: custom_post_binary(v.contents),
            _outputs.audio_output_from_path,
        )
    # example model: facebook/wav2vec2-base-960h
    elif p == "automatic-speech-recognition":
        inputs = mo.ui.file(
            filetypes=["audio/*"], label="Upload an audio file", kind="area"
        )
        # outputs = components.Textbox(label="Output")
        examples = [
            "https://gradio-builds.s3.amazonaws.com/demo-files/audio_sample.wav"
        ]
        fn = _outputs.construct_output_function(
            _load_utils.file_contents_wrapper(
                client.automatic_speech_recognition
            )
        )
    # example model: microsoft/DialoGPT-medium
    elif p == "conversational":
        raise NotImplementedError
        # inputs = [
        #    components.Textbox(render=False),
        #    components.State(render=False),
        # ]
        # outputs = [
        #    components.Chatbot(render=False),
        #    components.State(render=False),
        # ]
        # examples = [["Hello World"]]
        # preprocess = external_utils.chatbot_preprocess
        # postprocess = external_utils.chatbot_postprocess
        # fn = client.conversational
    # example model: julien-c/distilbert-feature-extraction
    elif p == "feature-extraction":
        inputs = mo.ui.text_area(label="Text to featurize")
        # outputs = components.Dataframe(label="Output")
        postprocess = lambda v: v[0] if len(v) == 1 else v  # type: ignore # noqa: E731
        fn = _outputs.construct_output_function(client.feature_extraction)
    # example model: distilbert/distilbert-base-uncased
    elif p == "fill-mask":
        inputs = mo.ui.text_area(label="Masked text")
        # outputs = components.Label(label="Classification")
        examples = [
            "Hugging Face is the AI community, working together, to "
            "[MASK] the future."
        ]
        postprocess = _load_utils.postprocess_mask_tokens
        fn = _outputs.construct_output_function(client.fill_mask)
    # Example: google/vit-base-patch16-224
    elif p == "image-classification":
        inputs = mo.ui.file(
            filetypes="image/*", label="Input Image", kind="area"
        )
        # outputs = components.Label(label="Classification")
        postprocess = _load_utils.postprocess_label
        examples = [
            "https://gradio-builds.s3.amazonaws.com/demo-files/cheetah-002.jpg"
        ]
        fn = _outputs.construct_output_function(
            _load_utils.file_contents_wrapper(client.image_classification)
        )
    # Example: deepset/xlm-roberta-base-squad2
    elif p == "question-answering":
        inputs = mo.ui.array(
            [
                mo.ui.text_area(label="Question"),
                mo.ui.text_area(rows=7, label="Context"),
            ]
        )
        # outputs = [
        #    components.Textbox(label="Answer"),
        #    components.Label(label="Score"),
        # ]
        examples = [
            [
                "What entity was responsible for the Apollo program?",
                "The Apollo program, also known as Project Apollo, was the "
                "third United States human spaceflight"
                " program carried out by the National Aeronautics and Space "
                "Administration (NASA), which accomplished"
                " landing the first humans on the Moon from 1969 to 1972.",
            ]
        ]
        postprocess = _load_utils.postprocess_question_answering
        fn = _outputs.construct_output_function(client.question_answering)
    # Example: facebook/bart-large-cnn
    elif p == "summarization":
        inputs = mo.ui.text_area(label="Text to summarize")
        # outputs = components.Textbox(label="Summary")
        examples = [
            [
                "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."  # noqa: E501
            ]
        ]
        fn = _outputs.construct_output_function(client.summarization)
    # Example: distilbert-base-uncased-finetuned-sst-2-english
    elif p == "text-classification":
        inputs = mo.ui.text_area(label="Text to classify")
        # outputs = components.Label(label="Classification")
        examples = ["I feel great"]
        postprocess = _load_utils.postprocess_label
        fn = _outputs.construct_output_function(client.text_classification)
    # Example: gpt2
    elif p == "text-generation":
        inputs = mo.ui.text_area(label="Prompt")
        # outputs = inputs
        examples = ["Once upon a time"]
        fn = _outputs.construct_output_function(
            _load_utils.text_generation_wrapper(client)
        )
    # Example: valhalla/t5-small-qa-qg-hl
    elif p == "text2text-generation":
        inputs = mo.ui.text_area(label="Input text")
        # outputs = components.Textbox(label="Generated Text")
        examples = ["Translate English to Arabic: How are you?"]
        fn = _outputs.construct_output_function(client.text_generation)
    # Example: Helsinki-NLP/opus-mt-en-ar
    elif p == "translation":
        inputs = mo.ui.text_area(label="Text to translate")
        # outputs = components.Textbox(label="Translation")
        examples = ["Hello, how are you?"]
        fn = _outputs.construct_output_function(client.translation)
    # Example: facebook/bart-large-mnli
    elif p == "zero-shot-classification":
        inputs = mo.ui.array(
            [
                mo.ui.text_area(label="Input text"),
                mo.ui.text_area(
                    label="Possible class names (comma-separated)"
                ),
                mo.ui.checkbox(label="Allow multiple true classes"),
            ]
        )
        # outputs = components.Label(label="Classification")
        postprocess = _load_utils.postprocess_label
        examples = [["I feel great", "happy, sad", False]]
        fn = _outputs.construct_output_function(
            _load_utils.zero_shot_classification_wrapper(client)
        )
    # Example: sentence-transformers/distilbert-base-nli-stsb-mean-tokens
    elif p == "sentence-similarity":
        inputs = mo.ui.array(
            [
                mo.ui.text_area(
                    label="Source Sentence",
                    placeholder="Enter an original sentence",
                ),
                mo.ui.text_area(
                    rows=7,
                    placeholder=(
                        "Sentences to compare to -- separate each "
                        "sentence by a newline"
                    ),
                    label="Sentences to compare to",
                ),
            ]
        )
        # outputs = components.JSON(label="Similarity scores")
        examples = [["That is a happy person", "That person is very happy"]]
        fn = _outputs.construct_output_function(
            _load_utils.sentence_similarity_wrapper(client)
        )
    # Example: julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train  # noqa: E501
    elif p == "text-to-speech":
        inputs = mo.ui.text_area(label="Text")
        # outputs = components.Audio(label="Audio")
        examples = ["Hello, how are you?"]
        fn = _outputs.construct_output_function(
            client.text_to_speech, _outputs.audio_output_from_bytes
        )
    # example model: osanseviero/BigGAN-deep-128
    elif p == "text-to-image":
        inputs = mo.ui.text_area(label="Prompt")
        # outputs = components.Image(label="Output")
        examples = ["A beautiful sunset"]
        fn = _outputs.construct_output_function(
            client.text_to_image, _outputs.image_output
        )
    # example model: huggingface-course/bert-finetuned-ner
    elif p == "token-classification":
        inputs = mo.ui.text_area(label="Input")
        # outputs = components.HighlightedText(label="Output")
        examples = [
            "marimo is a new kind of programming environment for ML/AI."
        ]
        fn = _outputs.construct_output_function(
            _load_utils.token_classification_wrapper(client)
        )
    # example model: impira/layoutlm-document-qa
    elif p == "document-question-answering":
        inputs = mo.ui.array(
            [
                mo.ui.file(label="Upload a document", kind="area"),
                mo.ui.text_area(label="Question"),
            ]
        )
        postprocess = _load_utils.postprocess_label
        # outputs = components.Label(label="Label")
        fn = _outputs.construct_output_function(
            lambda file_upload_results, text: client.document_question_answering(  # noqa: E501
                file_upload_results[0].contents,
                text,
            )
        )
    # example model: dandelin/vilt-b32-finetuned-vqa
    elif p == "visual-question-answering":
        inputs = mo.ui.array(
            [
                mo.ui.file(
                    filetypes=["image/*"], label="Input Image", kind="area"
                ),
                mo.ui.text_area(label="Question"),
            ]
        )
        # outputs = components.Label(label="Label")
        postprocess = _load_utils.postprocess_visual_question_answering
        examples = [
            [
                "https://gradio-builds.s3.amazonaws.com/demo-files/cheetah-002.jpg",
                "What animal is in the image?",
            ]
        ]
        fn = _outputs.construct_output_function(
            lambda file_upload_results, text: client.visual_question_answering(
                file_upload_results[0].contents,
                text,
            )
        )
    # example model: Salesforce/blip-image-captioning-base
    elif p == "image-to-text":
        inputs = mo.ui.file(
            filetypes=["image/*"], label="Input Image", kind="area"
        )
        # outputs = components.Textbox(label="Generated Text")
        examples = [
            "https://gradio-builds.s3.amazonaws.com/demo-files/cheetah-002.jpg"
        ]
        fn = _outputs.construct_output_function(
            _load_utils.file_contents_wrapper(client.image_to_text)
        )
    # example model: rajistics/autotrain-Adult-934630783
    elif p in ["tabular-classification", "tabular-regression"]:
        raise NotImplementedError
        # examples = _load_utils.get_tabular_examples(model_name)
        # col_names, examples = _load_utils.cols_to_rows(examples)
        # examples = [[examples]] if examples else None
        # inputs = components.Dataframe(
        #     label="Input Rows",
        #     type="pandas",
        #     headers=col_names,
        #     col_count=(len(col_names), "fixed"),
        #     render=False,
        # )
        # outputs = components.Dataframe(
        #    label="Predictions", type="array", headers=["prediction"]
        # )
        # fn = external_utils.tabular_wrapper
        # output_function = fn
    # example model: microsoft/table-transformer-detection
    elif p == "object-detection":
        inputs = mo.ui.file(
            filetypes=["image/*"], label="Input Image", kind="area"
        )
        # outputs = components.AnnotatedImage(label="Annotations")
        # TODO(akshayka): output representation
        fn = _outputs.construct_output_function(
            _load_utils.file_contents_wrapper(
                _load_utils.object_detection_wrapper(client)
            )
        )
    else:
        raise ValueError(f"Unsupported pipeline type: {p}")

    def query_huggingface_inference_endpoints(data: Any) -> _outputs.Output:
        if not isinstance(data, (list, tuple)):
            data = [data]

        if preprocess is not None:
            data = preprocess(*data)
        output = fn(*data)  # type: ignore
        value = output.value
        if postprocess is not None:
            value = postprocess(value)  # type: ignore
        output.value = value
        return output

    query_huggingface_inference_endpoints.__name__ = model_name
    return HFModel(
        title=model_name,
        inputs=inputs.form(bordered=False),  # type: ignore
        examples=examples,
        inference_function=query_huggingface_inference_endpoints,
    )
