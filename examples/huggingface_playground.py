import marimo

__generated_with = "0.3.9"
app = marimo.App()


@app.cell
def __():
    import marimo_labs as molabs
    return molabs,


@app.cell(hide_code=True)
def __(mo, model_type, model_type_to_model):
    models = mo.ui.dropdown(
        model_type_to_model[model_type.value], label="Choose a model"
    )

    mo.hstack([model_type, models if model_type.value else ""], justify="start")
    return models,


@app.cell
def load_model(mo, models, molabs):
    mo.stop(models.value is None)
    model = molabs.huggingface.load(models.value)
    return model,


@app.cell
def __(mo, model):
    mo.md(
        f"""
        Example inputs:

        {mo.as_html(model.examples)}
        """
    ) if model.examples is not None else None
    return


@app.cell
def __(mo, model):
    inputs = model.inputs


    mo.vstack([mo.md("_Submit inputs to run inference_ 👇"), inputs])
    return inputs,


@app.cell
def __(inputs, mo, model):
    mo.stop(inputs.value is None)

    output = model.inference_function(inputs.value)
    output
    return output,


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell(hide_code=True)
def __(mo):
    audio_models = {
        "audio classification": "models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "audio to audio": "models/facebook/xm_transformer_sm_all-en",
        "speech recognition": "models/facebook/wav2vec2-base-960h",
        "text to speech": "models/julien-c/ljspeech_tts_train_tacotron2_raw_phn_tacotron_g2p_en_no_space_train",
    }

    image_models = {
        "image classification": "models/google/vit-base-patch16-224",
        "text to image": "models/runwayml/stable-diffusion-v1-5",
        "image to text": "models/Salesforce/blip-image-captioning-base",
        "object detection": "models/microsoft/table-transformer-detection",
    }

    text_models = {
        "feature extraction": "models/julien-c/distilbert-feature-extraction",
        "fill mask": "models/distilbert/distilbert-base-uncased",
        "zero-shot classification": "models/facebook/bart-large-mnli",
        "visual question answering": "models/dandelin/vilt-b32-finetuned-vqa",
        "sentence similarity": "models/sentence-transformers/all-MiniLM-L6-v2",
        "question answering": "models/deepset/xlm-roberta-base-squad2",
        "summarization": "models/facebook/bart-large-cnn",
        "text-classification": "models/distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "text generation": "models/openai-community/gpt2",
        "text2text generation": "models/valhalla/t5-small-qa-qg-hl",
        "translation": "models/Helsinki-NLP/opus-mt-en-ar",
        "token classification": "models/huggingface-course/bert-finetuned-ner",
        "document question answering": "models/impira/layoutlm-document-qa",
    }

    model_type_to_model = {
        "text": text_models,
        "image": image_models,
        "audio": audio_models,
        None: [],
    }

    model_type = mo.ui.dropdown(
        ["text", "image", "audio"], label="Choose a model type"
    )
    return (
        audio_models,
        image_models,
        model_type,
        model_type_to_model,
        text_models,
    )


if __name__ == "__main__":
    app.run()
