import gradio as gr
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

def create_speaker_embedding():
    """
    Creates a speaker embedding from a reliable audio source.
    This version uses a pre-processed dataset that is more stable.
    """
    try:
        embeddings_dataset = load_dataset("huggingface-course/cmu-arctic-xvectors-processed", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
        print("Speaker embedding created successfully.")
        return speaker_embedding
    except Exception as e:
        print(f"Error creating speaker embedding: {e}")
        print("Falling back to a zero-tensor speaker embedding.")
        return torch.zeros((1, 512)).to(device)

speaker_embedding = create_speaker_embedding()

def synthesize_speech(text):
    """
    Generates speech from the input text using the pre-loaded models and speaker embedding.
    """
    inputs = processor(text=text, return_tensors="pt").to(device)
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)
    speech = vocoder(spectrogram)
    speech_numpy = speech.detach().cpu().numpy()
    
    return (16000, speech_numpy)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Day 19: AI Text-to-Speech Converter 🗣️
        This application uses the `microsoft/speecht5_tts` model to convert your text into realistic human speech.
        Enter some text, and click the button to generate the audio.
        """
    )
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Your Text",
                placeholder="Type or paste your text here...",
                lines=5
            )
            generate_button = gr.Button("Generate Speech", variant="primary")
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech", autoplay=False)

    gr.Examples(
        examples=[
            "Hello, my name is Süleyman and I am an AI enthusiast!",
            "Artificial intelligence will reshape the future of technology.",
            "This text-to-speech conversion is powered by Hugging Face transformers.",
        ],
        inputs=text_input
    )
    gr.Markdown(
        """
        ---
        *Created by Süleyman Toklu for the #30DayAIMarathon.*
        """
    )
    generate_button.click(
        fn=synthesize_speech,
        inputs=text_input,
        outputs=audio_output
    )
if __name__ == "__main__":
    demo.launch() 