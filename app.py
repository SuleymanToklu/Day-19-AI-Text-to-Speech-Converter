import gradio as gr
import torch
import numpy as np
import requests
import soundfile as sf
from io import BytesIO
import librosa 
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

VOICE_URLS = {
    "Davut (Male)": "https://raw.githubusercontent.com/microsoft/cognitive-services-speech-sdk/master/samples/csharp/sharedcontent/console/whatstheweatherlike.wav",
    "Selin (Female)": "https://github.com/librosa/librosa/blob/main/tests/data/test1_44100.wav?raw=true"
}

speaker_embeddings = {}

print("Generating speaker embeddings from live audio files...")
for name, url in VOICE_URLS.items():
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        audio_data, sample_rate = sf.read(BytesIO(response.content))
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if sample_rate != 16000:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)

        embedding = processor(audio=audio_data, sampling_rate=16000, return_tensors="pt").speaker_embeddings
        speaker_embeddings[name] = embedding.to(device)
        print(f"- Voice '{name}' generated successfully.")

    except Exception as e:
        print(f"Could not generate voice '{name}'. Error: {e}")
        continue

speaker_names = list(speaker_embeddings.keys())

def synthesize_speech(text, speaker_name):
    """
    Generates audio from text using the selected speaker's voice.
    """
    if not text or not speaker_name:
        return (16000, np.zeros(0).astype(np.int16))

    inputs = processor(text=text, return_tensors="pt").to(device)
    
    selected_embedding = speaker_embeddings[speaker_name]
    
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings=selected_embedding)
    
    with torch.no_grad():
        speech = vocoder(spectrogram)
    
    return (16000, speech.cpu().numpy())


theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue, 
    font=gr.themes.GoogleFont("Inter")
)

with gr.Blocks(theme=theme) as demo:
    gr.HTML('<h1 style="text-align: center; font-size: 2.5em;">AI Text-to-Speech Converter 🗣️</h1>')
    gr.Markdown("<p style='text-align: center;'>Enter any text, choose a voice, and listen to the AI speak! This app uses Microsoft's SpeechT5 model.</p>")
    
    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Text to Speak", 
                placeholder="Type your text here...",
                lines=4
            )
            speaker_dropdown = gr.Dropdown(
                choices=speaker_names, 
                value=speaker_names[0] if speaker_names else None, 
                label="Choose a Voice"
            )
            generate_button = gr.Button("🔊 Generate Speech", variant="primary")
            
        with gr.Column(scale=2):
            audio_output = gr.Audio(label="AI-Generated Speech", autoplay=False)

    generate_button.click(
        fn=synthesize_speech,
        inputs=[text_input, speaker_dropdown],
        outputs=audio_output
    )

    gr.Examples(
        examples=[
            ["Hello, this is a test of the text to speech system.", "Selin (Female)"],
            ["Artificial intelligence will reshape the world.", "Davut (Male)"],
        ],
        inputs=[text_input, speaker_dropdown],
        outputs=audio_output,
        fn=synthesize_speech,
        cache_examples=False 
    )
    
    with gr.Accordion("How does this work?", open=False):
        gr.Markdown("""
        This app uses two main AI models from Hugging Face:
        1.  **[microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts):** A powerful model that converts your text into a spectrogram (a visual representation of sound).
        2.  **[microsoft/speecht5_hifigan](https://huggingface.co/microsoft/speecht5_hifigan):** A vocoder that transforms the spectrogram into a high-fidelity audio waveform that you can hear.
        
        The different voices are created by generating 'speaker embeddings' from short, reliable audio clips on the fly, which makes the app robust against broken links.
        """)
    
    gr.Markdown("""--- \n Created by [Süleyman Toklu](https://github.com/SuleymanToklu) for the #30DayAIMarathon.""")


demo.launch()

