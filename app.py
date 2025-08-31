import gradio as gr
import torch
import numpy as np
import requests
import soundfile as sf
from io import BytesIO
import librosa
import traceback 
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import SpeakerRecognition 


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """Loads all the required models from Hugging Face."""
    print("Loading AI models...")
    try:
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        speaker_model = SpeakerRecognition.from_huggingface(
            "speechbrain/speaker-recognition-ecapa-tdnn", 
            savedir="speechbrain_models"
        ).to(device)
        print("Models loaded successfully.")
        return processor, model, vocoder, speaker_model
    except Exception as e:
        print("--- FATAL ERROR: Could not load base models. ---")
        traceback.print_exc()
        return None, None, None, None

processor, model, vocoder, speaker_model = load_models()


STABLE_VOICE_URL = "https://github.com/gradio-app/gradio/raw/main/demo/audio_debugger/cantina.wav"
speaker_embedding = None

if all([processor, model, vocoder, speaker_model]): 
    print("Generating single speaker embedding from a stable audio file...")
    try:
        response = requests.get(STABLE_VOICE_URL, timeout=15)
        response.raise_for_status()
        
        audio_data, sample_rate = sf.read(BytesIO(response.content))
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if sample_rate != 16000:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)

 
        with torch.no_grad():
            embedding_tensor = speaker_model.encode_batch(torch.tensor(audio_data).to(device))
 
            speaker_embedding = embedding_tensor.squeeze().unsqueeze(0).to(device)
        
        print("- Single voice generated successfully.")

    except Exception:
        print(f"--- FATAL ERROR: Could not generate voice from the stable URL. ---")
        traceback.print_exc()

def synthesize_speech(text):
    """
    Generates audio from text using the single, reliable speaker voice.
    """
    if not text or speaker_embedding is None:
        return (16000, np.zeros(0).astype(np.int16))

    inputs = processor(text=text, return_tensors="pt").to(device)
    
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding)
    
    with torch.no_grad():
        speech = vocoder(spectrogram)
    
    return (16000, speech.cpu().numpy())



theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue, 
    font=gr.themes.GoogleFont("Inter")
)

with gr.Blocks(theme=theme) as demo:
    if not speaker_embedding or not model:

        gr.HTML('<h1 style="text-align: center; color: red;">Application Startup Failed</h1>')
        gr.Markdown("""
        ## Fatal Error: The application could not start.
        This could be due to a failure in loading the core AI models or the voice generation sample.
        This is often a temporary network issue on Hugging Face's servers.
        
        **What you can do:**
        1.  Try **restarting the Space** from the settings menu (three dots icon ፧ at the top-right).
        2.  Check the **Logs** tab for detailed error messages.
        """)
    else:
        gr.HTML('<h1 style="text-align: center; font-size: 2.5em;">AI Text-to-Speech Converter 🗣️</h1>')
        gr.Markdown("<p style='text-align: center;'>Enter any text, and listen to the AI speak! This app uses Microsoft's SpeechT5 model.</p>")
        
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to Speak", 
                    placeholder="Type your text here...",
                    lines=5
                )
                generate_button = gr.Button("🔊 Generate Speech", variant="primary")
                
            with gr.Column(scale=2):
                audio_output = gr.Audio(label="AI-Generated Speech", autoplay=False)

        generate_button.click(
            fn=synthesize_speech,
            inputs=text_input, 
            outputs=audio_output
        )

        gr.Examples(
            examples=[
                ["Hello, this is a test of the text to speech system."],
                ["Artificial intelligence will reshape the world."],
            ],
            inputs=text_input,
            outputs=audio_output,
            fn=synthesize_speech,
            cache_examples=False 
        )
        
        with gr.Accordion("How does this work?", open=False):
            gr.Markdown("""
            This app uses three main AI models:
            1.  **[microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts):** Converts your text into a spectrogram (a visual representation of sound).
            2.  **[microsoft/speecht5_hifigan](https://huggingface.co/microsoft/speecht5_hifigan):** A vocoder that transforms the spectrogram into a high-fidelity audio waveform.
            3.  **[speechbrain/speaker-recognition-ecapa-tdnn](https://huggingface.co/speechbrain/speaker-recognition-ecapa-tdnn):** Analyzes a voice clip to create a unique 'speaker embedding', which gives the AI its voice.
            """)
        
        gr.Markdown("""--- \n Created by [Süleyman Toklu](https://github.com/SuleymanToklu) for the #30DayAIMarathon.""")


demo.launch()

