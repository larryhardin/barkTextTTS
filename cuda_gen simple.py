import warnings
import logging
import os
import re
import scipy.io.wavfile
import torch
import numpy as np
from transformers import BarkModel, AutoProcessor


warnings.filterwarnings("ignore", message=".*Passing `generation_config` together with generation-related arguments*")

class _SuppressTransformersLengthWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Both `max_new_tokens`" not in msg or "and `max_length`" not in msg


logging.getLogger("transformers").addFilter(_SuppressTransformersLengthWarning())

def clean_text_for_bark(text: str) -> str:
    """
    Cleans raw text to improve Bark's voice synthesis quality.
    Spells out common abbreviations and formats acronyms.
    """
    text = re.sub(r'\b(CUDA|TTS|AI|GPU)\b', lambda m: " ".join(m.group(1)), text)
    
    replacements = {
        r'\bDr\b\.?': "Doctor",
        r'\bMr\b\.?': "Mister",
        r'\bMs\b\.?': "Missus",
        r'\betc\b\.?': "et cetera",
        r'\bvs\b\.?': "versus",
    }
    for pattern, replacement in replacements.items():
        text = re.compile(pattern, re.IGNORECASE).sub(replacement, text)
        
    return text

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Boosts the audio volume and normalizes the waveform.
    Casts the array to float32 to prevent scipy saving errors.
    """
    if np.max(np.abs(audio_data)) == 0:
        return audio_data.astype(np.float32)
    
    # Scale to maximum range and cast to a valid scipy data type
    normalized = audio_data / np.max(np.abs(audio_data))
    return normalized.astype(np.float32)

def generate_high_quality_tts():
    raw_text = "To bring that whole preview phase to life, you use that Vee C Rio for the configurationwe just talked about."
    processed_text = clean_text_for_bark(raw_text)
    print(f"Processing text: '{processed_text}'")

    # 1. Load the processor and model directly onto the GPU using native classes
    print("Loading Suno Bark model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained("suno/bark")
    
    # Using torch.float16 for VRAM management and faster generation
    model = BarkModel.from_pretrained(
        "suno/bark", 
        torch_dtype=torch.float16
    ).to(device)

    # 2. Process text and voice preset into tensors
    print("Processing text and voice preset tensors...")
    #voice_preset = "v2/en_speaker_0" # (American, female, neutral)
    #voice_preset = "v2/en_speaker_1" # (American, female, neutral)
    #voice_preset = "v2/en_speaker_2" # (American, female, neutral)
    #voice_preset = "v2/en_speaker_3" # (American, female, neutral
    #voice_preset = "v2/en_speaker_4" # (American, female, neutral
    #voice_preset = "v2/en_speaker_5" # (American, female, neutral
    #voice_preset = "v2/en_speaker_6" # (American, male, neutral)
    voice_preset = "v2/en_speaker_7" # (Naural tone)
    inputs = processor(text=[processed_text], voice_preset=voice_preset)
    
    # Move all input tensors to the GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3. Generate audio directly from the model weights (bypasses pipeline validation)
    print("Generating audio waveform via native model execution...")
    with torch.no_grad():
        # Bark generates native audio arrays directly when called this way
        audio_outputs = model.generate(**inputs)
        
    # Convert tensor output back to a CPU numpy array
    audio_array = audio_outputs.cpu().numpy().squeeze()

    # 4. Post-process audio array
    # The default sampling rate for the Suno Bark model is 24000 Hz
    sampling_rate = 24000 
    normalized_audio = normalize_audio(audio_array)

    # 5. Save the crisp output to disk
    output_filename = "bark_fixed_cuda_output.wav"
    scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=normalized_audio)
    print(f"Saved optimized audio file to: {output_filename}")

if __name__ == "__main__":
    os.environ["SUNO_ENABLE_CPU_OFFLOAD"] = "True"
    
    if torch.cuda.is_available():
        generate_high_quality_tts()
    else:
        print("ERROR: CUDA is not available on this system. Please check your PyTorch installation.")
