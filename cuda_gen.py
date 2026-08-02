import argparse
import json
import logging
import os
import re
import uuid
import warnings
from typing import Any, List, Optional

import numpy as np
import scipy.io.wavfile
import torch
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings("ignore", message=".*Passing `generation_config` together with generation-related arguments*")

_BARK_PROCESSOR: Optional[Any] = None
_BARK_MODEL: Optional[Any] = None


class _SuppressTransformersLengthWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "max_new_tokens" in msg.lower() or "max_length" in msg.lower():
            return False
        return True


logging.getLogger("transformers").addFilter(_SuppressTransformersLengthWarning())
logging.getLogger("transformers").setLevel(logging.ERROR)


def clean_text_for_bark(text: str) -> str:
    """
    Cleans raw text to improve Bark's voice synthesis quality.
    Spells out common abbreviations and formats acronyms.
    """
    text = re.sub(r"\b(CUDA|TTS|AI|GPU)\b", lambda m: " ".join(m.group(1)), text)

    replacements = {
        r"\bDr\b\.?": "Doctor",
        r"\bMr\b\.?": "Mister",
        r"\bMs\b\.?": "Missus",
        r"\betc\b\.?": "et cetera",
        r"\bvs\b\.?": "versus",
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


def log(message: str) -> None:
    print(message, flush=True)


def generate_output_filename() -> str:
    return f"{uuid.uuid4()}.wav"


def concatenate_audio_segments(audio_segments: List[np.ndarray]) -> np.ndarray:
    if not audio_segments:
        return np.array([], dtype=np.float32)

    return np.concatenate([segment.astype(np.float32, copy=False) for segment in audio_segments], axis=0)


def load_bark_model_components() -> tuple[Any, Any]:
    global _BARK_PROCESSOR, _BARK_MODEL

    if _BARK_PROCESSOR is not None and _BARK_MODEL is not None:
        return _BARK_PROCESSOR, _BARK_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    log("Loading Suno Bark model and processor...")
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark", torch_dtype=dtype).to(device)

    _BARK_PROCESSOR = processor
    _BARK_MODEL = model
    return processor, model


def cleanup_bark_model_components() -> None:
    global _BARK_PROCESSOR, _BARK_MODEL

    _BARK_PROCESSOR = None
    _BARK_MODEL = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Bark TTS audio from text, a TXT script, or a JSON script")
    parser.add_argument(
        "--text-to-speak",
        "--text_to_speak",
        dest="text_to_speak",
        default=None,
        help="Text to synthesize. If omitted, the default fallback text is used only when no script JSON is provided.",
    )
    parser.add_argument(
        "--voice-profile",
        "--voice_profile",
        dest="voice_profile",
        default="v2/en_speaker_7",
        choices=[f"v2/en_speaker_{i}" for i in range(10)],
        help="Voice preset to use for synthesis.",
    )
    parser.add_argument(
        "--script-json",
        "--script_json",
        dest="script_json",
        default=None,
        help="Optional JSON file containing segments to synthesize one at a time.",
    )
    parser.add_argument(
        "--script-txt",
        "--script_txt",
        dest="script_txt",
        default=None,
        help="Optional TXT file containing one line of text per segment; blank lines add a one-second pause.",
    )
    parser.add_argument(
        "--cuda-device",
        "--cuda_device",
        dest="cuda_device",
        default=None,
        help="Optional CUDA device index to expose to this process, for example 0 or 1.",
    )
    return parser


def apply_cuda_device(cuda_device: Optional[str]) -> None:
    if cuda_device is None:
        return

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    log(f"Using CUDA device index: {cuda_device}")


def load_segments_from_json(script_json_path: str) -> List[Any]:
    with open(script_json_path, "r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)

    if isinstance(payload, list):
        return [str(item) if not isinstance(item, dict) else item for item in payload]

    if isinstance(payload, dict):
        for key in ("segments", "items", "script"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    raise ValueError(f"Unsupported JSON structure in {script_json_path}")


def load_segments_from_file(script_path: str) -> List[str]:
    with open(script_path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def load_segments_for_processing(script_path: str) -> List[str]:
    if script_path.lower().endswith(".txt"):
        return load_segments_from_file(script_path)

    segments = load_segments_from_json(script_path)
    return [extract_segment_text(segment) or "" for segment in segments]


def extract_segment_text(segment: Any) -> Optional[str]:
    if isinstance(segment, str):
        return segment

    if isinstance(segment, dict):
        for key in ("text", "segment", "content", "sentence", "transcript", "value"):
            if key in segment and isinstance(segment[key], str):
                return segment[key]

    return None


def generate_high_quality_tts(
    text_to_speak: str,
    vprofile: str = "v2/en_speaker_7",
    output_filename: Optional[str] = None,
    return_audio_array: bool = False,
) -> Optional[np.ndarray]:
    raw_text = text_to_speak
    processed_text = clean_text_for_bark(raw_text)
    log(f"Speaking text: '{processed_text}'")

    processor, model = load_bark_model_components()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Process text and voice preset into tensors
    log("Processing text and voice preset tensors...")
    inputs = processor(text=[processed_text], voice_preset=vprofile)

    # Move all input tensors to the GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3. Generate audio directly from the model weights (bypasses pipeline validation)
    log("Generating audio waveform via native model execution...")
    with torch.no_grad():
        audio_outputs = model.generate(**inputs)

    audio_array = audio_outputs.cpu().numpy().squeeze()

    # 4. Post-process audio array
    sampling_rate = 24000
    normalized_audio = normalize_audio(audio_array)

    if return_audio_array:
        return normalized_audio

    if output_filename is None:
        output_filename = generate_output_filename()

    if output_filename:
        scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=normalized_audio)
        log(f"Saved optimized audio file to: {output_filename}")

    return None


def process_segments_to_audio(segments: List[str], voice_profile: str) -> np.ndarray:
    combined_audio_segments: List[np.ndarray] = []

    for index, segment in enumerate(segments, start=1):
        if not segment.strip():
            pause_samples = int(24000 * 1.0)
            combined_audio_segments.append(np.zeros(pause_samples, dtype=np.float32))
            log(f"Blank line {index}: adding 1 second pause")
            continue

        cleaned_text = clean_text_for_bark(segment)
        log(f"Generating segment {index} with voice profile {voice_profile}...")
        generated_audio = generate_high_quality_tts(cleaned_text, voice_profile, output_filename=None, return_audio_array=True)
        if generated_audio is not None:
            combined_audio_segments.append(generated_audio)

    return concatenate_audio_segments(combined_audio_segments)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    apply_cuda_device(args.cuda_device)

    if not torch.cuda.is_available():
        log("ERROR: CUDA is not available on this system or the selected CUDA device is not visible.")
        return

    log(
        "Starting TTS generation with parameters: "
        f"text_to_speak='{args.text_to_speak}', voice_profile='{args.voice_profile}', script_json='{args.script_json}', script_txt='{args.script_txt}', cuda_device='{args.cuda_device}'"
    )

    if args.script_json:
        segments = load_segments_for_processing(args.script_json)
        combined_audio = process_segments_to_audio(segments, args.voice_profile)

        if combined_audio.size:
            output_filename = generate_output_filename()
            scipy.io.wavfile.write(output_filename, rate=24000, data=combined_audio)
            log(f"Saved combined audio file to: {output_filename}")
    elif args.script_txt:
        segments = load_segments_for_processing(args.script_txt)

        combined_audio = process_segments_to_audio(segments, args.voice_profile)

        if combined_audio.size:
            output_filename = generate_output_filename()
            scipy.io.wavfile.write(output_filename, rate=24000, data=combined_audio)
            log(f"Saved combined audio file to: {output_filename}")
    else:
        text_to_use = args.text_to_speak or "you ain't axe me to says nuffin"
        generate_high_quality_tts(text_to_use, args.voice_profile)


if __name__ == "__main__":
    os.environ["SUNO_ENABLE_CPU_OFFLOAD"] = "True"

    try:
        main()
    finally:
        cleanup_bark_model_components()
