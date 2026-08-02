import argparse
import json
import re
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.io.wavfile
from KokoroVoices import KokoroVoices

SAMPLE_RATE = 24000
_KOKORO_PIPELINES: Dict[str, Any] = {}
_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
_SUPPORTED_VOICE_IDS = {
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
}


def _build_voice_maps() -> tuple[Dict[str, str], Dict[str, str]]:
    name_to_id: Dict[str, str] = {}
    id_to_id: Dict[str, str] = {}

    for group_name in dir(KokoroVoices):
        if group_name.startswith("_"):
            continue
        group = getattr(KokoroVoices, group_name)
        if not isinstance(group, type):
            continue

        for voice_name in dir(group):
            if voice_name.startswith("_"):
                continue
            voice_id = getattr(group, voice_name)
            if not isinstance(voice_id, str):
                continue

            name_to_id[voice_name.upper()] = voice_id
            id_to_id[voice_id.upper()] = voice_id

    return name_to_id, id_to_id


_VOICE_NAME_TO_ID, _VOICE_ID_TO_ID = _build_voice_maps()
_DEFAULT_VOICE_NAME = "BELLA"


def resolve_voice_selection(voice_parameter: Optional[str]) -> tuple[str, str]:
    default_voice_id = _VOICE_NAME_TO_ID.get(_DEFAULT_VOICE_NAME, "af_bella")
    if not voice_parameter:
        return _DEFAULT_VOICE_NAME, default_voice_id

    voice_key = str(voice_parameter).upper()
    if voice_key in _VOICE_NAME_TO_ID:
        return voice_key, _VOICE_NAME_TO_ID[voice_key]

    if voice_key in _VOICE_ID_TO_ID:
        resolved_voice_id = _VOICE_ID_TO_ID[voice_key]
        if resolved_voice_id.lower() in _SUPPORTED_VOICE_IDS:
            return voice_key, resolved_voice_id
        log(
            f"Voice '{voice_parameter}' resolves to unsupported Kokoro voice id '{resolved_voice_id}'. Defaulting to {_DEFAULT_VOICE_NAME} ({default_voice_id})."
        )
        return _DEFAULT_VOICE_NAME, default_voice_id

    log(f"Voice '{voice_parameter}' not found. Defaulting to {_DEFAULT_VOICE_NAME} ({default_voice_id}).")
    return _DEFAULT_VOICE_NAME, default_voice_id


def clean_text_for_kokoro(text: str) -> str:
    text = re.sub(r"\b(CUDA|TTS|AI|GPU)\b", lambda match: " ".join(match.group(1)), text)

    replacements = {
        r"\bDr\b\.?": "Doctor",
        r"\bMr\b\.?": "Mister",
        r"\bMs\b\.?": "Missus",
        r"\betc\b\.?": "et cetera",
        r"\bvs\b\.?": "versus",
        "McGee": "Mick Ghee",
        "Laberge": "la-Barge",
    }
    for pattern, replacement in replacements.items():
        text = re.compile(pattern, re.IGNORECASE).sub(replacement, text)

    return text


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.size == 0:
        return np.array([], dtype=np.float32)

    if np.max(np.abs(audio_data)) == 0:
        return audio_data.astype(np.float32)

    normalized = audio_data / np.max(np.abs(audio_data))
    return normalized.astype(np.float32)


def log(message: str) -> None:
    print(message, flush=True)


def generate_output_filename(voice_label: str) -> str:
    safe_voice_label = re.sub(r"[^A-Z0-9_-]", "_", voice_label.upper())
    return f"{safe_voice_label}_{uuid.uuid4()}.wav"


def concatenate_audio_segments(audio_segments: List[np.ndarray]) -> np.ndarray:
    if not audio_segments:
        return np.array([], dtype=np.float32)

    return np.concatenate([segment.astype(np.float32, copy=False) for segment in audio_segments], axis=0)


def load_kokoro_pipeline(lang_code: str) -> Any:
    if lang_code in _KOKORO_PIPELINES:
        return _KOKORO_PIPELINES[lang_code]

    try:
        from kokoro import KPipeline  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "Kokoro is not installed. Install it with `pip install kokoro soundfile` and install espeak-ng on Windows."
        ) from exc

    log(f"Loading Kokoro pipeline for language code '{lang_code}'...")
    pipeline = KPipeline(lang_code=lang_code, repo_id=_KOKORO_REPO_ID)
    _KOKORO_PIPELINES[lang_code] = pipeline
    return pipeline


def clear_kokoro_pipelines() -> None:
    _KOKORO_PIPELINES.clear()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Kokoro-82M TTS audio from text, a TXT script, or a JSON script")
    parser.add_argument(
        "--text-to-speak",
        "--text_to_speak",
        dest="text_to_speak",
        default=None,
        help="Text to synthesize. If omitted, the default fallback text is used only when no script file is provided.",
    )
    parser.add_argument(
        "--voice",
        "--voice-profile",
        "--voice_profile",
        dest="voice_profile",
        default=_DEFAULT_VOICE_NAME,
        help="Kokoro voice name from KokoroVoices.py, for example BELLA.",
    )
    parser.add_argument(
        "--lang-code",
        "--lang_code",
        dest="lang_code",
        default="a",
        help="Kokoro language code. 'a' is American English and 'b' is British English.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier. 1.0 is normal speed.",
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

    import os

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


def generate_kokoro_tts(
    text_to_speak: str,
    voice_profile: str = "af_heart",
    voice_label: str = _DEFAULT_VOICE_NAME,
    lang_code: str = "a",
    speed: float = 1.0,
    output_filename: Optional[str] = None,
    return_audio_array: bool = False,
) -> Optional[np.ndarray]:
    processed_text = clean_text_for_kokoro(text_to_speak)
    log(f"Speaking text: '{processed_text}'")

    pipeline = load_kokoro_pipeline(lang_code)
    audio_segments: List[np.ndarray] = []

    log("Generating audio waveform with Kokoro...")
    for _, _, audio in pipeline(processed_text, voice=voice_profile, speed=speed, split_pattern=r"\n+"):
        audio_segments.append(np.asarray(audio, dtype=np.float32))

    normalized_audio = normalize_audio(concatenate_audio_segments(audio_segments))

    if return_audio_array:
        return normalized_audio

    if output_filename is None:
        output_filename = generate_output_filename(voice_label)

    if normalized_audio.size and output_filename:
        scipy.io.wavfile.write(output_filename, rate=SAMPLE_RATE, data=normalized_audio)
        log(f"Saved Kokoro audio file to: {output_filename}")
    else:
        log("No audio was generated.")

    return None


def process_segments_to_audio(segments: List[str], voice_profile: str, lang_code: str, speed: float) -> np.ndarray:
    combined_audio_segments: List[np.ndarray] = []

    for index, segment in enumerate(segments, start=1):
        if not segment.strip():
            pause_samples = int(SAMPLE_RATE * 1.0)
            combined_audio_segments.append(np.zeros(pause_samples, dtype=np.float32))
            log(f"Blank line {index}: adding 1 second pause")
            continue

        log(f"Generating segment {index} with voice {voice_profile} and language code {lang_code}...")
        generated_audio = generate_kokoro_tts(
            segment,
            voice_profile=voice_profile,
            lang_code=lang_code,
            speed=speed,
            output_filename=None,
            return_audio_array=True,
        )
        if generated_audio is not None and generated_audio.size:
            combined_audio_segments.append(generated_audio)

    return concatenate_audio_segments(combined_audio_segments)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    apply_cuda_device(args.cuda_device)
    resolved_voice_label, resolved_voice_profile = resolve_voice_selection(args.voice_profile)

    log(
        "Starting Kokoro TTS generation with parameters: "
        f"text_to_speak='{args.text_to_speak}', voice_profile='{args.voice_profile}', resolved_voice_label='{resolved_voice_label}', resolved_voice='{resolved_voice_profile}', lang_code='{args.lang_code}', speed='{args.speed}', script_json='{args.script_json}', script_txt='{args.script_txt}', cuda_device='{args.cuda_device}'"
    )

    if args.script_json:
        segments = load_segments_for_processing(args.script_json)
        combined_audio = process_segments_to_audio(segments, resolved_voice_profile, args.lang_code, args.speed)

        if combined_audio.size:
            output_filename = generate_output_filename(resolved_voice_label)
            scipy.io.wavfile.write(output_filename, rate=SAMPLE_RATE, data=combined_audio)
            log(f"Saved combined audio file to: {output_filename}")
    elif args.script_txt:
        segments = load_segments_for_processing(args.script_txt)
        combined_audio = process_segments_to_audio(segments, resolved_voice_profile, args.lang_code, args.speed)

        if combined_audio.size:
            output_filename = generate_output_filename(resolved_voice_label)
            scipy.io.wavfile.write(output_filename, rate=SAMPLE_RATE, data=combined_audio)
            log(f"Saved combined audio file to: {output_filename}")
    else:
        text_to_use = args.text_to_speak or "you ain't axe me to says nuffin"
        generate_kokoro_tts(text_to_use, resolved_voice_profile, resolved_voice_label, args.lang_code, args.speed)


if __name__ == "__main__":
    try:
        main()
    finally:
        clear_kokoro_pipelines()