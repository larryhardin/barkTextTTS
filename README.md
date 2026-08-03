# barkTextTTS
Text-to-speech scripts for Bark and Kokoro-82M.

## UI App (Recommended)

The project now includes a desktop UI in `ui_app.py` that drives both generators.

Launch:

```powershell
python .\ui_app.py
```

### Current UI Features

- Model selector: `Kokoro` (default) or `Bark`
- Dynamic form rendering per model
	- Kokoro uses `kokoro_gen.py` arguments
	- Bark uses `cuda_gen.py` arguments
- Input source modes
	- Inline text
	- Script file picker with `*.txt` and `*.json` filter
- Kokoro voice flow (cascading dropdowns)
	- Voice language -> Voice sex -> Voice
	- Display names remove trailing `_F` / `_M`
	- Display names are normalized to title-case
	- Only supported Kokoro-82M voices are shown
- CUDA device dropdown
	- Auto-detected from available GPUs
	- Includes `Auto` plus indexed GPU entries
- Model loading UX
	- Loading message + progress bar
	- 2-second hold after 100%, then hidden
- Generation UX
	- Modal popup with live log output and cancel button
	- Main UI is disabled while popup is open
	- Main UI is restored and refocused when popup closes
- Playback
	- `Play Generated File` button appears after successful generation

### Performance Notes

- The UI runs generation in-process to minimize time between runs.
- When settings are unchanged and only text changes, repeat generation is faster because the model stays warm in the running UI process.
- Cancel in this mode is cooperative (it does not hard-kill an external subprocess).

### Dependencies

Minimum Python packages used by this project:

```powershell
pip install numpy scipy torch transformers kokoro soundfile
```

Windows note:
Install `espeak-ng` for Kokoro pronunciation/fallback paths.

## Kokoro Gen: Quick Start

Install dependencies:

```powershell
pip install kokoro soundfile
```

Windows note:
Install espeak-ng (required by Kokoro for fallback pronunciation paths).

First successful run:

```powershell
python .\kokoro_gen.py --text-to-speak "Hello from Kokoro TTS" --voice BELLA --lang-code a
```

The script writes output as `VOICE_UUID.wav`.
Example: `BELLA_550e8400-e29b-41d4-a716-446655440000.wav`.

## Kokoro Gen: All Run Scenarios

### Voice Mapping Rules

The `--voice` parameter now maps to names defined in `KokoroVoices.py`.

- Matching is case-insensitive (`bella`, `BELLA`, and `BeLlA` all work)
- The input is normalized to uppercase for lookup
- Internal voice identifiers are still accepted (`af_bella` works)
- If no match is found, the script logs a warning and defaults to `BELLA` (`af_bella`)
- Unsupported voices are filtered/fallbacked to avoid missing voice files in `hexgrad/Kokoro-82M`

Examples:

```powershell
python .\kokoro_gen.py --text-to-speak "Name-based voice" --voice BELLA --lang-code a
python .\kokoro_gen.py --text-to-speak "Case-insensitive voice" --voice bella --lang-code a
python .\kokoro_gen.py --text-to-speak "Internal ID also works" --voice af_bella --lang-code a
python .\kokoro_gen.py --text-to-speak "Unknown voice falls back" --voice notARealVoice --lang-code a
```

### 1) Speak a short inline string

```powershell
python .\kokoro_gen.py --text-to-speak "Sam McGee was from Tennessee" --voice BELLA --lang-code a
```

### 2) Speak multiline text from a PowerShell here-string

```powershell
$text = @"
There are strange things done in the midnight sun

By the men who moil for gold;

The Arctic trails have their secret tales

That would make your blood run cold;
"@

python .\kokoro_gen.py --text-to-speak $text --voice BELLA --lang-code a
```

### 3) Speak from a TXT script file

One line = one segment. Blank lines add a one-second pause.

```powershell
python .\kokoro_gen.py --script-txt .\transcripts\TheCremationofSamMcGee.txt --voice BELLA --lang-code a
```

### 4) Speak from a JSON script file

Accepted JSON shapes:
- A raw list of strings or objects
- An object containing one of: segments, items, or script (as a list)

```powershell
python .\kokoro_gen.py --script-json .\transcripts\TheCremationOfSamMcGee.json --voice BELLA --lang-code a
```

### 5) Select GPU (RTX 3060 = 0, RTX 3080 Ti = 1)

```powershell
python .\kokoro_gen.py --cuda-device 0 --text-to-speak "GPU-targeted run" --voice BELLA --lang-code a
```

### 6) Slow down or speed up speech

```powershell
python .\kokoro_gen.py --text-to-speak "Speaking slightly slower" --voice BELLA --lang-code a --speed 0.9
python .\kokoro_gen.py --text-to-speak "Speaking slightly faster" --voice BELLA --lang-code a --speed 1.1
```

### 7) Use alternate flag aliases

All of these are valid and map to the same destinations:

```powershell
python .\kokoro_gen.py --text_to_speak "Alias test" --voice-profile BELLA --lang_code a --cuda_device 0
```

## Command Reference (Kokoro)

| Parameter | Aliases | Purpose | Example |
|---|---|---|---|
| text_to_speak | --text-to-speak, --text_to_speak | Inline text input | --text-to-speak "Hello" |
| voice_profile | --voice, --voice-profile, --voice_profile | Voice name from KokoroVoices.py (case-insensitive); unknown values default to BELLA | --voice BELLA |
| lang_code | --lang-code, --lang_code | Language code | --lang-code a |
| speed | --speed | Speech speed multiplier | --speed 1.0 |
| script_json | --script-json, --script_json | JSON segments input | --script-json .\x.json |
| script_txt | --script-txt, --script_txt | TXT line-based input | --script-txt .\x.txt |
| cuda_device | --cuda-device, --cuda_device | GPU index visibility | --cuda-device 0 |

## Common Failure Patterns

### Wrong dash style

Invalid:

```powershell
python .\kokoro_gen.py -text-to-speak "hello" -voice af_heart
```

Correct:

```powershell
python .\kokoro_gen.py --text-to-speak "hello" --voice BELLA --lang-code a
```

### Wrong flag for file type

If your file is .txt, use --script-txt.
If your file is .json, use --script-json.

### Bark voice passed to Kokoro

Bark voices like v2/en_speaker_8 are not Kokoro voices.
Use names from `KokoroVoices.py` such as `BELLA`, `HEART`, `SARAH`, `GEORGE`, or `EMMA`.

## Bark Script (Legacy)

Bark is still available through cuda_gen.py:

```powershell
python .\cuda_gen.py --text-to-speak "Hello from Bark" --voice-profile v2/en_speaker_9
```
