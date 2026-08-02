# barkTextTTS
Text-to-speech scripts for Bark and Kokoro-82M.

## Kokoro Gen: Quick Start

Install dependencies:

```powershell
pip install kokoro soundfile
```

Windows note:
Install espeak-ng (required by Kokoro for fallback pronunciation paths).

First successful run:

```powershell
python .\kokoro_gen.py --text-to-speak "Hello from Kokoro TTS" --voice af_heart --lang-code a
```

The script writes output to a random UUID filename ending in .wav.

## Kokoro Gen: All Run Scenarios

### 1) Speak a short inline string

```powershell
python .\kokoro_gen.py --text-to-speak "Sam McGee was from Tennessee" --voice af_heart --lang-code a
```

### 2) Speak multiline text from a PowerShell here-string

```powershell
$text = @"
There are strange things done in the midnight sun

By the men who moil for gold;

The Arctic trails have their secret tales

That would make your blood run cold;
"@

python .\kokoro_gen.py --text-to-speak $text --voice af_heart --lang-code a
```

### 3) Speak from a TXT script file

One line = one segment. Blank lines add a one-second pause.

```powershell
python .\kokoro_gen.py --script-txt .\transcripts\TheCremationofSamMcGee.txt --voice af_heart --lang-code a
```

### 4) Speak from a JSON script file

Accepted JSON shapes:
- A raw list of strings or objects
- An object containing one of: segments, items, or script (as a list)

```powershell
python .\kokoro_gen.py --script-json .\transcripts\TheCremationOfSamMcGee.json --voice af_heart --lang-code a
```

### 5) Select GPU (RTX 3060 = 0, RTX 3080 Ti = 1)

```powershell
python .\kokoro_gen.py --cuda-device 0 --text-to-speak "GPU-targeted run" --voice af_heart --lang-code a
```

### 6) Slow down or speed up speech

```powershell
python .\kokoro_gen.py --text-to-speak "Speaking slightly slower" --voice af_heart --lang-code a --speed 0.9
python .\kokoro_gen.py --text-to-speak "Speaking slightly faster" --voice af_heart --lang-code a --speed 1.1
```

### 7) Use alternate flag aliases

All of these are valid and map to the same destinations:

```powershell
python .\kokoro_gen.py --text_to_speak "Alias test" --voice-profile af_heart --lang_code a --cuda_device 0
```

## Command Reference (Kokoro)

| Parameter | Aliases | Purpose | Example |
|---|---|---|---|
| text_to_speak | --text-to-speak, --text_to_speak | Inline text input | --text-to-speak "Hello" |
| voice_profile | --voice, --voice-profile, --voice_profile | Kokoro voice selection | --voice af_heart |
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
python .\kokoro_gen.py --text-to-speak "hello" --voice af_heart --lang-code a
```

### Wrong flag for file type

If your file is .txt, use --script-txt.
If your file is .json, use --script-json.

### Bark voice passed to Kokoro

Bark voices like v2/en_speaker_8 are not Kokoro voices.
Use Kokoro voices such as af_heart.

## Bark Script (Legacy)

Bark is still available through cuda_gen.py:

```powershell
python .\cuda_gen.py --text-to-speak "Hello from Bark" --voice-profile v2/en_speaker_9
```
