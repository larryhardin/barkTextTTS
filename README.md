# barkTextTTS
Simple program to generate speech using Suno's bark model.

## Quick Start

```powershell
python .\cuda_gen.py --text-to-speak "Hello from Bark TTS" --voice-profile v2/en_speaker_9
```

## Kokoro Quick Start

Install the Python package first:

```powershell
pip install kokoro soundfile
```

On Windows, install `espeak-ng` as well because Kokoro depends on it for English fallback and some language support.

```powershell
python .\kokoro_gen.py --text-to-speak "Hello from Kokoro TTS" --voice af_heart --lang-code a
```

## Run the script (PowerShell)

The script expects named flags, not positional text arguments.

### Option 1: Inline multiline text

```powershell
$text = @"
There are strange things done in the midnight sun

	By the men who moil for gold;

The Arctic trails have their secret tales

	That would make your blood run cold;

The Northern Lights have seen queer sights,

	But the queerest they ever did see

Was that night on the marge of Lake Lebarge

	I cremated Sam McGee.
"@

python .\cuda_gen.py --text-to-speak $text --voice-profile v2/en_speaker_9
```

### Option 2: Use a text file (recommended for longer scripts)

Create a file such as `poem.txt` with one line per segment.
Blank lines add a one-second pause between segments.

```powershell
python .\cuda_gen.py --script-txt .\poem.txt --voice-profile v2/en_speaker_9
```

## Common mistake

This will fail because `text_to_speak` is not a positional command:

```powershell
python cuda_gen.py text_to_speak "..."
```

Use `--text-to-speak "..."` instead.
