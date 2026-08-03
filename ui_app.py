import os
import queue
import re
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from KokoroVoices import KokoroVoices

_SUPPORTED_KOKORO_VOICE_IDS = {
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


@dataclass
class _GenerationJob:
    model: str
    source_kind: str
    text: Optional[str] = None
    script_path: Optional[str] = None
    output_filename: Optional[str] = None
    kokoro_voice_label: Optional[str] = None
    kokoro_voice_profile: Optional[str] = None
    kokoro_lang_code: Optional[str] = None
    kokoro_speed: Optional[float] = None
    bark_voice_profile: Optional[str] = None
    cuda_device: Optional[str] = None


class _QueueLogWriter:
    def __init__(self, output_queue: queue.Queue[Tuple[str, object]]) -> None:
        self.output_queue = output_queue
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.output_queue.put(("log", line + "\n"))

        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self.output_queue.put(("log", self._buffer))
            self._buffer = ""


def _display_voice_name(raw_voice_name: str) -> str:
    base_name = re.sub(r"_(F|M)$", "", raw_voice_name, flags=re.IGNORECASE)
    return base_name[:1].upper() + base_name[1:].lower() if base_name else raw_voice_name


def _voice_sex_from_name_or_id(raw_voice_name: str, raw_voice_id: str) -> Optional[str]:
    name_upper = raw_voice_name.upper()
    if name_upper.endswith("_F"):
        return "Female"
    if name_upper.endswith("_M"):
        return "Male"

    if len(raw_voice_id) >= 2:
        prefix = raw_voice_id[1].lower()
        if prefix == "f":
            return "Female"
        if prefix == "m":
            return "Male"

    return None


def build_kokoro_voice_catalog() -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    catalog: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}

    for group_name in dir(KokoroVoices):
        if group_name.startswith("_"):
            continue

        group = getattr(KokoroVoices, group_name)
        if not isinstance(group, type):
            continue

        language_bucket = {"Female": [], "Male": []}

        for voice_name in dir(group):
            if voice_name.startswith("_"):
                continue

            voice_value = getattr(group, voice_name)
            if not isinstance(voice_value, str):
                continue

            sex = _voice_sex_from_name_or_id(voice_name, voice_value)
            if sex is None:
                continue

            if voice_value.lower() not in _SUPPORTED_KOKORO_VOICE_IDS:
                continue

            display_name = _display_voice_name(voice_name)
            language_bucket[sex].append((display_name, voice_value))

        for sex in ("Female", "Male"):
            language_bucket[sex].sort(key=lambda item: item[0])

        if language_bucket["Female"] or language_bucket["Male"]:
            catalog[group_name] = language_bucket

    return dict(sorted(catalog.items(), key=lambda item: item[0]))


def build_cuda_device_options() -> Tuple[List[str], Dict[str, Optional[str]]]:
    try:
        import torch
    except ImportError:
        return ["Auto"], {"Auto": None}

    options = ["Auto"]
    mapping: Dict[str, Optional[str]] = {"Auto": None}

    if not torch.cuda.is_available():
        return options, mapping

    for index in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(index)
        label = f"{index}: {device_name}"
        options.append(label)
        mapping[label] = str(index)

    return options, mapping


class TTSApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Bark/Kokoro TTS UI")
        self.root.geometry("900x680")
        self.root.minsize(820, 600)

        self.workspace_dir = Path(__file__).resolve().parent
        self.kokoro_script = self.workspace_dir / "kokoro_gen.py"
        self.bark_script = self.workspace_dir / "cuda_gen.py"

        self.kokoro_voice_catalog = build_kokoro_voice_catalog()
        self.kokoro_languages = list(self.kokoro_voice_catalog.keys())
        self.cuda_device_options, self.cuda_device_lookup = build_cuda_device_options()
        self.bark_voices = [f"v2/en_speaker_{i}" for i in range(10)]

        self.model_var = tk.StringVar(value="Kokoro")
        self.source_mode_var = tk.StringVar(value="text")
        self.file_path_var = tk.StringVar(value="")

        default_language = "American" if "American" in self.kokoro_languages else (self.kokoro_languages[0] if self.kokoro_languages else "")
        self.kokoro_language_var = tk.StringVar(value=default_language)
        self.kokoro_sex_var = tk.StringVar(value="Female")
        self.kokoro_voice_display_var = tk.StringVar(value="")
        self.kokoro_voice_lookup: Dict[str, str] = {}
        self.kokoro_lang_var = tk.StringVar(value="a")
        self.kokoro_speed_var = tk.StringVar(value="1.0")
        self.kokoro_cuda_var = tk.StringVar(value="Auto")

        self.bark_voice_var = tk.StringVar(value="v2/en_speaker_7")
        self.bark_cuda_var = tk.StringVar(value="Auto")

        self.inline_text_widget: Optional[tk.Text] = None
        self.model_specific_frame: Optional[ttk.Frame] = None
        self.kokoro_language_combo: Optional[ttk.Combobox] = None
        self.kokoro_sex_combo: Optional[ttk.Combobox] = None
        self.kokoro_voice_combo: Optional[ttk.Combobox] = None
        self.kokoro_cuda_combo: Optional[ttk.Combobox] = None
        self.bark_cuda_combo: Optional[ttk.Combobox] = None

        self.loading_frame: Optional[ttk.Frame] = None
        self.loading_progress: Optional[ttk.Progressbar] = None
        self.loading_status_var = tk.StringVar(value="")
        self.loading_after_id: Optional[str] = None
        self.loading_hide_after_id: Optional[str] = None
        self.loading_step = 0

        self.start_button: Optional[ttk.Button] = None
        self.play_button: Optional[ttk.Button] = None
        self.last_generated_file: Optional[Path] = None

        self.popup: Optional[tk.Toplevel] = None
        self.popup_log_widget: Optional[tk.Text] = None
        self.popup_progress: Optional[ttk.Progressbar] = None
        self.popup_status_var = tk.StringVar(value="")
        self.cancel_button: Optional[ttk.Button] = None
        self.root_window_disabled = False
        self.widget_state_cache: Dict[str, object] = {}

        self.proc_thread: Optional[threading.Thread] = None
        self.proc_queue: queue.Queue[Tuple[str, object]] = queue.Queue()
        self.cancel_requested = False
        self.run_start_time = 0.0

        self._build_ui()
        self._render_model_fields()

    def _build_ui(self) -> None:
        self.main_container = ttk.Frame(self.root, padding=14)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(self.main_container, text="Text to Speech Generator", font=("Segoe UI", 16, "bold"))
        title.pack(anchor=tk.W, pady=(0, 10))

        model_row = ttk.Frame(self.main_container)
        model_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(model_row, text="Model:", width=18).pack(side=tk.LEFT)

        model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=["Kokoro", "Bark"],
            state="readonly",
            width=24,
        )
        model_combo.pack(side=tk.LEFT)
        model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        self.loading_frame = ttk.Frame(self.main_container)
        self.loading_frame.pack(fill=tk.X, pady=(0, 12))
        self.loading_frame.pack_forget()

        ttk.Label(self.loading_frame, textvariable=self.loading_status_var).pack(anchor=tk.W)
        self.loading_progress = ttk.Progressbar(self.loading_frame, maximum=100, mode="determinate")
        self.loading_progress.pack(fill=tk.X, pady=(3, 0))

        source_frame = ttk.LabelFrame(self.main_container, text="Input Source", padding=10)
        source_frame.pack(fill=tk.X, pady=(0, 10))

        radio_row = ttk.Frame(source_frame)
        radio_row.pack(fill=tk.X)

        ttk.Radiobutton(
            radio_row,
            text="Inline text",
            value="text",
            variable=self.source_mode_var,
            command=self._render_source_mode,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Radiobutton(
            radio_row,
            text="Script file (.txt or .json)",
            value="file",
            variable=self.source_mode_var,
            command=self._render_source_mode,
        ).pack(side=tk.LEFT)

        self.source_content_frame = ttk.Frame(source_frame)
        self.source_content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        model_group = ttk.LabelFrame(self.main_container, text="Model Settings", padding=10)
        model_group.pack(fill=tk.X, pady=(0, 10))
        self.model_specific_frame = ttk.Frame(model_group)
        self.model_specific_frame.pack(fill=tk.X)

        actions = ttk.Frame(self.main_container)
        actions.pack(fill=tk.X, pady=(4, 0))

        self.start_button = ttk.Button(actions, text="Start Generation", command=self._start_generation)
        self.start_button.pack(side=tk.LEFT)

        self.play_button = ttk.Button(actions, text="Play Generated File", command=self._play_generated, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=(10, 0))
        self.play_button.pack_forget()

        self._render_source_mode()

    def _clear_frame(self, frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _on_model_selected(self, _event: object = None) -> None:
        self._render_model_fields()
        self._start_load_animation()

    def _render_source_mode(self) -> None:
        self._clear_frame(self.source_content_frame)

        if self.source_mode_var.get() == "text":
            ttk.Label(self.source_content_frame, text="Text to speak:").pack(anchor=tk.W)
            self.inline_text_widget = tk.Text(self.source_content_frame, height=8, wrap=tk.WORD)
            self.inline_text_widget.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
            self.inline_text_widget.insert("1.0", "")
            return

        self.inline_text_widget = None
        row = ttk.Frame(self.source_content_frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Script file:", width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.file_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_script).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            self.source_content_frame,
            text="Choose a .txt or .json script file. The UI will map it to the right CLI argument.",
        ).pack(anchor=tk.W, pady=(8, 0))

    def _render_model_fields(self) -> None:
        if self.model_specific_frame is None:
            return

        self._clear_frame(self.model_specific_frame)
        model = self.model_var.get()

        if model == "Kokoro":
            self._render_kokoro_fields()
        else:
            self._render_bark_fields()

    def _render_kokoro_fields(self) -> None:
        row1 = ttk.Frame(self.model_specific_frame)
        row1.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row1, text="Voice language:", width=18).pack(side=tk.LEFT)
        self.kokoro_language_combo = ttk.Combobox(
            row1,
            textvariable=self.kokoro_language_var,
            values=self.kokoro_languages,
            state="readonly",
            width=30,
        )
        self.kokoro_language_combo.pack(side=tk.LEFT)
        self.kokoro_language_combo.bind("<<ComboboxSelected>>", self._on_kokoro_language_changed)

        row2 = ttk.Frame(self.model_specific_frame)
        row2.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row2, text="Voice sex:", width=18).pack(side=tk.LEFT)
        self.kokoro_sex_combo = ttk.Combobox(
            row2,
            textvariable=self.kokoro_sex_var,
            values=["Female", "Male"],
            state="readonly",
            width=30,
        )
        self.kokoro_sex_combo.pack(side=tk.LEFT)
        self.kokoro_sex_combo.bind("<<ComboboxSelected>>", self._on_kokoro_sex_changed)

        row3 = ttk.Frame(self.model_specific_frame)
        row3.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row3, text="Voice:", width=18).pack(side=tk.LEFT)
        self.kokoro_voice_combo = ttk.Combobox(
            row3,
            textvariable=self.kokoro_voice_display_var,
            values=[],
            state="readonly",
            width=30,
        )
        self.kokoro_voice_combo.pack(side=tk.LEFT)

        row4 = ttk.Frame(self.model_specific_frame)
        row4.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row4, text="Language code:", width=18).pack(side=tk.LEFT)
        ttk.Entry(row4, textvariable=self.kokoro_lang_var, width=12).pack(side=tk.LEFT)

        row5 = ttk.Frame(self.model_specific_frame)
        row5.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row5, text="Speed:", width=18).pack(side=tk.LEFT)
        ttk.Entry(row5, textvariable=self.kokoro_speed_var, width=12).pack(side=tk.LEFT)

        row6 = ttk.Frame(self.model_specific_frame)
        row6.pack(fill=tk.X)
        ttk.Label(row6, text="CUDA device (opt):", width=18).pack(side=tk.LEFT)
        self.kokoro_cuda_combo = ttk.Combobox(
            row6,
            textvariable=self.kokoro_cuda_var,
            values=self.cuda_device_options,
            state="readonly",
            width=30,
        )
        self.kokoro_cuda_combo.pack(side=tk.LEFT)
        if self.kokoro_cuda_var.get() not in self.cuda_device_options:
            self.kokoro_cuda_var.set("Auto")

        self._refresh_kokoro_sex_options()
        self._refresh_kokoro_voice_options()

    def _on_kokoro_language_changed(self, _event: object = None) -> None:
        self._refresh_kokoro_sex_options()
        self._refresh_kokoro_voice_options()

    def _on_kokoro_sex_changed(self, _event: object = None) -> None:
        self._refresh_kokoro_voice_options()

    def _refresh_kokoro_sex_options(self) -> None:
        language = self.kokoro_language_var.get().strip()
        language_data = self.kokoro_voice_catalog.get(language, {"Female": [], "Male": []})

        available_sexes = [sex for sex in ("Female", "Male") if language_data.get(sex)]
        if not available_sexes:
            available_sexes = ["Female", "Male"]

        if self.kokoro_sex_combo is not None:
            self.kokoro_sex_combo.configure(values=available_sexes)

        current_sex = self.kokoro_sex_var.get().strip()
        if current_sex not in available_sexes:
            self.kokoro_sex_var.set(available_sexes[0])

    def _refresh_kokoro_voice_options(self) -> None:
        language = self.kokoro_language_var.get().strip()
        sex = self.kokoro_sex_var.get().strip()
        language_data = self.kokoro_voice_catalog.get(language, {"Female": [], "Male": []})
        voice_entries = language_data.get(sex, [])

        display_values = [display_name for display_name, _ in voice_entries]
        self.kokoro_voice_lookup = {display_name: raw_name for display_name, raw_name in voice_entries}

        if self.kokoro_voice_combo is not None:
            self.kokoro_voice_combo.configure(values=display_values)

        current_display = self.kokoro_voice_display_var.get().strip()
        if current_display not in self.kokoro_voice_lookup:
            self.kokoro_voice_display_var.set(display_values[0] if display_values else "")

    def _render_bark_fields(self) -> None:
        row1 = ttk.Frame(self.model_specific_frame)
        row1.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row1, text="Voice profile:", width=18).pack(side=tk.LEFT)
        voice_combo = ttk.Combobox(
            row1,
            textvariable=self.bark_voice_var,
            values=self.bark_voices,
            state="readonly",
            width=30,
        )
        voice_combo.pack(side=tk.LEFT)

        row2 = ttk.Frame(self.model_specific_frame)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="CUDA device (opt):", width=18).pack(side=tk.LEFT)
        self.bark_cuda_combo = ttk.Combobox(
            row2,
            textvariable=self.bark_cuda_var,
            values=self.cuda_device_options,
            state="readonly",
            width=30,
        )
        self.bark_cuda_combo.pack(side=tk.LEFT)
        if self.bark_cuda_var.get() not in self.cuda_device_options:
            self.bark_cuda_var.set("Auto")

    def _browse_script(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select script file",
            filetypes=[("Text or JSON", "*.txt *.json"), ("Text files", "*.txt"), ("JSON files", "*.json")],
        )
        if selected:
            self.file_path_var.set(selected)

    def _start_load_animation(self) -> None:
        if self.loading_frame is None or self.loading_progress is None:
            return

        if self.loading_after_id is not None:
            self.root.after_cancel(self.loading_after_id)
            self.loading_after_id = None
        if self.loading_hide_after_id is not None:
            self.root.after_cancel(self.loading_hide_after_id)
            self.loading_hide_after_id = None

        self.loading_step = 0
        self.loading_status_var.set(f"Loading {self.model_var.get()} model...")
        self.loading_progress["value"] = 0
        self.loading_frame.pack(fill=tk.X, pady=(0, 12))
        self._tick_load_animation()

    def _tick_load_animation(self) -> None:
        if self.loading_progress is None:
            return

        self.loading_step += 5
        self.loading_progress["value"] = min(self.loading_step, 100)
        if self.loading_step < 100:
            self.loading_after_id = self.root.after(60, self._tick_load_animation)
            return

        self.loading_status_var.set(f"{self.model_var.get()} model ready.")
        self.loading_hide_after_id = self.root.after(2000, self._hide_loading)

    def _hide_loading(self) -> None:
        if self.loading_frame is not None:
            self.loading_frame.pack_forget()
        self.loading_after_id = None
        self.loading_hide_after_id = None

    def _collect_command(self) -> _GenerationJob:
        model = self.model_var.get()

        source_mode = self.source_mode_var.get()
        if source_mode == "text":
            if self.inline_text_widget is None:
                raise ValueError("Text input is not available.")
            text_value = self.inline_text_widget.get("1.0", tk.END).strip()
            if text_value:
                source_kind = "text"
                source_value = text_value
            else:
                raise ValueError("Please enter text to speak.")
        else:
            selected_path = self.file_path_var.get().strip()
            if not selected_path:
                raise ValueError("Please select a .txt or .json script file.")
            if not os.path.isfile(selected_path):
                raise ValueError("The selected script path does not exist.")

            lower = selected_path.lower()
            if lower.endswith(".txt"):
                source_kind = "file"
                source_value = selected_path
            elif lower.endswith(".json"):
                source_kind = "file"
                source_value = selected_path
            else:
                raise ValueError("Only .txt and .json files are supported.")

        if model == "Kokoro":
            selected_display = self.kokoro_voice_display_var.get().strip()
            voice = self.kokoro_voice_lookup.get(selected_display)
            if not voice:
                raise ValueError("Please select a Kokoro voice.")
            lang_code = self.kokoro_lang_var.get().strip() or "a"
            speed = self.kokoro_speed_var.get().strip() or "1.0"

            try:
                float(speed)
            except ValueError as exc:
                raise ValueError("Speed must be a valid number.") from exc
            return _GenerationJob(
                model="Kokoro",
                source_kind=source_kind,
                text=source_value if source_kind == "text" else None,
                script_path=source_value if source_kind == "file" else None,
                kokoro_voice_label=selected_display,
                kokoro_voice_profile=voice,
                kokoro_lang_code=lang_code,
                kokoro_speed=float(speed),
                cuda_device=self.cuda_device_lookup.get(self.kokoro_cuda_var.get().strip()),
            )
        else:
            voice = self.bark_voice_var.get().strip()
            return _GenerationJob(
                model="Bark",
                source_kind=source_kind,
                text=source_value if source_kind == "text" else None,
                script_path=source_value if source_kind == "file" else None,
                bark_voice_profile=voice,
                cuda_device=self.cuda_device_lookup.get(self.bark_cuda_var.get().strip()),
            )

    def _start_generation(self) -> None:
        if self.proc_thread is not None and self.proc_thread.is_alive():
            messagebox.showinfo("Generation Running", "A generation job is already running.")
            return

        try:
            job = self._collect_command()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.last_generated_file = None
        if self.play_button is not None:
            self.play_button.configure(state=tk.DISABLED)
            self.play_button.pack_forget()

        self.run_start_time = time.time()
        self.cancel_requested = False
        self._open_progress_popup(job.model)
        self._set_ui_interaction(enabled=False)

        self.proc_thread = threading.Thread(target=self._run_generation_worker, args=(job,), daemon=True)
        self.proc_thread.start()
        self.root.after(120, self._poll_proc_queue)

    def _open_progress_popup(self, model: str) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Generation In Progress")
        popup.geometry("760x420")
        popup.transient(self.root)
        popup.grab_set()
        popup.protocol("WM_DELETE_WINDOW", lambda: None)
        self.popup = popup

        try:
            self.root.attributes("-disabled", True)
            self.root_window_disabled = True
        except tk.TclError:
            self.root_window_disabled = False

        frame = ttk.Frame(popup, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        self.popup_status_var.set(f"{model} generation is running...")
        ttk.Label(frame, textvariable=self.popup_status_var, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        self.popup_progress = ttk.Progressbar(frame, mode="indeterminate")
        self.popup_progress.pack(fill=tk.X, pady=(8, 8))
        self.popup_progress.start(12)

        log_label = ttk.Label(frame, text="Log output:")
        log_label.pack(anchor=tk.W)

        self.popup_log_widget = tk.Text(frame, height=16, wrap=tk.WORD)
        self.popup_log_widget.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.cancel_button = ttk.Button(frame, text="Cancel", command=self._cancel_generation)
        self.cancel_button.pack(anchor=tk.E)

        self._append_popup_log("Starting process...\n")

    def _append_popup_log(self, message: str) -> None:
        if self.popup_log_widget is None:
            return
        self.popup_log_widget.insert(tk.END, message)
        self.popup_log_widget.see(tk.END)

    def _run_generation_worker(self, job: _GenerationJob) -> None:
        output_path: Optional[Path] = None
        log_writer = _QueueLogWriter(self.proc_queue)

        try:
            with redirect_stdout(log_writer), redirect_stderr(log_writer):
                if job.cuda_device is not None:
                    if job.model == "Kokoro":
                        import kokoro_gen as kokoro_module

                        kokoro_module.apply_cuda_device(job.cuda_device)
                    else:
                        import cuda_gen as bark_module

                        bark_module.apply_cuda_device(job.cuda_device)

                if job.model == "Kokoro":
                    import kokoro_gen as kokoro_module

                    output_path = self._run_kokoro_job(kokoro_module, job)
                else:
                    import cuda_gen as bark_module

                    output_path = self._run_bark_job(bark_module, job)
        except Exception as exc:  # pragma: no cover - UI error path
            self.proc_queue.put(("error", str(exc)))
            return
        finally:
            log_writer.flush()

        self.proc_queue.put(("done", (0, self.cancel_requested, output_path)))

    def _run_kokoro_job(self, kokoro_module: object, job: _GenerationJob) -> Optional[Path]:
        output_path = self.workspace_dir / kokoro_module.generate_output_filename(job.kokoro_voice_label or "KOKORO")

        if job.source_kind == "text":
            kokoro_module.generate_kokoro_tts(
                job.text or "",
                voice_profile=job.kokoro_voice_profile or "af_heart",
                voice_label=job.kokoro_voice_label or "KOKORO",
                lang_code=job.kokoro_lang_code or "a",
                speed=job.kokoro_speed or 1.0,
                output_filename=str(output_path),
            )
        else:
            segments = kokoro_module.load_segments_for_processing(job.script_path or "")
            combined_audio = kokoro_module.process_segments_to_audio(
                segments,
                job.kokoro_voice_profile or "af_heart",
                job.kokoro_lang_code or "a",
                job.kokoro_speed or 1.0,
            )

            if combined_audio.size:
                import scipy.io.wavfile

                scipy.io.wavfile.write(output_path, rate=kokoro_module.SAMPLE_RATE, data=combined_audio)

        return output_path if output_path.exists() else None

    def _run_bark_job(self, bark_module: object, job: _GenerationJob) -> Optional[Path]:
        output_path = self.workspace_dir / bark_module.generate_output_filename()

        if job.source_kind == "text":
            bark_module.generate_high_quality_tts(
                job.text or "",
                job.bark_voice_profile or "v2/en_speaker_7",
                output_filename=str(output_path),
            )
        else:
            segments = bark_module.load_segments_for_processing(job.script_path or "")
            combined_audio = bark_module.process_segments_to_audio(segments, job.bark_voice_profile or "v2/en_speaker_7")

            if combined_audio.size:
                import scipy.io.wavfile

                scipy.io.wavfile.write(output_path, rate=24000, data=combined_audio)

        return output_path if output_path.exists() else None

    def _poll_proc_queue(self) -> None:
        saw_done = False
        while True:
            try:
                event, payload = self.proc_queue.get_nowait()
            except queue.Empty:
                break

            if event == "log":
                self._append_popup_log(str(payload))
            elif event == "error":
                self._finish_generation_with_error(str(payload))
                saw_done = True
            elif event == "done":
                code, was_cancelled, out_path = payload  # type: ignore[misc]
                self._finish_generation(int(code), bool(was_cancelled), out_path)
                saw_done = True

        if not saw_done:
            self.root.after(120, self._poll_proc_queue)

    def _cancel_generation(self) -> None:
        self.cancel_requested = True
        if self.popup_status_var.get():
            self.popup_status_var.set("Cancel requested. Stopping generation...")

        self._append_popup_log(
            "\nCancel is now cooperative because generation runs inside the UI process. The current job will finish, but the loaded model stays warm for the next run.\n"
        )

    def _finish_generation_with_error(self, message: str) -> None:
        self._append_popup_log(f"\nFailed to start process: {message}\n")
        self._finish_generation(return_code=1, cancelled=False, output_file=None)

    def _finish_generation(self, return_code: int, cancelled: bool, output_file: Optional[Path]) -> None:
        if self.popup_progress is not None:
            self.popup_progress.stop()

        status_message = "Generation complete."
        if cancelled:
            status_message = "Generation cancelled."
        elif return_code != 0:
            status_message = f"Generation failed (exit code {return_code})."

        self.popup_status_var.set(status_message)
        self._append_popup_log(f"\n{status_message}\n")

        if self.cancel_button is not None:
            self.cancel_button.configure(text="Close", command=self._close_popup)

        if self.popup is not None and self.popup.winfo_exists():
            try:
                self.popup.grab_release()
            except tk.TclError:
                pass

        if not cancelled and return_code == 0 and output_file is not None and output_file.exists():
            self.last_generated_file = output_file
            if self.play_button is not None:
                self.play_button.pack(side=tk.LEFT, padx=(10, 0))
            self._append_popup_log(f"Generated file: {output_file}\n")

    def _close_popup(self) -> None:
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.grab_release()
            self.popup.destroy()
        self.popup = None
        self.popup_log_widget = None
        self.popup_progress = None
        self.cancel_button = None
        if self.root_window_disabled:
            try:
                self.root.attributes("-disabled", False)
            except tk.TclError:
                pass
            self.root_window_disabled = False
        self._set_ui_interaction(enabled=True)
        try:
            self.root.deiconify()
            self.root.update_idletasks()
            self.root.lift()
            self.root.focus_force()
        except tk.TclError:
            pass

    def _set_widget_interaction(self, widget: tk.Widget, enabled: bool) -> None:
        widget_type = widget.winfo_class()

        if widget_type in {"Button", "TButton", "Radiobutton", "TRadiobutton", "Entry", "TEntry", "Combobox", "TCombobox", "Text"}:
            try:
                if widget_type == "Text":
                    if enabled:
                        original_state = self.widget_state_cache.get(str(widget), tk.NORMAL)
                        widget.configure(state=original_state)
                    else:
                        self.widget_state_cache[str(widget)] = widget.cget("state")
                        widget.configure(state=tk.DISABLED)
                elif widget_type in {"Combobox", "TCombobox"}:
                    if enabled:
                        original_state = self.widget_state_cache.get(str(widget), "readonly")
                        widget.configure(state=original_state)
                    else:
                        self.widget_state_cache[str(widget)] = widget.cget("state")
                        widget.configure(state="disabled")
                else:
                    if enabled:
                        original_state = self.widget_state_cache.get(str(widget), tk.NORMAL)
                        widget.configure(state=original_state)
                    else:
                        self.widget_state_cache[str(widget)] = widget.cget("state")
                        widget.configure(state=tk.DISABLED)
            except tk.TclError:
                pass

        for child in widget.winfo_children():
            self._set_widget_interaction(child, enabled)

    def _set_ui_interaction(self, enabled: bool) -> None:
        if hasattr(self, "main_container"):
            self._set_widget_interaction(self.main_container, enabled)

        if enabled:
            if self.last_generated_file is not None and self.last_generated_file.exists():
                if self.play_button is not None:
                    self.play_button.pack(side=tk.LEFT, padx=(10, 0))
                    self.play_button.configure(state=tk.NORMAL)
        elif self.play_button is not None:
            self.play_button.configure(state=tk.DISABLED)
            self.play_button.pack_forget()

    def _find_latest_generated_wav(self, after_ts: float) -> Optional[Path]:
        candidates: List[Path] = []
        for wav_file in self.workspace_dir.glob("*.wav"):
            try:
                if wav_file.stat().st_mtime >= after_ts:
                    candidates.append(wav_file)
            except OSError:
                continue

        if not candidates:
            return None

        return max(candidates, key=lambda item: item.stat().st_mtime)

    def _play_generated(self) -> None:
        if self.last_generated_file is None or not self.last_generated_file.exists():
            messagebox.showerror("No File", "No generated WAV file is available.")
            return

        try:
            import winsound

            winsound.PlaySound(str(self.last_generated_file), winsound.SND_ASYNC | winsound.SND_FILENAME)
        except Exception as exc:  # pragma: no cover - platform/runtime path
            messagebox.showerror("Playback Error", f"Could not play audio: {exc}")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")

    app = TTSApp(root)

    def on_root_close() -> None:
        is_generation_running = app.proc_thread is not None and app.proc_thread.is_alive()
        if is_generation_running:
            if not messagebox.askyesno("Exit", "Generation is running. Cancel and exit?"):
                return
            app._cancel_generation()

        if app.popup is not None:
            app._close_popup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_root_close)
    root.mainloop()


if __name__ == "__main__":
    main()