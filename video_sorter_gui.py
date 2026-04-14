#!/usr/bin/env python3
"""
Tkinter GUI for the ROCm female-face video sorter.

This GUI launches the backend sorter script, lets you choose source and
destination folders, and streams log output into the window.
"""

from __future__ import annotations

import queue
import json
import os
import subprocess
import sys
import threading
import tkinter as tk
import uuid
from pathlib import Path
from typing import Any
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk


APP_TITLE = "Female Face Video Sorter"
BACKEND_SCRIPT = Path(__file__).with_name("sort_videos_by_female_faces_gpu.py")
SETTINGS_FILE = Path(__file__).with_name("video_sorter_settings.json")
REVIEW_STATE_FILE = ".review_queue/review_state.json"
RESULT_JSON_PREFIX = "[RESULT_JSON] "
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PROFILE_TO_BACKEND = {
    "Fast": "fast",
    "Balanced": "balanced",
    "High Accuracy": "high_accuracy",
}
BACKEND_TO_PROFILE = {value: key for key, value in PROFILE_TO_BACKEND.items()}


def parse_result_json_line(line: str) -> dict | None:
    stripped = line.strip()
    if not stripped.startswith(RESULT_JSON_PREFIX):
        return None
    raw_json = stripped[len(RESULT_JSON_PREFIX) :].strip()
    if not raw_json:
        return None
    try:
        payload = json.loads(raw_json)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def format_result_row(payload: dict) -> tuple[str, str, str, str, str]:
    video = str(payload.get("video", "")).strip()
    video_name = str(payload.get("video_name", "")).strip() or (Path(video).name if video else "")
    decision = str(payload.get("decision_label", "")).strip()
    try:
        confidence = f"{float(payload.get('confidence_score', 0.0)):.3f}"
    except Exception:
        confidence = "0.000"
    reason_summary = str(payload.get("reason_summary", "")).strip()

    tags_value = payload.get("reason_tags", [])
    if isinstance(tags_value, list):
        tags = ", ".join(str(tag) for tag in tags_value if str(tag).strip())
    else:
        tags = str(tags_value).strip()
    return video_name, decision, confidence, reason_summary, tags


class VideoSorterGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("940x760")
        self.root.minsize(820, 620)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.reader_thread: threading.Thread | None = None
        self.stop_flag_path: Path | None = None
        self.stop_requested = False

        self.source_var = tk.StringVar()
        self.destination_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=True)
        self.include_generated_var = tk.BooleanVar(value=False)
        self.include_generated_button_var = tk.StringVar()
        self.review_mode_var = tk.BooleanVar(value=False)
        self.review_mode_button_var = tk.StringVar()
        self.learning_enabled_var = tk.BooleanVar(value=True)
        self.learning_button_var = tk.StringVar()
        self.live_trace_var = tk.BooleanVar(value=False)
        self.live_trace_button_var = tk.StringVar()
        self.max_seconds_var = tk.StringVar(value="60")
        self.sample_every_var = tk.StringVar(value="2.0")
        self.stabilization_var = tk.StringVar(value="8.0")
        self.resize_width_var = tk.StringVar(value="960")
        self.max_workers_var = tk.StringVar(value="1")
        self.profile_var = tk.StringVar(value="Balanced")
        self.status_var = tk.StringVar(value="Ready")
        self.runtime_var = tk.StringVar(value="Runtime: checking...")
        self.live_status_var = tk.StringVar(value="Live: idle")
        self.progress_status_var = tk.StringVar(value="Progress: 0/0 videos")
        self.review_status_var = tk.StringVar(value="Review Queue: 0 pending")

        self.review_window: tk.Toplevel | None = None
        self.review_items: list[dict] = []
        self.review_index = 0
        self.review_preview_image: ImageTk.PhotoImage | None = None
        self.review_counter_var = tk.StringVar(value="0/0")
        self.review_predicted_var = tk.StringVar(value="")
        self.review_confidence_var = tk.StringVar(value="")
        self.review_reason_var = tk.StringVar(value="")
        self.review_suggested_var = tk.StringVar(value="")
        self.review_memory_var = tk.StringVar(value="")
        self.review_path_var = tk.StringVar(value="")
        self.results_tree: ttk.Treeview | None = None
        self.identity_window: tk.Toplevel | None = None
        self.identity_tree: ttk.Treeview | None = None
        self.identity_rows: dict[str, dict] = {}
        self.learning_stats_window: tk.Toplevel | None = None
        self.learning_stats_tree: ttk.Treeview | None = None
        self.duplicate_window: tk.Toplevel | None = None
        self.duplicate_groups_tree: ttk.Treeview | None = None
        self.duplicate_items_tree: ttk.Treeview | None = None
        self.duplicate_groups_map: dict[str, dict] = {}

        self._load_settings()
        self._update_include_generated_button_text()
        self._update_review_mode_button_text()
        self._update_learning_button_text()
        self._update_live_trace_button_text()
        self._build_ui()
        self._update_reprocess_button_state()
        self._refresh_runtime_status()
        self._refresh_review_queue_status()
        self.root.after(100, self._poll_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=2)
        self.root.rowconfigure(4, weight=1)

        top = ttk.Frame(self.root, padding=14)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Source Folder").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 8))
        ttk.Entry(top, textvariable=self.source_var).grid(row=0, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(top, text="Browse", command=self._choose_source).grid(row=0, column=2, padx=(10, 0), pady=(0, 8))

        ttk.Label(top, text="Destination Folder").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(0, 8))
        ttk.Entry(top, textvariable=self.destination_var).grid(row=1, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(top, text="Browse", command=self._choose_destination).grid(
            row=1, column=2, padx=(10, 0), pady=(0, 8)
        )

        runtime_bar = ttk.Frame(self.root, padding=(14, 0, 14, 10))
        runtime_bar.grid(row=1, column=0, sticky="ew")
        runtime_bar.columnconfigure(0, weight=1)

        ttk.Label(runtime_bar, textvariable=self.runtime_var).grid(row=0, column=0, sticky="w")
        ttk.Button(runtime_bar, text="Refresh Runtime", command=self._refresh_runtime_status).grid(
            row=0, column=1, padx=(10, 0)
        )

        options = ttk.LabelFrame(self.root, text="Options", padding=14)
        options.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        for idx in range(4):
            options.columnconfigure(idx, weight=1)

        ttk.Checkbutton(
            options,
            text="Scan child folders recursively",
            variable=self.recursive_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Button(
            options,
            textvariable=self.include_generated_button_var,
            command=self._toggle_include_generated,
        ).grid(row=0, column=2, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Button(
            options,
            textvariable=self.live_trace_button_var,
            command=self._toggle_live_trace,
        ).grid(row=1, column=2, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Button(
            options,
            textvariable=self.review_mode_button_var,
            command=self._toggle_review_mode,
        ).grid(row=5, column=2, columnspan=2, sticky="ew", pady=(10, 2))

        ttk.Button(
            options,
            textvariable=self.learning_button_var,
            command=self._toggle_learning_mode,
        ).grid(row=6, column=2, columnspan=2, sticky="ew", pady=(6, 2))

        ttk.Label(options, text="Max Seconds").grid(row=1, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.max_seconds_var, width=12).grid(row=2, column=0, sticky="ew", padx=(0, 10))

        ttk.Label(options, text="Sample Every (sec)").grid(row=1, column=1, sticky="w")
        ttk.Entry(options, textvariable=self.sample_every_var, width=12).grid(row=2, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(options, text="Preset Profile").grid(row=3, column=0, sticky="w")
        self.profile_combo = ttk.Combobox(
            options,
            values=list(PROFILE_TO_BACKEND.keys()),
            textvariable=self.profile_var,
            state="readonly",
        )
        self.profile_combo.grid(row=4, column=0, sticky="ew", padx=(0, 10))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_selected)

        ttk.Button(options, text="Apply Profile", command=self._apply_selected_profile_to_fields).grid(
            row=4, column=1, sticky="ew", padx=(0, 10)
        )

        ttk.Label(options, text="Stabilization (sec)").grid(row=3, column=2, sticky="w")
        ttk.Entry(options, textvariable=self.stabilization_var, width=12).grid(
            row=4, column=2, sticky="ew", padx=(0, 10)
        )

        ttk.Label(options, text="Resize Width").grid(row=3, column=3, sticky="w")
        ttk.Entry(options, textvariable=self.resize_width_var, width=12).grid(row=4, column=3, sticky="ew")

        ttk.Label(options, text="Max Workers").grid(row=5, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(options, textvariable=self.max_workers_var, width=12).grid(row=6, column=0, sticky="ew", pady=(0, 2))

        ttk.Label(
            options,
            text="Source videos are discovered by extension and sorted from child folders too when recursive scan is enabled.",
            wraplength=760,
            foreground="#444444",
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(12, 0))

        log_frame = ttk.LabelFrame(self.root, text="Logs", padding=10)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=20, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)
        ttk.Label(log_frame, textvariable=self.live_status_var).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(log_frame, textvariable=self.progress_status_var).grid(row=2, column=0, sticky="w", pady=(4, 0))

        results_frame = ttk.LabelFrame(self.root, text="Results This Run", padding=10)
        results_frame.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("video", "decision", "confidence", "why", "tags"),
            show="headings",
            height=7,
        )
        self.results_tree.heading("video", text="Video")
        self.results_tree.heading("decision", text="Decision")
        self.results_tree.heading("confidence", text="Confidence")
        self.results_tree.heading("why", text="Why")
        self.results_tree.heading("tags", text="Tags")
        self.results_tree.column("video", width=210, anchor="w")
        self.results_tree.column("decision", width=110, anchor="w")
        self.results_tree.column("confidence", width=90, anchor="center")
        self.results_tree.column("why", width=280, anchor="w")
        self.results_tree.column("tags", width=220, anchor="w")
        self.results_tree.grid(row=0, column=0, sticky="nsew")

        result_scroll = ttk.Scrollbar(results_frame, command=self.results_tree.yview)
        result_scroll.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=result_scroll.set)

        footer = ttk.Frame(self.root, padding=(14, 0, 14, 14))
        footer.grid(row=5, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        footer.columnconfigure(2, weight=1)
        footer.columnconfigure(3, weight=1)
        footer.columnconfigure(4, weight=1)

        self.start_button = ttk.Button(footer, text="Start Sorting", command=self._start_sorting)
        self.start_button.grid(row=0, column=0, padx=(0, 8))

        self.stop_button = ttk.Button(footer, text="Stop", command=self._stop_sorting, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="w")

        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, value=0)
        self.progress.grid(row=0, column=2, sticky="ew", padx=12)

        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=3, sticky="e")

        self.open_review_button = ttk.Button(footer, text="Open Review Queue", command=self._open_review_queue, state="disabled")
        self.open_review_button.grid(row=1, column=0, padx=(0, 8), pady=(8, 0), sticky="w")
        self.identity_tools_button = ttk.Button(footer, text="Identity Tools", command=self._open_identity_tools)
        self.identity_tools_button.grid(row=1, column=1, padx=(0, 8), pady=(8, 0), sticky="w")
        self.duplicate_finder_button = ttk.Button(footer, text="Duplicate Finder", command=self._open_duplicate_finder)
        self.duplicate_finder_button.grid(row=1, column=2, padx=(0, 8), pady=(8, 0), sticky="w")
        self.reprocess_uncertain_button = ttk.Button(
            footer,
            text="Reprocess Uncertain",
            command=self._start_reprocess_uncertain,
            state="disabled",
        )
        self.reprocess_uncertain_button.grid(row=1, column=3, padx=(0, 8), pady=(8, 0), sticky="w")
        ttk.Label(footer, textvariable=self.review_status_var).grid(row=1, column=4, sticky="w", pady=(8, 0))

    def _choose_source(self) -> None:
        selected = filedialog.askdirectory(title="Choose Source Folder")
        if selected:
            self.source_var.set(selected)
            self._save_settings()

    def _choose_destination(self) -> None:
        selected = filedialog.askdirectory(title="Choose Destination Folder")
        if selected:
            self.destination_var.set(selected)
            self._save_settings()
            self._refresh_review_queue_status()
            self._update_reprocess_button_state()

    def _destination_ready(self) -> bool:
        destination_text = self.destination_var.get().strip()
        return bool(destination_text and Path(destination_text).expanduser().is_dir())

    def _update_reprocess_button_state(self) -> None:
        if not hasattr(self, "reprocess_uncertain_button"):
            return
        state = "normal" if self.process is None and self._destination_ready() else "disabled"
        self.reprocess_uncertain_button.configure(state=state)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_results_table(self) -> None:
        if self.results_tree is None:
            return
        for item_id in self.results_tree.get_children():
            self.results_tree.delete(item_id)

    def _append_result_row(self, payload: dict) -> None:
        if self.results_tree is None:
            return
        self.results_tree.insert("", "end", values=format_result_row(payload))

    def _update_include_generated_button_text(self) -> None:
        if self.include_generated_var.get():
            self.include_generated_button_var.set("Generated Folders: Included")
        else:
            self.include_generated_button_var.set("Generated Folders: Skipped")

    def _update_review_mode_button_text(self) -> None:
        if self.review_mode_var.get():
            self.review_mode_button_var.set("Review Mode: On")
        else:
            self.review_mode_button_var.set("Review Mode: Off")

    def _update_learning_button_text(self) -> None:
        if self.learning_enabled_var.get():
            self.learning_button_var.set("Learning: On")
        else:
            self.learning_button_var.set("Learning: Off")

    def _update_live_trace_button_text(self) -> None:
        if self.live_trace_var.get():
            self.live_trace_button_var.set("Live Preview: On")
        else:
            self.live_trace_button_var.set("Live Preview: Off")

    def _toggle_include_generated(self) -> None:
        self.include_generated_var.set(not self.include_generated_var.get())
        self._update_include_generated_button_text()
        self._save_settings()

    def _toggle_review_mode(self) -> None:
        self.review_mode_var.set(not self.review_mode_var.get())
        self._update_review_mode_button_text()
        self._save_settings()

    def _toggle_learning_mode(self) -> None:
        self.learning_enabled_var.set(not self.learning_enabled_var.get())
        self._update_learning_button_text()
        self._save_settings()

    def _toggle_live_trace(self) -> None:
        self.live_trace_var.set(not self.live_trace_var.get())
        self._update_live_trace_button_text()
        self._save_settings()

    def _on_profile_selected(self, event: Any = None) -> None:
        self._apply_selected_profile_to_fields()

    def _apply_selected_profile_to_fields(self) -> None:
        selected = self.profile_var.get().strip()
        if selected not in PROFILE_TO_BACKEND:
            self.profile_var.set("Balanced")
            selected = "Balanced"

        if selected == "Fast":
            self.max_seconds_var.set("40")
            self.sample_every_var.set("2.5")
            self.stabilization_var.set("6.0")
            self.resize_width_var.set("720")
            self.max_workers_var.set("2")
        elif selected == "High Accuracy":
            self.max_seconds_var.set("90")
            self.sample_every_var.set("1.0")
            self.stabilization_var.set("10.0")
            self.resize_width_var.set("1152")
            self.max_workers_var.set("1")
        else:
            self.max_seconds_var.set("60")
            self.sample_every_var.set("2.0")
            self.stabilization_var.set("8.0")
            self.resize_width_var.set("960")
            self.max_workers_var.set("1")
        self._save_settings()

    def _load_settings(self) -> None:
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return

        self.source_var.set(str(data.get("source_folder", "")))
        self.destination_var.set(str(data.get("destination_folder", "")))
        self.recursive_var.set(bool(data.get("recursive", True)))
        self.include_generated_var.set(bool(data.get("include_generated_folders", False)))
        self.review_mode_var.set(bool(data.get("review_mode", False)))
        self.learning_enabled_var.set(bool(data.get("learning_enabled", True)))
        self.live_trace_var.set(bool(data.get("live_trace", False)))
        self._update_include_generated_button_text()
        self._update_review_mode_button_text()
        self._update_learning_button_text()
        self._update_live_trace_button_text()
        self.max_seconds_var.set(str(data.get("max_seconds", "60")))
        self.sample_every_var.set(str(data.get("sample_every_sec", "2.0")))
        self.stabilization_var.set(str(data.get("stabilization_seconds", "8.0")))
        self.resize_width_var.set(str(data.get("resize_width", "960")))
        self.max_workers_var.set(str(data.get("max_workers", "1")))
        profile_backend = str(data.get("profile", "balanced")).strip().lower().replace("-", "_")
        self.profile_var.set(BACKEND_TO_PROFILE.get(profile_backend, "Balanced"))

    def _save_settings(self) -> None:
        data = {
            "source_folder": self.source_var.get().strip(),
            "destination_folder": self.destination_var.get().strip(),
            "recursive": self.recursive_var.get(),
            "include_generated_folders": self.include_generated_var.get(),
            "review_mode": self.review_mode_var.get(),
            "learning_enabled": self.learning_enabled_var.get(),
            "live_trace": self.live_trace_var.get(),
            "max_seconds": self.max_seconds_var.get().strip(),
            "sample_every_sec": self.sample_every_var.get().strip(),
            "stabilization_seconds": self.stabilization_var.get().strip(),
            "resize_width": self.resize_width_var.get().strip(),
            "max_workers": self.max_workers_var.get().strip(),
            "profile": PROFILE_TO_BACKEND.get(self.profile_var.get().strip(), "balanced"),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"[WARN] Failed to save settings: {exc}\n")

    def _refresh_runtime_status(self) -> None:
        if not BACKEND_SCRIPT.exists():
            self.runtime_var.set("Runtime: backend script not found")
            return

        try:
            command = [
                sys.executable,
                str(BACKEND_SCRIPT),
                "--input-dir",
                ".",
                "--output-dir",
                ".",
                "--print-runtime-json",
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(BACKEND_SCRIPT.parent),
                check=True,
            )
            info = json.loads(completed.stdout.strip())
            acceleration = info.get("acceleration", "Unknown")
            reason = info.get("reason", "")
            device_names = info.get("device_names") or []
            if device_names:
                device_text = ", ".join(device_names)
                self.runtime_var.set(f"Runtime: {acceleration} | {device_text}")
            else:
                self.runtime_var.set(f"Runtime: {acceleration} | {reason}")
        except Exception as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                detail = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()[-1]
                self.runtime_var.set(f"Runtime: check failed | {detail}")
            else:
                self.runtime_var.set(f"Runtime: check failed ({exc})")

    def _parse_numeric_options(self) -> tuple[int, float, float, int, int] | None:
        try:
            max_seconds = int(float(self.max_seconds_var.get().strip()))
            sample_every = float(self.sample_every_var.get().strip())
            stabilization = float(self.stabilization_var.get().strip())
            resize_width = int(float(self.resize_width_var.get().strip()))
            max_workers = int(float(self.max_workers_var.get().strip()))
        except ValueError:
            messagebox.showerror(APP_TITLE, "Numeric options must contain valid numbers.")
            return None

        if max_seconds <= 0 or sample_every <= 0 or stabilization <= 0 or resize_width <= 0 or max_workers <= 0:
            messagebox.showerror(APP_TITLE, "Numeric options must be greater than zero.")
            return None
        return max_seconds, sample_every, stabilization, resize_width, max_workers

    def _prepare_stop_flag(self) -> None:
        self.stop_flag_path = BACKEND_SCRIPT.parent / f".stop_{uuid.uuid4().hex}.flag"
        self.stop_requested = False
        self._clear_stop_flag()

    def _build_backend_command_base(
        self,
        destination: Path,
        max_seconds: int,
        sample_every: float,
        stabilization: float,
        resize_width: int,
        max_workers: int,
    ) -> list[str]:
        assert self.stop_flag_path is not None
        command = [
            sys.executable,
            str(BACKEND_SCRIPT),
            "--output-dir",
            str(destination.resolve()),
            "--profile",
            PROFILE_TO_BACKEND.get(self.profile_var.get().strip(), "balanced"),
            "--max-seconds",
            str(max_seconds),
            "--sample-every-sec",
            str(sample_every),
            "--stabilization-seconds",
            str(stabilization),
            "--resize-width",
            str(resize_width),
            "--max-workers",
            str(max_workers),
            "--stop-flag-file",
            str(self.stop_flag_path),
        ]
        if not self.learning_enabled_var.get():
            command.append("--no-learning-enabled")
        if self.live_trace_var.get():
            command.append("--live-trace")
        return command

    def _validate_inputs(self) -> list[str] | None:
        source_text = self.source_var.get().strip()
        destination_text = self.destination_var.get().strip()
        source = Path(source_text).expanduser()
        destination = Path(destination_text).expanduser()

        if not source.is_dir():
            messagebox.showerror(APP_TITLE, "Please choose a valid source folder.")
            return None
        if not destination_text:
            messagebox.showerror(APP_TITLE, "Please choose a destination folder.")
            return None
        if not BACKEND_SCRIPT.exists():
            messagebox.showerror(APP_TITLE, f"Backend script not found:\n{BACKEND_SCRIPT}")
            return None

        parsed = self._parse_numeric_options()
        if parsed is None:
            return None
        max_seconds, sample_every, stabilization, resize_width, max_workers = parsed

        self._prepare_stop_flag()
        command = self._build_backend_command_base(
            destination,
            max_seconds=max_seconds,
            sample_every=sample_every,
            stabilization=stabilization,
            resize_width=resize_width,
            max_workers=max_workers,
        )
        command.extend(["--input-dir", str(source.resolve())])
        if not self.recursive_var.get():
            command.append("--no-recursive")
        if self.include_generated_var.get():
            command.append("--include-generated-folders")
        if self.review_mode_var.get():
            command.append("--review-mode")
        return command

    def _validate_reprocess_uncertain_inputs(self) -> list[str] | None:
        destination_text = self.destination_var.get().strip()
        destination = Path(destination_text).expanduser()
        if not destination_text:
            messagebox.showerror(APP_TITLE, "Please choose a destination folder.")
            return None
        if not destination.is_dir():
            messagebox.showerror(APP_TITLE, "Destination folder does not exist.")
            return None
        if not BACKEND_SCRIPT.exists():
            messagebox.showerror(APP_TITLE, f"Backend script not found:\n{BACKEND_SCRIPT}")
            return None

        parsed = self._parse_numeric_options()
        if parsed is None:
            return None
        max_seconds, sample_every, stabilization, resize_width, max_workers = parsed

        self._prepare_stop_flag()
        command = self._build_backend_command_base(
            destination,
            max_seconds=max_seconds,
            sample_every=sample_every,
            stabilization=stabilization,
            resize_width=resize_width,
            max_workers=max_workers,
        )
        command.append("--reprocess-uncertain-only")
        return command

    def _clear_stop_flag(self) -> None:
        if self.stop_flag_path is None:
            return
        try:
            if self.stop_flag_path.exists():
                self.stop_flag_path.unlink()
        except Exception:
            pass

    def _run_backend_json(self, extra_args: list[str]) -> dict:
        destination_text = self.destination_var.get().strip()
        if not destination_text:
            raise RuntimeError("Destination folder is required to access review queue.")

        destination = Path(destination_text).expanduser().resolve()
        command = [
            sys.executable,
            str(BACKEND_SCRIPT),
            "--output-dir",
            str(destination),
            *extra_args,
        ]
        if not self.learning_enabled_var.get():
            command.append("--no-learning-enabled")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(BACKEND_SCRIPT.parent),
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        payload: dict | None = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    payload = parsed
                    break
            except Exception:
                continue

        if payload is None:
            detail = stderr or stdout or f"Backend exited with code {completed.returncode}"
            raise RuntimeError(detail)

        if completed.returncode != 0:
            raise RuntimeError(str(payload.get("error", payload)))
        return payload

    def _fetch_review_state(self) -> dict:
        return self._run_backend_json(["--review-list-json"])

    def _fetch_identity_state(self) -> dict:
        return self._run_backend_json(["--identity-list-json"])

    def _fetch_learning_summary(self) -> dict:
        return self._run_backend_json(["--learning-summary-json"])

    def _fetch_duplicate_scan(self) -> dict:
        return self._run_backend_json(["--duplicates-scan-json"])

    def _apply_duplicate_move(self, paths: list[str], action_label: str) -> dict | None:
        if not paths:
            messagebox.showinfo(APP_TITLE, "No duplicate paths selected.")
            return None
        payload_json = json.dumps({"paths": paths})
        try:
            payload = self._run_backend_json(["--duplicates-apply-json", payload_json])
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Duplicate move failed:\n{exc}")
            return None

        if not bool(payload.get("ok", False)):
            messagebox.showerror(APP_TITLE, f"Duplicate move failed:\n{payload.get('error', payload)}")
            return None

        details = payload.get("details", {})
        moved_count = int(details.get("moved_count", 0) or 0) if isinstance(details, dict) else 0
        skipped_count = int(details.get("skipped_count", 0) or 0) if isinstance(details, dict) else 0
        error_count = int(details.get("error_count", 0) or 0) if isinstance(details, dict) else 0
        self._append_log(
            f"[DUPLICATES] {action_label}: moved={moved_count} skipped={skipped_count} errors={error_count}\n"
        )
        self._refresh_duplicate_finder()
        self._refresh_review_queue_status()
        return payload

    def _build_duplicate_window(self) -> None:
        if self.duplicate_window is not None and self.duplicate_window.winfo_exists():
            return

        win = tk.Toplevel(self.root)
        win.title(f"{APP_TITLE} - Duplicate Finder")
        win.geometry("1040x760")
        win.minsize(920, 620)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        self.duplicate_window = win

        body = ttk.Frame(win, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, text="Duplicate Groups").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(body, text="Duplicate Clips In Selected Group(s)").grid(row=0, column=1, sticky="w", pady=(0, 6))

        groups_tree = ttk.Treeview(
            body,
            columns=("group", "type", "primary", "count"),
            show="headings",
            selectmode="extended",
            height=18,
        )
        groups_tree.heading("group", text="Group")
        groups_tree.heading("type", text="Type")
        groups_tree.heading("primary", text="Primary")
        groups_tree.heading("count", text="Duplicate Count")
        groups_tree.column("group", width=90, anchor="w")
        groups_tree.column("type", width=80, anchor="center")
        groups_tree.column("primary", width=300, anchor="w")
        groups_tree.column("count", width=120, anchor="center")
        groups_tree.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.duplicate_groups_tree = groups_tree

        groups_scroll = ttk.Scrollbar(body, command=groups_tree.yview)
        groups_scroll.grid(row=1, column=0, sticky="nse", padx=(0, 8))
        groups_tree.configure(yscrollcommand=groups_scroll.set)
        groups_tree.bind("<<TreeviewSelect>>", self._on_duplicate_group_select)

        items_tree = ttk.Treeview(
            body,
            columns=("path", "folder"),
            show="headings",
            selectmode="extended",
            height=18,
        )
        items_tree.heading("path", text="Path")
        items_tree.heading("folder", text="Folder")
        items_tree.column("path", width=420, anchor="w")
        items_tree.column("folder", width=160, anchor="w")
        items_tree.grid(row=1, column=1, sticky="nsew")
        self.duplicate_items_tree = items_tree

        items_scroll = ttk.Scrollbar(body, command=items_tree.yview)
        items_scroll.grid(row=1, column=1, sticky="nse")
        items_tree.configure(yscrollcommand=items_scroll.set)

        controls = ttk.Frame(win, padding=(12, 0, 12, 12))
        controls.grid(row=1, column=0, sticky="ew")
        for idx in range(5):
            controls.columnconfigure(idx, weight=1)

        ttk.Button(controls, text="Scan", command=self._refresh_duplicate_finder).grid(row=0, column=0, padx=4, sticky="ew")
        ttk.Button(controls, text="Move Selected", command=self._move_selected_duplicates).grid(
            row=0, column=1, padx=4, sticky="ew"
        )
        ttk.Button(controls, text="Move All Duplicates", command=self._move_all_duplicates).grid(
            row=0, column=2, padx=4, sticky="ew"
        )
        ttk.Button(controls, text="Refresh", command=self._refresh_duplicate_finder).grid(
            row=0, column=3, padx=4, sticky="ew"
        )
        ttk.Button(controls, text="Close", command=self._close_duplicate_window).grid(row=0, column=4, padx=4, sticky="ew")

        win.protocol("WM_DELETE_WINDOW", self._close_duplicate_window)

    def _close_duplicate_window(self) -> None:
        if self.duplicate_window is not None and self.duplicate_window.winfo_exists():
            self.duplicate_window.destroy()
        self.duplicate_window = None
        self.duplicate_groups_tree = None
        self.duplicate_items_tree = None
        self.duplicate_groups_map = {}

    def _refresh_duplicate_finder(self) -> None:
        groups_tree = self.duplicate_groups_tree
        items_tree = self.duplicate_items_tree
        if groups_tree is None or items_tree is None:
            return

        try:
            payload = self._fetch_duplicate_scan()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to scan duplicates:\n{exc}")
            return

        groups = payload.get("groups", [])
        if not isinstance(groups, list):
            groups = []

        self.duplicate_groups_map = {}
        for item_id in groups_tree.get_children():
            groups_tree.delete(item_id)
        for item_id in items_tree.get_children():
            items_tree.delete(item_id)

        for group in groups:
            if not isinstance(group, dict):
                continue
            group_id = str(group.get("group_id", "")).strip()
            match_type = str(group.get("match_type", "")).strip()
            primary_path = str(group.get("primary_path", "")).strip()
            duplicates = group.get("duplicate_paths", [])
            if not group_id or not isinstance(duplicates, list):
                continue
            self.duplicate_groups_map[group_id] = group
            groups_tree.insert(
                "",
                "end",
                iid=group_id,
                values=(group_id, match_type, Path(primary_path).name if primary_path else "", len(duplicates)),
            )

        self._append_log(f"[DUPLICATES] scanned groups={len(self.duplicate_groups_map)}\n")

    def _on_duplicate_group_select(self, event: Any = None) -> None:
        groups_tree = self.duplicate_groups_tree
        items_tree = self.duplicate_items_tree
        if groups_tree is None or items_tree is None:
            return
        for item_id in items_tree.get_children():
            items_tree.delete(item_id)

        selected = list(groups_tree.selection())
        for group_id in selected:
            group = self.duplicate_groups_map.get(group_id)
            if not isinstance(group, dict):
                continue
            duplicate_paths = group.get("duplicate_paths", [])
            if not isinstance(duplicate_paths, list):
                continue
            for path_text in duplicate_paths:
                path_str = str(path_text)
                folder = Path(path_str).parent.name if path_str else ""
                iid = f"{group_id}::{path_str}"
                items_tree.insert("", "end", iid=iid, values=(path_str, folder))

    def _selected_duplicate_paths(self) -> list[str]:
        items_tree = self.duplicate_items_tree
        if items_tree is None:
            return []
        paths: list[str] = []
        for item_id in items_tree.selection():
            values = items_tree.item(item_id, "values")
            if not values:
                continue
            path_text = str(values[0]).strip()
            if path_text:
                paths.append(path_text)
        return paths

    def _move_selected_duplicates(self) -> None:
        paths = self._selected_duplicate_paths()
        if not paths:
            messagebox.showinfo(APP_TITLE, "Select one or more duplicate clips first.")
            return
        if not messagebox.askyesno(APP_TITLE, f"Move {len(paths)} selected duplicate clips to Duplicates folder?"):
            return
        self._apply_duplicate_move(paths, "move_selected")

    def _move_all_duplicates(self) -> None:
        paths: list[str] = []
        for group in self.duplicate_groups_map.values():
            if not isinstance(group, dict):
                continue
            duplicate_paths = group.get("duplicate_paths", [])
            if not isinstance(duplicate_paths, list):
                continue
            for path_text in duplicate_paths:
                path_str = str(path_text).strip()
                if path_str:
                    paths.append(path_str)
        deduped = sorted(set(paths))
        if not deduped:
            messagebox.showinfo(APP_TITLE, "No duplicates found to move.")
            return
        if not messagebox.askyesno(APP_TITLE, f"Move all {len(deduped)} duplicate clips to Duplicates folder?"):
            return
        self._apply_duplicate_move(deduped, "move_all")

    def _open_duplicate_finder(self) -> None:
        if self.process is not None:
            messagebox.showinfo(APP_TITLE, "Please wait until sorting finishes.")
            return
        self._build_duplicate_window()
        self._refresh_duplicate_finder()
        if self.duplicate_window is not None:
            self.duplicate_window.deiconify()
            self.duplicate_window.lift()
            self.duplicate_window.focus_force()

    def _selected_identity_names(self) -> list[str]:
        tree = self.identity_tree
        if tree is None:
            return []
        names: list[str] = []
        for item_id in tree.selection():
            values = tree.item(item_id, "values")
            if not values:
                continue
            name = str(values[0]).strip()
            if name:
                names.append(name)
        return names

    def _build_identity_window(self) -> None:
        if self.identity_window is not None and self.identity_window.winfo_exists():
            return

        win = tk.Toplevel(self.root)
        win.title(f"{APP_TITLE} - Identity Tools")
        win.geometry("980x700")
        win.minsize(860, 560)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        self.identity_window = win

        body = ttk.Frame(win, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        tree = ttk.Treeview(
            body,
            columns=("folder", "videos", "locked", "samples", "last_used"),
            show="headings",
            selectmode="extended",
        )
        tree.heading("folder", text="Folder")
        tree.heading("videos", text="Videos")
        tree.heading("locked", text="Locked")
        tree.heading("samples", text="Memory Samples")
        tree.heading("last_used", text="Last Used")
        tree.column("folder", width=300, anchor="w")
        tree.column("videos", width=90, anchor="center")
        tree.column("locked", width=90, anchor="center")
        tree.column("samples", width=120, anchor="center")
        tree.column("last_used", width=260, anchor="w")
        tree.grid(row=0, column=0, sticky="nsew")
        self.identity_tree = tree

        scroll = ttk.Scrollbar(body, command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)

        controls = ttk.Frame(win, padding=(12, 0, 12, 12))
        controls.grid(row=1, column=0, sticky="ew")
        for idx in range(7):
            controls.columnconfigure(idx, weight=1)

        ttk.Button(controls, text="Merge", command=self._identity_merge).grid(row=0, column=0, padx=4, sticky="ew")
        ttk.Button(controls, text="Split", command=self._identity_split).grid(row=0, column=1, padx=4, sticky="ew")
        ttk.Button(controls, text="Lock", command=self._identity_lock).grid(row=0, column=2, padx=4, sticky="ew")
        ttk.Button(controls, text="Unlock", command=self._identity_unlock).grid(row=0, column=3, padx=4, sticky="ew")
        ttk.Button(controls, text="Learning Stats", command=self._open_learning_stats).grid(
            row=0, column=4, padx=4, sticky="ew"
        )
        ttk.Button(controls, text="Refresh", command=self._refresh_identity_tools).grid(row=0, column=5, padx=4, sticky="ew")
        ttk.Button(controls, text="Close", command=self._close_identity_window).grid(row=0, column=6, padx=4, sticky="ew")

        win.protocol("WM_DELETE_WINDOW", self._close_identity_window)

    def _close_identity_window(self) -> None:
        if self.identity_window is not None and self.identity_window.winfo_exists():
            self.identity_window.destroy()
        self.identity_window = None
        self.identity_tree = None
        self.identity_rows = {}
        self._close_learning_stats_window()

    def _build_learning_stats_window(self) -> None:
        if self.learning_stats_window is not None and self.learning_stats_window.winfo_exists():
            return
        win = tk.Toplevel(self.root)
        win.title(f"{APP_TITLE} - Learning Stats")
        win.geometry("960x520")
        win.minsize(820, 420)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        self.learning_stats_window = win

        body = ttk.Frame(win, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        tree = ttk.Treeview(
            body,
            columns=("label", "positive", "negative", "consistency", "adaptive", "trend", "corrected"),
            show="headings",
            selectmode="browse",
        )
        tree.heading("label", text="Identity")
        tree.heading("positive", text="Positive")
        tree.heading("negative", text="Negative")
        tree.heading("consistency", text="Consistency")
        tree.heading("adaptive", text="Adaptive Auto")
        tree.heading("trend", text="Trend")
        tree.heading("corrected", text="Last Corrected")
        tree.column("label", width=260, anchor="w")
        tree.column("positive", width=90, anchor="center")
        tree.column("negative", width=90, anchor="center")
        tree.column("consistency", width=110, anchor="center")
        tree.column("adaptive", width=110, anchor="center")
        tree.column("trend", width=120, anchor="center")
        tree.column("corrected", width=220, anchor="w")
        tree.grid(row=0, column=0, sticky="nsew")
        self.learning_stats_tree = tree

        scroll = ttk.Scrollbar(body, command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)

        controls = ttk.Frame(win, padding=(12, 0, 12, 12))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        ttk.Button(controls, text="Refresh", command=self._refresh_learning_stats).grid(row=0, column=0, padx=4, sticky="ew")
        ttk.Button(controls, text="Close", command=self._close_learning_stats_window).grid(row=0, column=1, padx=4, sticky="ew")

        win.protocol("WM_DELETE_WINDOW", self._close_learning_stats_window)

    def _close_learning_stats_window(self) -> None:
        if self.learning_stats_window is not None and self.learning_stats_window.winfo_exists():
            self.learning_stats_window.destroy()
        self.learning_stats_window = None
        self.learning_stats_tree = None

    def _refresh_learning_stats(self) -> None:
        tree = self.learning_stats_tree
        if tree is None:
            return
        try:
            payload = self._fetch_learning_summary()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to load learning summary:\n{exc}")
            return
        items = payload.get("items", [])
        if not isinstance(items, list):
            items = []

        for item_id in tree.get_children():
            tree.delete(item_id)
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            positive = int(item.get("positive_feedback_count", 0) or 0)
            negative = int(item.get("negative_feedback_count", 0) or 0)
            consistency = float(item.get("correction_consistency_score", 0.0) or 0.0)
            adaptive = float(item.get("adaptive_auto_threshold", 0.0) or 0.0)
            trend = str(item.get("recent_trend", "")).strip()
            corrected = str(item.get("last_corrected_at", "")).strip()
            tree.insert(
                "",
                "end",
                values=(label, positive, negative, f"{consistency:.3f}", f"{adaptive:.3f}", trend, corrected),
            )

    def _open_learning_stats(self) -> None:
        self._build_learning_stats_window()
        self._refresh_learning_stats()
        if self.learning_stats_window is not None:
            self.learning_stats_window.deiconify()
            self.learning_stats_window.lift()
            self.learning_stats_window.focus_force()

    def _refresh_identity_tools(self) -> None:
        tree = self.identity_tree
        if tree is None:
            return
        try:
            payload = self._fetch_identity_state()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to load identities:\n{exc}")
            return
        identities = payload.get("identities", [])
        if not isinstance(identities, list):
            identities = []

        self.identity_rows = {}
        for item_id in tree.get_children():
            tree.delete(item_id)

        for item in identities:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            videos = int(item.get("video_count", 0) or 0)
            locked = bool(item.get("locked", False))
            samples = int(item.get("memory_sample_count", 0) or 0)
            last_used = str(item.get("last_used", "")).strip()
            tree.insert(
                "",
                "end",
                values=(name, videos, "Yes" if locked else "No", samples, last_used),
            )
            self.identity_rows[name] = item

    def _open_identity_tools(self) -> None:
        if self.process is not None:
            messagebox.showinfo(APP_TITLE, "Please wait until sorting finishes.")
            return
        self._build_identity_window()
        self._refresh_identity_tools()
        if self.identity_window is not None:
            self.identity_window.deiconify()
            self.identity_window.lift()
            self.identity_window.focus_force()

    def _apply_identity_action(self, args: list[str], action_name: str) -> dict | None:
        try:
            payload = self._run_backend_json(args)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Identity action failed:\n{exc}")
            return None

        if not bool(payload.get("ok", False)):
            messagebox.showerror(APP_TITLE, f"Identity action failed:\n{payload.get('error', payload)}")
            return None

        details = payload.get("details", {})
        self._append_log(f"[IDENTITY] {action_name} -> {details}\n")
        self._refresh_identity_tools()
        self._refresh_learning_stats()
        self._refresh_review_queue_status()
        return payload

    def _identity_merge(self) -> None:
        selected = self._selected_identity_names()
        if len(selected) != 2:
            messagebox.showinfo(APP_TITLE, "Select exactly two identity folders: source then target.")
            return
        source, target = selected[0], selected[1]
        if source == target:
            messagebox.showerror(APP_TITLE, "Source and target folders must be different.")
            return
        if not messagebox.askyesno(
            APP_TITLE,
            f"Merge all videos from '{source}' into '{target}'?\n\nThis also updates learning memory labels.",
        ):
            return
        self._apply_identity_action(
            [
                "--identity-action",
                "merge",
                "--identity-source-folder",
                source,
                "--identity-target-folder",
                target,
            ],
            "merge",
        )

    def _identity_lock(self) -> None:
        selected = self._selected_identity_names()
        if len(selected) != 1:
            messagebox.showinfo(APP_TITLE, "Select exactly one identity folder to lock.")
            return
        folder = selected[0]
        self._apply_identity_action(
            [
                "--identity-action",
                "lock",
                "--identity-folder",
                folder,
            ],
            "lock",
        )

    def _identity_unlock(self) -> None:
        selected = self._selected_identity_names()
        if len(selected) != 1:
            messagebox.showinfo(APP_TITLE, "Select exactly one identity folder to unlock.")
            return
        folder = selected[0]
        self._apply_identity_action(
            [
                "--identity-action",
                "unlock",
                "--identity-folder",
                folder,
            ],
            "unlock",
        )

    def _identity_split(self) -> None:
        selected = self._selected_identity_names()
        if len(selected) != 1:
            messagebox.showinfo(APP_TITLE, "Select exactly one source identity folder to split.")
            return

        source_folder = selected[0]
        destination_text = self.destination_var.get().strip()
        if not destination_text:
            messagebox.showerror(APP_TITLE, "Choose destination folder first.")
            return
        source_dir = Path(destination_text).expanduser().resolve() / source_folder
        if not source_dir.is_dir():
            messagebox.showerror(APP_TITLE, f"Source identity folder missing:\n{source_dir}")
            return

        videos = [
            p
            for p in sorted(source_dir.iterdir(), key=lambda item: item.name.lower())
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
        if not videos:
            messagebox.showinfo(APP_TITLE, "No files found in selected source identity folder.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Split Identity - {source_folder}")
        dialog.geometry("760x560")
        dialog.minsize(680, 460)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=1)

        ttk.Label(
            dialog,
            text=f"Select videos to move from '{source_folder}' and choose destination folder:",
            padding=(12, 12, 12, 6),
        ).grid(row=0, column=0, sticky="w")

        list_frame = ttk.Frame(dialog, padding=(12, 0, 12, 8))
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(list_frame, command=listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        listbox.configure(yscrollcommand=scroll.set)

        for video in videos:
            listbox.insert("end", video.name)

        target_frame = ttk.Frame(dialog, padding=(12, 0, 12, 8))
        target_frame.grid(row=2, column=0, sticky="ew")
        target_frame.columnconfigure(1, weight=1)

        ttk.Label(target_frame, text="Target Folder").grid(row=0, column=0, sticky="w", padx=(0, 8))
        target_var = tk.StringVar()
        target_entry = ttk.Entry(target_frame, textvariable=target_var)
        target_entry.grid(row=0, column=1, sticky="ew")

        existing = [name for name in self.identity_rows.keys() if name.lower() != source_folder.lower()]
        existing_var = tk.StringVar()
        existing_combo = ttk.Combobox(target_frame, values=existing, textvariable=existing_var, state="readonly")
        existing_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(target_frame, text="Pick Existing").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))

        def _use_existing(event: Any = None) -> None:
            value = existing_var.get().strip()
            if value:
                target_var.set(value)

        existing_combo.bind("<<ComboboxSelected>>", _use_existing)

        controls = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        def _apply_split() -> None:
            indices = listbox.curselection()
            if not indices:
                messagebox.showerror(APP_TITLE, "Select at least one video to split.")
                return
            target = target_var.get().strip()
            if not target:
                messagebox.showerror(APP_TITLE, "Target folder is required.")
                return
            selected_paths = [str(videos[idx]) for idx in indices]
            payload = self._apply_identity_action(
                [
                    "--identity-action",
                    "split",
                    "--identity-source-folder",
                    source_folder,
                    "--identity-target-folder",
                    target,
                    "--identity-video-paths-json",
                    json.dumps(selected_paths),
                ],
                "split",
            )
            if payload is not None:
                dialog.destroy()

        ttk.Button(controls, text="Apply Split", command=_apply_split).grid(row=0, column=0, padx=4, sticky="ew")
        ttk.Button(controls, text="Select All", command=lambda: listbox.select_set(0, "end")).grid(
            row=0, column=1, padx=4, sticky="ew"
        )
        ttk.Button(controls, text="Cancel", command=dialog.destroy).grid(row=0, column=2, padx=4, sticky="ew")

    def _refresh_review_queue_status(self) -> None:
        destination_text = self.destination_var.get().strip()
        if not destination_text:
            self.review_status_var.set("Review Queue: choose destination")
            self.open_review_button.configure(state="disabled")
            return

        destination = Path(destination_text).expanduser()
        if not destination.exists():
            self.review_status_var.set("Review Queue: destination not found")
            self.open_review_button.configure(state="disabled")
            return

        try:
            state = self._fetch_review_state()
            pending_count = int(state.get("pending_count", 0))
            run_id = str(state.get("run_id", "")).strip()
            if run_id:
                self.review_status_var.set(f"Review Queue: {pending_count} pending | {run_id}")
            else:
                self.review_status_var.set(f"Review Queue: {pending_count} pending")
            if self.process is None and pending_count > 0:
                self.open_review_button.configure(state="normal")
            else:
                self.open_review_button.configure(state="disabled")
        except Exception as exc:
            self.review_status_var.set(f"Review Queue: unavailable ({exc})")
            self.open_review_button.configure(state="disabled")

    def _current_review_item(self) -> dict | None:
        if not self.review_items:
            return None
        self.review_index = max(0, min(self.review_index, len(self.review_items) - 1))
        return self.review_items[self.review_index]

    def _build_review_window(self) -> None:
        if self.review_window is not None and self.review_window.winfo_exists():
            return

        win = tk.Toplevel(self.root)
        win.title(f"{APP_TITLE} - Review Queue")
        win.geometry("920x760")
        win.minsize(820, 680)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(2, weight=1)
        self.review_window = win

        top = ttk.Frame(win, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Item").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.review_counter_var).grid(row=0, column=1, sticky="w")

        details = ttk.LabelFrame(win, text="Prediction", padding=12)
        details.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))
        details.columnconfigure(1, weight=1)
        ttk.Label(details, text="Label").grid(row=0, column=0, sticky="w")
        ttk.Label(details, textvariable=self.review_predicted_var).grid(row=0, column=1, sticky="w")
        ttk.Label(details, text="Confidence").grid(row=1, column=0, sticky="w")
        ttk.Label(details, textvariable=self.review_confidence_var).grid(row=1, column=1, sticky="w")
        ttk.Label(details, text="Suggested Folder").grid(row=2, column=0, sticky="w")
        ttk.Label(details, textvariable=self.review_suggested_var).grid(row=2, column=1, sticky="w")
        ttk.Label(details, text="Memory Match").grid(row=3, column=0, sticky="w")
        ttk.Label(details, textvariable=self.review_memory_var).grid(row=3, column=1, sticky="w")
        ttk.Label(details, text="Reason").grid(row=4, column=0, sticky="nw")
        ttk.Label(details, textvariable=self.review_reason_var, wraplength=760).grid(row=4, column=1, sticky="w")
        ttk.Label(details, text="Video").grid(row=5, column=0, sticky="nw")
        ttk.Label(details, textvariable=self.review_path_var, wraplength=760).grid(row=5, column=1, sticky="w")

        preview = ttk.LabelFrame(win, text="Preview Frame", padding=10)
        preview.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 10))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.review_preview_label = ttk.Label(preview, text="Preview unavailable")
        self.review_preview_label.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(win, padding=(12, 0, 12, 12))
        controls.grid(row=3, column=0, sticky="ew")
        for idx in range(7):
            controls.columnconfigure(idx, weight=1)

        ttk.Button(controls, text="Open Video", command=self._open_current_video).grid(row=0, column=0, sticky="ew", padx=4)
        ttk.Button(controls, text="Approve Suggested", command=lambda: self._apply_review_action("approve_suggested")).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(controls, text="Move To No Female", command=lambda: self._apply_review_action("move_no_female")).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(controls, text="Reassign Existing", command=self._reassign_existing).grid(
            row=0, column=3, sticky="ew", padx=4
        )
        ttk.Button(controls, text="Reassign New", command=self._reassign_new).grid(row=0, column=4, sticky="ew", padx=4)
        ttk.Button(controls, text="Skip", command=lambda: self._apply_review_action("skip")).grid(row=0, column=5, sticky="ew", padx=4)
        ttk.Button(controls, text="Close", command=self._close_review_window).grid(row=0, column=6, sticky="ew", padx=4)

        nav = ttk.Frame(win, padding=(12, 0, 12, 12))
        nav.grid(row=4, column=0, sticky="ew")
        nav.columnconfigure(0, weight=1)
        nav.columnconfigure(1, weight=1)
        ttk.Button(nav, text="Previous", command=self._prev_review_item).grid(row=0, column=0, sticky="w")
        ttk.Button(nav, text="Next", command=self._next_review_item).grid(row=0, column=1, sticky="e")

        win.protocol("WM_DELETE_WINDOW", self._close_review_window)

    def _close_review_window(self) -> None:
        if self.review_window is not None and self.review_window.winfo_exists():
            self.review_window.destroy()
        self.review_window = None
        self.review_preview_image = None

    def _load_review_preview(self, video_path: Path) -> None:
        label = getattr(self, "review_preview_label", None)
        if label is None:
            return
        if not video_path.exists():
            label.configure(text="Video not found", image="")
            self.review_preview_image = None
            return

        cap = cv2.VideoCapture(str(video_path))
        frame = None
        try:
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                target = max(0, total_frames // 3)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        frame = None
        finally:
            cap.release()

        if frame is None:
            label.configure(text="Preview unavailable", image="")
            self.review_preview_image = None
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((760, 420))
        photo = ImageTk.PhotoImage(image=image)
        label.configure(image=photo, text="")
        self.review_preview_image = photo
        label.image = photo

    def _render_review_item(self) -> None:
        item = self._current_review_item()
        if item is None:
            self.review_counter_var.set("0/0")
            self.review_predicted_var.set("")
            self.review_confidence_var.set("")
            self.review_reason_var.set("")
            self.review_suggested_var.set("")
            self.review_memory_var.set("")
            self.review_path_var.set("")
            label = getattr(self, "review_preview_label", None)
            if label is not None:
                label.configure(text="No pending items", image="")
            self.review_preview_image = None
            return

        idx = self.review_index + 1
        total = len(self.review_items)
        self.review_counter_var.set(f"{idx}/{total}")
        self.review_predicted_var.set(str(item.get("predicted_label", "")))
        self.review_confidence_var.set(str(item.get("confidence", "")))
        self.review_reason_var.set(str(item.get("reason", "")))
        self.review_suggested_var.set(str(item.get("suggested_folder", "")))
        memory_label = str(item.get("memory_match_label", "")).strip()
        try:
            memory_score = float(item.get("memory_match_score", 0.0))
        except Exception:
            memory_score = 0.0
        if memory_label:
            self.review_memory_var.set(f"{memory_label} ({memory_score:.3f})")
        else:
            suggestion = str(item.get("memory_suggestion", "")).strip()
            self.review_memory_var.set(suggestion)

        pending_path = Path(str(item.get("pending_path", "")))
        self.review_path_var.set(str(pending_path))
        self._load_review_preview(pending_path)

    def _open_review_queue(self) -> None:
        if self.process is not None:
            messagebox.showinfo(APP_TITLE, "Please wait until sorting finishes.")
            return

        try:
            state = self._fetch_review_state()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to load review queue:\n{exc}")
            return

        items = state.get("items", [])
        if not isinstance(items, list):
            items = []
        self.review_items = [item for item in items if isinstance(item, dict) and item.get("status") == "pending"]

        if not self.review_items:
            messagebox.showinfo(APP_TITLE, "No pending review items.")
            self._refresh_review_queue_status()
            return

        self.review_index = 0
        self._build_review_window()
        if self.review_window is not None:
            self.review_window.deiconify()
            self.review_window.lift()
            self.review_window.focus_force()
        self._render_review_item()

    def _open_current_video(self) -> None:
        item = self._current_review_item()
        if item is None:
            return
        pending_path = Path(str(item.get("pending_path", "")))
        if not pending_path.exists():
            messagebox.showerror(APP_TITLE, f"Video not found:\n{pending_path}")
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(pending_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(pending_path)])
            else:
                subprocess.Popen(["xdg-open", str(pending_path)])
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to open video:\n{exc}")

    def _apply_review_action(self, action: str, target_folder: str = "") -> None:
        item = self._current_review_item()
        if item is None:
            return
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            messagebox.showerror(APP_TITLE, "Current review item has no id.")
            return

        args = [
            "--review-action",
            action,
            "--review-item-id",
            item_id,
        ]
        if target_folder.strip():
            args.extend(["--review-target-folder", target_folder.strip()])

        try:
            payload = self._run_backend_json(args)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to apply review action:\n{exc}")
            return

        if not bool(payload.get("ok", False)):
            messagebox.showerror(APP_TITLE, f"Review action failed:\n{payload.get('error', payload)}")
            return
        if bool(payload.get("learning_updated", False)):
            info = payload.get("learning_info", {})
            if isinstance(info, dict):
                self._append_log(
                    f"[LEARNING] updated memory file={info.get('memory_file', '')} "
                    f"feedback_events={info.get('feedback_events', '')}\n"
                )

        try:
            state = self._fetch_review_state()
            items = state.get("items", [])
            if not isinstance(items, list):
                items = []
            self.review_items = [entry for entry in items if isinstance(entry, dict) and entry.get("status") == "pending"]
            if self.review_index >= len(self.review_items):
                self.review_index = max(0, len(self.review_items) - 1)
            self._render_review_item()
            self._refresh_review_queue_status()
            if not self.review_items and self.review_window is not None:
                messagebox.showinfo(APP_TITLE, "Review queue completed.")
                self._close_review_window()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Failed to refresh review queue:\n{exc}")

    def _reassign_existing(self) -> None:
        item = self._current_review_item()
        if item is None:
            return
        default = str(item.get("suggested_folder", "")).strip()
        value = simpledialog.askstring(APP_TITLE, "Existing folder name", initialvalue=default)
        if value is None:
            return
        self._apply_review_action("reassign_existing", value)

    def _reassign_new(self) -> None:
        value = simpledialog.askstring(APP_TITLE, "New folder name")
        if value is None:
            return
        self._apply_review_action("reassign_new", value)

    def _next_review_item(self) -> None:
        if not self.review_items:
            return
        self.review_index = (self.review_index + 1) % len(self.review_items)
        self._render_review_item()

    def _prev_review_item(self) -> None:
        if not self.review_items:
            return
        self.review_index = (self.review_index - 1) % len(self.review_items)
        self._render_review_item()

    def _start_sorting(self) -> None:
        if self.process is not None:
            return

        command = self._validate_inputs()
        if command is None:
            return

        self._start_backend_command(command, "Sorting in progress...")

    def _start_reprocess_uncertain(self) -> None:
        if self.process is not None:
            return
        command = self._validate_reprocess_uncertain_inputs()
        if command is None:
            return
        self._start_backend_command(command, "Reprocessing uncertain videos...")

    def _start_backend_command(self, command: list[str], status_text: str) -> None:
        self._save_settings()
        self._append_log(f"$ {' '.join(command)}\n\n")
        self.status_var.set(status_text)
        self.live_status_var.set("Live: starting...")
        self.progress_status_var.set("Progress: 0/0 videos")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal", text="Stop (Save Progress)")
        self.open_review_button.configure(state="disabled")
        self.identity_tools_button.configure(state="disabled")
        self.duplicate_finder_button.configure(state="disabled")
        self.reprocess_uncertain_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=100)
        self._clear_results_table()

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BACKEND_SCRIPT.parent),
        )
        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _read_process_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)
        return_code = self.process.wait()
        self.log_queue.put(f"\nProcess finished with exit code {return_code}\n")
        self.log_queue.put("__PROCESS_DONE__")

    def _poll_log_queue(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if item == "__PROCESS_DONE__":
                self.process = None
                self.reader_thread = None
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled", text="Stop")
                self.identity_tools_button.configure(state="normal")
                self.duplicate_finder_button.configure(state="normal")
                self.status_var.set("Finished")
                self.live_status_var.set("Live: idle")
                self.stop_requested = False
                self._clear_stop_flag()
                self._refresh_review_queue_status()
                self._update_reprocess_button_state()
            else:
                result_payload = parse_result_json_line(item)
                if result_payload is not None:
                    self._append_result_row(result_payload)
                    continue
                if item.startswith("[TRACE] "):
                    self.live_status_var.set(item.replace("[TRACE] ", "Live: ").strip())
                if item.startswith("[PROGRESS] "):
                    payload = item[len("[PROGRESS] ") :].strip().split()
                    values: dict[str, int] = {}
                    for chunk in payload:
                        if "=" not in chunk:
                            continue
                        key, raw = chunk.split("=", 1)
                        try:
                            values[key] = int(raw)
                        except ValueError:
                            continue
                    total = max(1, values.get("total", 0))
                    done = values.get("done", 0)
                    female = values.get("female", 0)
                    no_female = values.get("no_female", 0)
                    errors = values.get("errors", 0)
                    self.progress.configure(maximum=total, value=min(done, total))
                    self.progress_status_var.set(
                        f"Progress: {done}/{total} videos | female={female} | no_female={no_female} | errors={errors}"
                    )
                self._append_log(item)

        self._update_reprocess_button_state()
        self.root.after(100, self._poll_log_queue)

    def _stop_sorting(self) -> None:
        if self.process is None:
            return
        if not self.stop_requested:
            self.stop_requested = True
            if self.stop_flag_path is not None:
                try:
                    self.stop_flag_path.write_text("stop\n", encoding="utf-8")
                except Exception as exc:
                    self._append_log(f"\n[WARN] Failed to write stop flag: {exc}\n")
                    self._append_log("Forcing stop instead...\n")
                    self.process.terminate()
                    self.status_var.set("Stopping (forced)...")
                    return
            self._append_log("\nGraceful stop requested. Finishing current work and moving processed files...\n")
            self.status_var.set("Stopping gracefully...")
            self.stop_button.configure(text="Force Stop")
            return

        self._append_log("\nForce stopping process...\n")
        self.process.terminate()
        self.status_var.set("Stopping (forced)...")

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno(APP_TITLE, "Sorting is still running. Close anyway and stop it?"):
                return
            self.process.terminate()
        self._clear_stop_flag()
        self._close_review_window()
        self._close_identity_window()
        self._close_learning_stats_window()
        self._close_duplicate_window()
        self._save_settings()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    ttk.Style(root).theme_use("clam")
    VideoSorterGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
