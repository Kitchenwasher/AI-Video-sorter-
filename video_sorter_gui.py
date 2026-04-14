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
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk


APP_TITLE = "Female Face Video Sorter"
BACKEND_SCRIPT = Path(__file__).with_name("sort_videos_by_female_faces_gpu.py")
SETTINGS_FILE = Path(__file__).with_name("video_sorter_settings.json")
REVIEW_STATE_FILE = ".review_queue/review_state.json"


class VideoSorterGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("860x680")
        self.root.minsize(760, 560)

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

        self._load_settings()
        self._update_include_generated_button_text()
        self._update_review_mode_button_text()
        self._update_learning_button_text()
        self._update_live_trace_button_text()
        self._build_ui()
        self._refresh_runtime_status()
        self._refresh_review_queue_status()
        self.root.after(100, self._poll_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

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

        footer = ttk.Frame(self.root, padding=(14, 0, 14, 14))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        footer.columnconfigure(2, weight=1)

        self.start_button = ttk.Button(footer, text="Start Sorting", command=self._start_sorting)
        self.start_button.grid(row=0, column=0, padx=(0, 8))

        self.stop_button = ttk.Button(footer, text="Stop", command=self._stop_sorting, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="w")

        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, value=0)
        self.progress.grid(row=0, column=2, sticky="ew", padx=12)

        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=3, sticky="e")

        self.open_review_button = ttk.Button(footer, text="Open Review Queue", command=self._open_review_queue, state="disabled")
        self.open_review_button.grid(row=1, column=0, padx=(0, 8), pady=(8, 0), sticky="w")
        ttk.Label(footer, textvariable=self.review_status_var).grid(row=1, column=1, columnspan=3, sticky="w", pady=(8, 0))

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

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

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

        self.stop_flag_path = BACKEND_SCRIPT.parent / f".stop_{uuid.uuid4().hex}.flag"
        self.stop_requested = False
        self._clear_stop_flag()

        command = [
            sys.executable,
            str(BACKEND_SCRIPT),
            "--input-dir",
            str(source.resolve()),
            "--output-dir",
            str(destination.resolve()),
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
        if not self.recursive_var.get():
            command.append("--no-recursive")
        if self.include_generated_var.get():
            command.append("--include-generated-folders")
        if self.review_mode_var.get():
            command.append("--review-mode")
        if not self.learning_enabled_var.get():
            command.append("--no-learning-enabled")
        if self.live_trace_var.get():
            command.append("--live-trace")
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

        self._save_settings()
        self._append_log(f"$ {' '.join(command)}\n\n")
        self.status_var.set("Sorting in progress...")
        self.live_status_var.set("Live: starting...")
        self.progress_status_var.set("Progress: 0/0 videos")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal", text="Stop (Save Progress)")
        self.open_review_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=100)

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
                self.status_var.set("Finished")
                self.live_status_var.set("Live: idle")
                self.stop_requested = False
                self._clear_stop_flag()
                self._refresh_review_queue_status()
            else:
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
