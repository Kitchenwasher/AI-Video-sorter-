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
import time
import tkinter as tk
import uuid
from collections import deque
from pathlib import Path
from typing import Any
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import umap  # type: ignore[import-not-found]
except Exception:
    umap = None


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
DARK_BG = "#15181d"
DARK_SURFACE = "#1f242c"
DARK_BORDER = "#2b3240"
DARK_TEXT = "#e6edf3"
DARK_MUTED = "#9aa4b2"
DARK_ACCENT = "#3b82f6"
DARK_ACCENT_ACTIVE = "#2563eb"
LOG_MAX_LINES = 3000
DASH_SPEED_HISTORY_MAX = 180
DASH_CONFIDENCE_HISTORY_MAX = 3000
DASH_CLUSTER_MAX_POINTS = 500
DASH_SPEED_ROLLING_WINDOW_SEC = 300.0
PROFILE_DEFAULT_FIELDS: dict[str, dict[str, str]] = {
    "Fast": {
        "max_seconds": "40",
        "sample_every_sec": "2.5",
        "stabilization_seconds": "6.0",
        "resize_width": "720",
        "max_workers": "2",
    },
    "Balanced": {
        "max_seconds": "60",
        "sample_every_sec": "2.0",
        "stabilization_seconds": "8.0",
        "resize_width": "960",
        "max_workers": "2",
    },
    "High Accuracy": {
        "max_seconds": "90",
        "sample_every_sec": "1.0",
        "stabilization_seconds": "10.0",
        "resize_width": "1152",
        "max_workers": "1",
    },
}


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


def resolve_profile_defaults(profile_name: str) -> dict[str, str]:
    selected = profile_name.strip()
    if selected not in PROFILE_DEFAULT_FIELDS:
        selected = "Balanced"
    return dict(PROFILE_DEFAULT_FIELDS[selected])


def combine_log_chunks(chunks: list[str]) -> str:
    return "".join(chunks)


def compute_log_trim_lines(total_lines: int, max_lines: int = LOG_MAX_LINES) -> int:
    return max(0, int(total_lines) - int(max_lines))


class VideoSorterGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("940x760")
        self.root.minsize(860, 640)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.runtime_refresh_queue: queue.Queue[tuple[int, str, str]] = queue.Queue()
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
        self.use_insightface_var = tk.BooleanVar(value=True)
        self.face_engine_button_var = tk.StringVar()
        self.cross_video_reid_var = tk.BooleanVar(value=False)
        self.cross_video_reid_button_var = tk.StringVar()
        self.reid_model_tier_var = tk.StringVar(value="balanced")
        self.video_io_prefetch_var = tk.BooleanVar(value=False)
        self.video_io_prefetch_button_var = tk.StringVar()
        self.live_trace_var = tk.BooleanVar(value=False)
        self.live_trace_button_var = tk.StringVar()
        self.max_seconds_var = tk.StringVar(value="60")
        self.sample_every_var = tk.StringVar(value="2.0")
        self.retry_checkpoint_step_pct_var = tk.IntVar(value=5)
        self.retry_checkpoint_step_label_var = tk.StringVar()
        self.stabilization_var = tk.StringVar(value="8.0")
        self.resize_width_var = tk.StringVar(value="960")
        self.max_workers_var = tk.StringVar(value="2")
        self.profile_var = tk.StringVar(value="Balanced")
        self.status_var = tk.StringVar(value="Ready")
        self.runtime_var = tk.StringVar(value="Runtime: checking...")
        self.face_engine_status_var = tk.StringVar(value="Face Engine: InsightFace (selected)")
        self.live_status_var = tk.StringVar(value="Live: idle")
        self.progress_status_var = tk.StringVar(value="Progress: 0/0 videos")
        self.review_status_var = tk.StringVar(value="Review Queue: 0 pending")
        self.dashboard_speed_var = tk.StringVar(value="Speed: -- videos/min")
        self.dashboard_eta_var = tk.StringVar(value="ETA (rolling): --")
        self.dashboard_projection_var = tk.StringVar(value="Projection: waiting for embeddings")

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
        self.dashboard_speed_canvas: tk.Canvas | None = None
        self.dashboard_conf_canvas: tk.Canvas | None = None
        self.dashboard_cluster_canvas: tk.Canvas | None = None
        self.dashboard_progress_points: deque[tuple[float, int]] = deque()
        self.dashboard_speed_points: deque[float] = deque(maxlen=DASH_SPEED_HISTORY_MAX)
        self.dashboard_confidences: deque[float] = deque(maxlen=DASH_CONFIDENCE_HISTORY_MAX)
        self.dashboard_embeddings: list[np.ndarray] = []
        self.dashboard_labels: list[str] = []
        self.dashboard_cluster_points_2d: np.ndarray | None = None
        self.dashboard_cluster_colors: list[str] = []
        self.dashboard_cluster_last_len = 0
        self.dashboard_total_videos = 0
        self.dashboard_last_done = 0
        self.dashboard_projection_last_method = ""
        self.dashboard_last_projection_at = 0.0
        self.dashboard_embedding_version = 0
        self.dashboard_cluster_projected_version = -1
        self.main_scroll_canvas: tk.Canvas | None = None
        self.main_scroll_content: ttk.Frame | None = None
        self.main_scroll_window_id: int | None = None
        self._runtime_refresh_token = 0

        self._load_settings()
        self._update_include_generated_button_text()
        self._update_review_mode_button_text()
        self._update_learning_button_text()
        self._update_face_engine_button_text()
        self._update_face_engine_status_text()
        self._update_cross_video_reid_button_text()
        self._update_video_io_prefetch_button_text()
        self._update_live_trace_button_text()
        self._update_retry_checkpoint_step_label()
        self._build_ui()
        self._dashboard_reset()
        self._update_reprocess_button_state()
        self._refresh_runtime_status()
        self._refresh_review_queue_status()
        self.root.after(100, self._poll_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        shell = ttk.Frame(self.root)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        canvas = tk.Canvas(shell, highlightthickness=0, borderwidth=0, bg=DARK_BG)
        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vscroll.set)

        content = ttk.Frame(canvas)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(3, weight=1, minsize=140)
        content.rowconfigure(4, weight=0, minsize=120)
        content.rowconfigure(5, weight=0, minsize=220)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        self.main_scroll_canvas = canvas
        self.main_scroll_content = content
        self.main_scroll_window_id = window_id

        def _sync_scroll_region(_event: tk.Event[tk.Misc]) -> None:
            if self.main_scroll_canvas is None:
                return
            self.main_scroll_canvas.configure(scrollregion=self.main_scroll_canvas.bbox("all"))

        def _fit_content_width(event: tk.Event[tk.Misc]) -> None:
            if self.main_scroll_canvas is None or self.main_scroll_window_id is None:
                return
            self.main_scroll_canvas.itemconfigure(self.main_scroll_window_id, width=event.width)

        def _is_descendant(widget: Any, ancestor: Any) -> bool:
            current = widget
            while current is not None:
                if current is ancestor:
                    return True
                current = getattr(current, "master", None)
            return False

        def _is_native_scroll_widget(widget: Any) -> bool:
            current = widget
            while current is not None:
                try:
                    klass = str(current.winfo_class())
                except Exception:
                    klass = ""
                if klass in {"Text", "Treeview", "Listbox"}:
                    return True
                current = getattr(current, "master", None)
            return False

        def _should_skip_global_scroll(widget: Any) -> bool:
            log_widget = getattr(self, "log_text", None)
            if bool(log_widget is not None and _is_descendant(widget, log_widget)):
                return True
            return _is_native_scroll_widget(widget)

        def _on_mousewheel(event: tk.Event[tk.Misc]) -> None:
            if self.main_scroll_canvas is None:
                return
            delta = int(getattr(event, "delta", 0))
            if delta != 0:
                self.main_scroll_canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        def _on_mousewheel_anywhere(event: tk.Event[tk.Misc]) -> None:
            if self.main_scroll_canvas is None:
                return
            widget = getattr(event, "widget", None)
            try:
                if widget is None or widget.winfo_toplevel() != self.root:
                    return
            except Exception:
                return
            if _should_skip_global_scroll(widget):
                return
            delta = int(getattr(event, "delta", 0))
            if delta != 0:
                self.main_scroll_canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        def _on_mousewheel_linux(event: tk.Event[tk.Misc]) -> None:
            if self.main_scroll_canvas is None:
                return
            widget = getattr(event, "widget", None)
            try:
                if widget is None or widget.winfo_toplevel() != self.root:
                    return
            except Exception:
                return
            if _should_skip_global_scroll(widget):
                return
            direction = -1 if int(getattr(event, "num", 0)) == 4 else 1
            self.main_scroll_canvas.yview_scroll(direction, "units")

        content.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _fit_content_width)
        canvas.bind("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<MouseWheel>", _on_mousewheel_anywhere, add="+")
        self.root.bind_all("<Button-4>", _on_mousewheel_linux, add="+")
        self.root.bind_all("<Button-5>", _on_mousewheel_linux, add="+")

        top = ttk.Frame(content, padding=14)
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

        runtime_bar = ttk.Frame(content, padding=(14, 0, 14, 10))
        runtime_bar.grid(row=1, column=0, sticky="ew")
        runtime_bar.columnconfigure(0, weight=1)

        ttk.Label(runtime_bar, textvariable=self.runtime_var).grid(row=0, column=0, sticky="w")
        ttk.Label(runtime_bar, textvariable=self.face_engine_status_var, foreground=DARK_MUTED).grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Button(runtime_bar, text="Refresh Runtime", command=self._refresh_runtime_status).grid(
            row=0, column=1, rowspan=2, padx=(10, 0)
        )

        options = ttk.LabelFrame(content, text="Options", padding=14)
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
            textvariable=self.face_engine_button_var,
            command=self._toggle_face_engine,
        ).grid(row=2, column=2, columnspan=2, sticky="ew", pady=(0, 10))

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
        ttk.Button(
            options,
            textvariable=self.video_io_prefetch_button_var,
            command=self._toggle_video_io_prefetch,
        ).grid(row=5, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        ttk.Button(
            options,
            textvariable=self.cross_video_reid_button_var,
            command=self._toggle_cross_video_reid,
        ).grid(row=6, column=1, sticky="ew", padx=(0, 10), pady=(0, 2))

        ttk.Label(options, textvariable=self.retry_checkpoint_step_label_var).grid(
            row=7, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Button(options, text="-5%", command=lambda: self._adjust_retry_checkpoint_step_pct(-5)).grid(
            row=7, column=1, sticky="ew", padx=(0, 10), pady=(8, 0)
        )
        ttk.Button(options, text="+5%", command=lambda: self._adjust_retry_checkpoint_step_pct(5)).grid(
            row=7, column=2, sticky="ew", padx=(0, 10), pady=(8, 0)
        )

        ttk.Label(
            options,
            text="Source videos are discovered by extension and sorted from child folders too when recursive scan is enabled.",
            wraplength=760,
            foreground="#444444",
        ).grid(row=8, column=0, columnspan=4, sticky="w", pady=(12, 0))

        log_frame = ttk.LabelFrame(content, text="Logs", padding=10)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            height=8,
            state="disabled",
            bg="#0f1318",
            fg=DARK_TEXT,
            insertbackground=DARK_TEXT,
            selectbackground="#2f6feb",
            selectforeground="#ffffff",
            highlightthickness=0,
            relief="flat",
            borderwidth=0,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)
        ttk.Label(log_frame, textvariable=self.live_status_var).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(log_frame, textvariable=self.progress_status_var).grid(row=2, column=0, sticky="w", pady=(4, 0))

        results_frame = ttk.LabelFrame(content, text="Results This Run", padding=10)
        results_frame.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(
            results_frame,
            columns=("video", "decision", "confidence", "why", "tags"),
            show="headings",
            height=6,
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

        dashboard_frame = ttk.LabelFrame(content, text="Batch Processing Dashboard", padding=10)
        dashboard_frame.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 10))
        dashboard_frame.columnconfigure(0, weight=1)
        dashboard_frame.columnconfigure(1, weight=1)
        dashboard_frame.columnconfigure(2, weight=1)

        ttk.Label(dashboard_frame, textvariable=self.dashboard_speed_var).grid(row=0, column=0, sticky="w")
        ttk.Label(dashboard_frame, textvariable=self.dashboard_eta_var).grid(row=0, column=1, sticky="w")
        ttk.Label(dashboard_frame, textvariable=self.dashboard_projection_var).grid(row=0, column=2, sticky="w")

        self.dashboard_speed_canvas = tk.Canvas(
            dashboard_frame,
            width=260,
            height=120,
            bg="#0f1318",
            highlightthickness=1,
            highlightbackground=DARK_BORDER,
        )
        self.dashboard_speed_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))

        self.dashboard_conf_canvas = tk.Canvas(
            dashboard_frame,
            width=260,
            height=120,
            bg="#0f1318",
            highlightthickness=1,
            highlightbackground=DARK_BORDER,
        )
        self.dashboard_conf_canvas.grid(row=1, column=1, sticky="nsew", padx=(0, 8), pady=(8, 0))

        self.dashboard_cluster_canvas = tk.Canvas(
            dashboard_frame,
            width=320,
            height=120,
            bg="#0f1318",
            highlightthickness=1,
            highlightbackground=DARK_BORDER,
        )
        self.dashboard_cluster_canvas.grid(row=1, column=2, sticky="nsew", pady=(8, 0))

        footer = ttk.Frame(content, padding=(14, 0, 14, 14))
        footer.grid(row=6, column=0, sticky="ew")
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
        total_lines = int(float(self.log_text.index("end-1c").split(".")[0]))
        trim_lines = compute_log_trim_lines(total_lines, LOG_MAX_LINES)
        if trim_lines > 0:
            self.log_text.delete("1.0", f"{trim_lines + 1}.0")
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

    def _dashboard_reset(self) -> None:
        self.dashboard_progress_points.clear()
        self.dashboard_speed_points.clear()
        self.dashboard_confidences.clear()
        self.dashboard_embeddings.clear()
        self.dashboard_labels.clear()
        self.dashboard_cluster_points_2d = None
        self.dashboard_cluster_colors.clear()
        self.dashboard_cluster_last_len = 0
        self.dashboard_total_videos = 0
        self.dashboard_last_done = 0
        self.dashboard_projection_last_method = ""
        self.dashboard_last_projection_at = 0.0
        self.dashboard_embedding_version = 0
        self.dashboard_cluster_projected_version = -1
        self.dashboard_speed_var.set("Speed: -- videos/min")
        self.dashboard_eta_var.set("ETA (rolling): --")
        self.dashboard_projection_var.set("Projection: waiting for embeddings")
        self._dashboard_draw_speed_trend()
        self._dashboard_draw_confidence_histogram()
        self._dashboard_draw_cluster_projection()

    @staticmethod
    def _dashboard_clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _dashboard_format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d}h {minutes:02d}m {secs:02d}s"
        if minutes > 0:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"

    @staticmethod
    def _dashboard_canvas_size(canvas: tk.Canvas, fallback_width: int, fallback_height: int) -> tuple[int, int]:
        width = int(canvas.winfo_width())
        height = int(canvas.winfo_height())
        if width < 40:
            width = int(float(canvas.cget("width")))
        if height < 40:
            height = int(float(canvas.cget("height")))
        return max(width, fallback_width), max(height, fallback_height)

    @staticmethod
    def _dashboard_identity_label_from_payload(payload: dict) -> str:
        candidates = (
            str(payload.get("suggested_folder_name", "")).strip(),
            str(payload.get("memory_match_label", "")).strip(),
            str(payload.get("decision_label", "")).strip(),
        )
        for label in candidates:
            if label:
                return label
        return "unknown"

    def _dashboard_update_from_result(self, payload: dict) -> None:
        try:
            confidence = float(payload.get("confidence_score", 0.0))
        except Exception:
            confidence = 0.0
        self.dashboard_confidences.append(self._dashboard_clamp01(confidence))
        self._dashboard_draw_confidence_histogram()

        embedding_raw = payload.get("embedding")
        embedding_added = False
        if isinstance(embedding_raw, list) and embedding_raw:
            try:
                vector = np.asarray(embedding_raw, dtype=np.float32).reshape(-1)
            except Exception:
                vector = np.asarray([], dtype=np.float32)
            if vector.size > 0 and np.isfinite(vector).all():
                norm = float(np.linalg.norm(vector))
                if norm > 1e-8:
                    vector = (vector / norm).astype(np.float32)
                    self.dashboard_embeddings.append(vector)
                    self.dashboard_labels.append(self._dashboard_identity_label_from_payload(payload))
                    if len(self.dashboard_embeddings) > DASH_CLUSTER_MAX_POINTS:
                        self.dashboard_embeddings = self.dashboard_embeddings[-DASH_CLUSTER_MAX_POINTS:]
                        self.dashboard_labels = self.dashboard_labels[-DASH_CLUSTER_MAX_POINTS:]
                    self.dashboard_embedding_version += 1
                    embedding_added = True

        if embedding_added:
            self._dashboard_maybe_refresh_projection(force=False)
        elif not self.dashboard_embeddings:
            self.dashboard_projection_var.set("Projection: waiting for embeddings")
            self._dashboard_draw_cluster_projection()

    def _dashboard_update_from_progress(self, done: int, total: int) -> None:
        now = time.time()
        done = max(0, int(done))
        total = max(0, int(total))
        self.dashboard_last_done = done
        self.dashboard_total_videos = max(self.dashboard_total_videos, total)

        if not self.dashboard_progress_points:
            self.dashboard_progress_points.append((now, done))
        else:
            last_time, last_done = self.dashboard_progress_points[-1]
            if done != last_done:
                self.dashboard_progress_points.append((now, done))
            else:
                self.dashboard_progress_points[-1] = (now, done)

        cutoff = now - DASH_SPEED_ROLLING_WINDOW_SEC
        while len(self.dashboard_progress_points) > 1 and self.dashboard_progress_points[1][0] < cutoff:
            self.dashboard_progress_points.popleft()

        speed = 0.0
        if len(self.dashboard_progress_points) >= 2:
            start_time, start_done = self.dashboard_progress_points[0]
            end_time, end_done = self.dashboard_progress_points[-1]
            dt = max(1e-6, float(end_time - start_time))
            dd = max(0, int(end_done) - int(start_done))
            speed = (dd / dt) * 60.0

        self.dashboard_speed_points.append(speed)
        self.dashboard_speed_var.set(f"Speed: {speed:.2f} videos/min")
        if total > 0 and done >= total:
            self.dashboard_eta_var.set("ETA (rolling): complete")
        elif speed > 1e-6 and total > 0:
            remaining = max(0, total - done)
            eta_seconds = (remaining / speed) * 60.0
            self.dashboard_eta_var.set(f"ETA (rolling): {self._dashboard_format_duration(eta_seconds)}")
        else:
            self.dashboard_eta_var.set("ETA (rolling): --")
        self._dashboard_draw_speed_trend()

    def _dashboard_project_embeddings(self, matrix: np.ndarray) -> tuple[np.ndarray, str]:
        if matrix.shape[0] < 2:
            return np.zeros((0, 2), dtype=np.float32), "waiting"

        if umap is not None and matrix.shape[0] >= 3:
            try:
                neighbors = max(2, min(15, matrix.shape[0] - 1))
                reducer = umap.UMAP(  # type: ignore[attr-defined]
                    n_components=2,
                    n_neighbors=neighbors,
                    min_dist=0.15,
                    metric="cosine",
                    random_state=42,
                )
                points = reducer.fit_transform(matrix).astype(np.float32)
                return points, "UMAP"
            except Exception:
                pass

        centered = matrix - matrix.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            basis = vh[:2].T if vh.shape[0] >= 2 else np.zeros((centered.shape[1], 2), dtype=np.float32)
            points = centered @ basis
        except Exception:
            axis = np.arange(matrix.shape[0], dtype=np.float32)
            points = np.column_stack((axis, np.zeros_like(axis)))
        return np.asarray(points, dtype=np.float32), "PCA fallback"

    @staticmethod
    def _dashboard_color_for_label(label: str) -> str:
        palette = (
            "#60a5fa",
            "#34d399",
            "#f59e0b",
            "#f87171",
            "#22d3ee",
            "#a78bfa",
            "#fbbf24",
            "#2dd4bf",
            "#fb7185",
            "#93c5fd",
        )
        return palette[hash(label) % len(palette)]

    def _dashboard_maybe_refresh_projection(self, force: bool = False) -> None:
        total = len(self.dashboard_embeddings)
        if total == 0:
            self.dashboard_cluster_points_2d = None
            self.dashboard_cluster_colors = []
            self.dashboard_cluster_last_len = 0
            self.dashboard_projection_var.set("Projection: waiting for embeddings")
            self._dashboard_draw_cluster_projection()
            return

        if total < 2:
            self.dashboard_cluster_points_2d = None
            self.dashboard_cluster_colors = []
            self.dashboard_cluster_last_len = total
            self.dashboard_projection_var.set(f"Projection: need >=2 embeddings (n={total})")
            self._dashboard_draw_cluster_projection()
            return

        now = time.time()
        should_project = force
        if self.dashboard_cluster_projected_version != self.dashboard_embedding_version:
            delta = self.dashboard_embedding_version - self.dashboard_cluster_projected_version
            if total <= 30 or delta >= 5 or (now - self.dashboard_last_projection_at) >= 1.5:
                should_project = True

        if not should_project:
            self._dashboard_draw_cluster_projection()
            return

        matrix = np.vstack(self.dashboard_embeddings).astype(np.float32)
        points, method = self._dashboard_project_embeddings(matrix)
        self.dashboard_cluster_points_2d = points
        self.dashboard_cluster_colors = [self._dashboard_color_for_label(label) for label in self.dashboard_labels]
        self.dashboard_cluster_last_len = total
        self.dashboard_projection_last_method = method
        self.dashboard_last_projection_at = now
        self.dashboard_cluster_projected_version = self.dashboard_embedding_version
        self.dashboard_projection_var.set(f"Projection: {method} (n={total})")
        self._dashboard_draw_cluster_projection()

    def _dashboard_draw_speed_trend(self) -> None:
        canvas = self.dashboard_speed_canvas
        if canvas is None:
            return
        width, height = self._dashboard_canvas_size(canvas, 260, 120)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#0f1318", outline="")

        margin = 10
        left = margin
        right = width - margin
        top = margin
        bottom = height - margin - 8
        canvas.create_line(left, bottom, right, bottom, fill=DARK_BORDER)
        canvas.create_line(left, top, left, bottom, fill=DARK_BORDER)

        points = list(self.dashboard_speed_points)
        if not points:
            canvas.create_text(width // 2, height // 2, text="Waiting for progress...", fill=DARK_MUTED)
            return

        max_speed = max(1.0, max(points))
        if len(points) == 1:
            x = (left + right) / 2.0
            y = bottom - ((points[0] / max_speed) * max(1.0, (bottom - top)))
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=DARK_ACCENT, outline="")
            return

        polyline: list[float] = []
        span_x = max(1.0, float(right - left))
        span_y = max(1.0, float(bottom - top))
        for idx, value in enumerate(points):
            x = left + (idx / max(1, len(points) - 1)) * span_x
            y = bottom - ((max(0.0, value) / max_speed) * span_y)
            polyline.extend([x, y])
        canvas.create_line(*polyline, fill=DARK_ACCENT, width=2, smooth=True)
        canvas.create_text(right, top, text=f"{max_speed:.1f}", fill=DARK_MUTED, anchor="ne")

    def _dashboard_draw_confidence_histogram(self) -> None:
        canvas = self.dashboard_conf_canvas
        if canvas is None:
            return
        width, height = self._dashboard_canvas_size(canvas, 260, 120)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#0f1318", outline="")

        margin = 10
        left = margin
        right = width - margin
        top = margin
        bottom = height - margin - 10
        canvas.create_line(left, bottom, right, bottom, fill=DARK_BORDER)
        canvas.create_line(left, top, left, bottom, fill=DARK_BORDER)

        values = list(self.dashboard_confidences)
        if not values:
            canvas.create_text(width // 2, height // 2, text="No confidence samples yet", fill=DARK_MUTED)
            return

        bins = 10
        hist = [0] * bins
        for value in values:
            idx = min(bins - 1, int(value * bins))
            hist[idx] += 1

        max_count = max(1, max(hist))
        bar_width = max(1.0, (right - left) / bins)
        for idx, count in enumerate(hist):
            x0 = left + idx * bar_width + 1
            x1 = left + (idx + 1) * bar_width - 1
            bar_height = (count / max_count) * max(1.0, (bottom - top))
            y0 = bottom - bar_height
            canvas.create_rectangle(x0, y0, x1, bottom, fill="#22d3ee", outline="")
        canvas.create_text(right, top, text=f"n={len(values)}", fill=DARK_MUTED, anchor="ne")

    def _dashboard_draw_cluster_projection(self) -> None:
        canvas = self.dashboard_cluster_canvas
        if canvas is None:
            return
        width, height = self._dashboard_canvas_size(canvas, 320, 120)
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill="#0f1318", outline="")

        points = self.dashboard_cluster_points_2d
        if points is None or points.size == 0 or points.shape[0] < 2:
            canvas.create_text(width // 2, height // 2, text="Need more embeddings...", fill=DARK_MUTED)
            return

        margin = 10
        left = margin
        right = width - margin
        top = margin
        bottom = height - margin
        canvas.create_rectangle(left, top, right, bottom, outline=DARK_BORDER, fill="")

        xs = points[:, 0]
        ys = points[:, 1]
        x_min = float(np.min(xs))
        x_max = float(np.max(xs))
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)

        colors = self.dashboard_cluster_colors
        for idx in range(points.shape[0]):
            x = left + ((float(xs[idx]) - x_min) / x_span) * max(1.0, (right - left))
            y = bottom - ((float(ys[idx]) - y_min) / y_span) * max(1.0, (bottom - top))
            color = colors[idx] if idx < len(colors) else DARK_ACCENT
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")

        method = self.dashboard_projection_last_method or "projection"
        canvas.create_text(right, top, text=method, fill=DARK_MUTED, anchor="ne")

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

    def _update_face_engine_button_text(self) -> None:
        if self.use_insightface_var.get():
            self.face_engine_button_var.set("Face Engine: InsightFace")
        else:
            self.face_engine_button_var.set("Face Engine: FaceNet (Legacy)")

    def _update_face_engine_status_text(self, active_engine: str = "") -> None:
        selected = "InsightFace" if self.use_insightface_var.get() else "FaceNet (Legacy)"
        if active_engine:
            self.face_engine_status_var.set(f"Face Engine: {active_engine} | Selected: {selected}")
        else:
            self.face_engine_status_var.set(f"Face Engine: selected {selected}")

    def _update_cross_video_reid_button_text(self) -> None:
        tier = str(self.reid_model_tier_var.get() or "balanced").strip().lower()
        if tier not in {"fast", "balanced", "high_accuracy"}:
            tier = "balanced"
            self.reid_model_tier_var.set(tier)
        if self.cross_video_reid_var.get():
            self.cross_video_reid_button_var.set(f"Cross-Video ReID: On ({tier})")
        else:
            self.cross_video_reid_button_var.set(f"Cross-Video ReID: Off ({tier})")

    def _update_video_io_prefetch_button_text(self) -> None:
        if self.video_io_prefetch_var.get():
            self.video_io_prefetch_button_var.set("Video I/O Prefetch: On")
        else:
            self.video_io_prefetch_button_var.set("Video I/O Prefetch: Off")

    def _update_live_trace_button_text(self) -> None:
        if self.live_trace_var.get():
            self.live_trace_button_var.set("Live Preview: On")
        else:
            self.live_trace_button_var.set("Live Preview: Off")

    def _update_retry_checkpoint_step_label(self) -> None:
        value = int(self.retry_checkpoint_step_pct_var.get())
        value = max(5, min(95, value))
        if value % 5 != 0:
            value = int(round(value / 5.0) * 5)
            value = max(5, min(95, value))
        self.retry_checkpoint_step_pct_var.set(value)
        self.retry_checkpoint_step_label_var.set(f"Retry Checkpoint Step: {value}%")

    def _adjust_retry_checkpoint_step_pct(self, delta: int) -> None:
        current = int(self.retry_checkpoint_step_pct_var.get())
        updated = current + int(delta)
        updated = max(5, min(95, updated))
        if updated % 5 != 0:
            updated = int(round(updated / 5.0) * 5)
            updated = max(5, min(95, updated))
        if updated == current:
            return
        self.retry_checkpoint_step_pct_var.set(updated)
        self._update_retry_checkpoint_step_label()
        self._save_settings()

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

    def _toggle_face_engine(self) -> None:
        self.use_insightface_var.set(not self.use_insightface_var.get())
        self._update_face_engine_button_text()
        self._update_face_engine_status_text()
        self._save_settings()

    def _toggle_cross_video_reid(self) -> None:
        self.cross_video_reid_var.set(not self.cross_video_reid_var.get())
        self._update_cross_video_reid_button_text()
        self._save_settings()

    def _toggle_video_io_prefetch(self) -> None:
        self.video_io_prefetch_var.set(not self.video_io_prefetch_var.get())
        self._update_video_io_prefetch_button_text()
        self._save_settings()

    def _toggle_live_trace(self) -> None:
        self.live_trace_var.set(not self.live_trace_var.get())
        self._update_live_trace_button_text()
        self._save_settings()

    def _on_profile_selected(self, event: Any = None) -> None:
        self._apply_selected_profile_to_fields()

    def _apply_selected_profile_to_fields(self) -> None:
        selected = self.profile_var.get().strip()
        defaults = resolve_profile_defaults(selected)
        if selected not in PROFILE_TO_BACKEND:
            self.profile_var.set("Balanced")
        self.max_seconds_var.set(defaults["max_seconds"])
        self.sample_every_var.set(defaults["sample_every_sec"])
        self.stabilization_var.set(defaults["stabilization_seconds"])
        self.resize_width_var.set(defaults["resize_width"])
        self.max_workers_var.set(defaults["max_workers"])
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
        self.use_insightface_var.set(bool(data.get("use_insightface", True)))
        self.cross_video_reid_var.set(bool(data.get("cross_video_reid", False)))
        self.reid_model_tier_var.set(str(data.get("reid_model_tier", "balanced")))
        self.video_io_prefetch_var.set(bool(data.get("video_io_prefetch", False)))
        self.live_trace_var.set(bool(data.get("live_trace", False)))
        try:
            retry_step = int(data.get("retry_checkpoint_step_pct", 5))
        except Exception:
            retry_step = 5
        self.retry_checkpoint_step_pct_var.set(retry_step)
        self._update_retry_checkpoint_step_label()
        self._update_include_generated_button_text()
        self._update_review_mode_button_text()
        self._update_learning_button_text()
        self._update_face_engine_button_text()
        self._update_face_engine_status_text()
        self._update_cross_video_reid_button_text()
        self._update_video_io_prefetch_button_text()
        self._update_live_trace_button_text()
        self.max_seconds_var.set(str(data.get("max_seconds", "60")))
        self.sample_every_var.set(str(data.get("sample_every_sec", "2.0")))
        self.stabilization_var.set(str(data.get("stabilization_seconds", "8.0")))
        self.resize_width_var.set(str(data.get("resize_width", "960")))
        self.max_workers_var.set(str(data.get("max_workers", "2")))
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
            "use_insightface": self.use_insightface_var.get(),
            "cross_video_reid": self.cross_video_reid_var.get(),
            "reid_model_tier": str(self.reid_model_tier_var.get().strip() or "balanced"),
            "video_io_prefetch": self.video_io_prefetch_var.get(),
            "live_trace": self.live_trace_var.get(),
            "retry_checkpoint_step_pct": int(self.retry_checkpoint_step_pct_var.get()),
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
            self._update_face_engine_status_text()
            return

        self._runtime_refresh_token += 1
        token = self._runtime_refresh_token
        self.runtime_var.set("Runtime: checking in background...")
        selected_engine = "InsightFace" if self.use_insightface_var.get() else "FaceNet (Legacy)"

        def _worker() -> None:
            message = "Runtime: check failed"
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
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(BACKEND_SCRIPT.parent),
                    check=True,
                )
                info = json.loads(completed.stdout.strip())
                acceleration = info.get("acceleration", "Unknown")
                reason = info.get("reason", "")
                device_names = info.get("device_names") or []
                if device_names:
                    device_text = ", ".join(device_names)
                    message = f"Runtime: {acceleration} | {device_text}"
                else:
                    message = f"Runtime: {acceleration} | {reason}"
            except Exception as exc:
                if isinstance(exc, subprocess.CalledProcessError):
                    detail = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()[-1]
                    message = f"Runtime: check failed | {detail}"
                else:
                    message = f"Runtime: check failed ({exc})"

            self.runtime_refresh_queue.put((token, message, selected_engine))

        threading.Thread(target=_worker, daemon=True, name="runtime-refresh").start()

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

        if self.live_trace_var.get() and max_workers > 1:
            max_workers = 1
            self.max_workers_var.set("1")
            self._append_log(
                "[INFO] Live Preview requires single worker. Auto-set Max Workers to 1 for this run.\n"
            )
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
            "-u",
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
            "--retry-checkpoint-step-pct",
            str(int(self.retry_checkpoint_step_pct_var.get())),
            "--stop-flag-file",
            str(self.stop_flag_path),
        ]
        if not self.learning_enabled_var.get():
            command.append("--no-learning-enabled")
        if self.use_insightface_var.get():
            command.append("--use-insightface")
        else:
            command.append("--no-use-insightface")
        if self.cross_video_reid_var.get():
            command.append("--cross-video-reid")
        else:
            command.append("--no-cross-video-reid")
        command.extend(["--reid-model-tier", str(self.reid_model_tier_var.get().strip() or "balanced")])
        if self.video_io_prefetch_var.get():
            command.append("--video-io-prefetch")
        else:
            command.append("--no-video-io-prefetch")
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
        if self.review_mode_var.get():
            command.append("--review-mode")
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
            "-u",
            str(BACKEND_SCRIPT),
            "--output-dir",
            str(destination),
            *extra_args,
        ]
        if self.use_insightface_var.get():
            command.append("--use-insightface")
        else:
            command.append("--no-use-insightface")
        if not self.learning_enabled_var.get():
            command.append("--no-learning-enabled")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BACKEND_SCRIPT.parent),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
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

    def _run_backend_json_async(
        self,
        extra_args: list[str],
        on_success: Any,
        on_error: Any,
    ) -> None:
        def _worker() -> None:
            try:
                payload = self._run_backend_json(extra_args)
            except Exception as exc:
                self.root.after(0, lambda err=exc: on_error(err))
                return
            self.root.after(0, lambda data=payload: on_success(data))

        threading.Thread(target=_worker, daemon=True, name="backend-json").start()

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

    def _populate_identity_tools_from_payload(self, payload: dict) -> None:
        tree = self.identity_tree
        if tree is None:
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

    def _refresh_identity_tools(self) -> None:
        tree = self.identity_tree
        if tree is None:
            return

        self.status_var.set("Loading identity tools...")

        def _on_success(payload: dict) -> None:
            self._populate_identity_tools_from_payload(payload)
            self.status_var.set("Identity tools loaded")

        def _on_error(exc: Exception) -> None:
            self.status_var.set("Failed to load identity tools")
            messagebox.showerror(APP_TITLE, f"Failed to load identities:\n{exc}")

        self._run_backend_json_async(["--identity-list-json"], _on_success, _on_error)

    def _open_identity_tools(self) -> None:
        if self.process is not None:
            messagebox.showinfo(APP_TITLE, "Please wait until sorting finishes.")
            return
        self._build_identity_window()
        if self.identity_window is not None:
            self.identity_window.deiconify()
            self.identity_window.lift()
            self.identity_window.focus_force()
        self._refresh_identity_tools()

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
        if isinstance(details, dict):
            learning_events = int(details.get("learning_feedback_events", 0) or 0)
            learning_with_embeddings = int(details.get("learning_feedback_with_embeddings", 0) or 0)
            cache_file = str(details.get("embedding_cache_file", "")).strip()
            if learning_events > 0 or learning_with_embeddings > 0 or cache_file:
                self._append_log(
                    "[LEARNING] "
                    f"events={learning_events} with_embeddings={learning_with_embeddings} "
                    f"cache={cache_file}\n"
                )
        self._refresh_identity_tools()
        self._refresh_learning_stats()
        self._refresh_review_queue_status()
        return payload

    def _identity_merge(self) -> None:
        selected = self._selected_identity_names()
        if len(selected) < 2:
            messagebox.showinfo(APP_TITLE, "Select at least two identity folders to merge.")
            return

        default_target = selected[0]
        target = simpledialog.askstring(
            APP_TITLE,
            "Target folder name (all other selected folders will merge into this target):",
            initialvalue=default_target,
        )
        if target is None:
            return
        target = target.strip()
        if not target:
            messagebox.showerror(APP_TITLE, "Target folder name is required.")
            return
        if target not in self.identity_rows:
            messagebox.showerror(APP_TITLE, f"Target folder not found in identities: {target}")
            return

        sources = [name for name in selected if name.lower() != target.lower()]
        if not sources:
            messagebox.showerror(APP_TITLE, "Select at least one source folder different from target.")
            return

        preview_list = "\n".join(f"- {name}" for name in sources[:10])
        if len(sources) > 10:
            preview_list += f"\n... (+{len(sources) - 10} more)"
        if not messagebox.askyesno(
            APP_TITLE,
            "Merge selected folders into target?\n\n"
            f"Target: {target}\n"
            f"Sources ({len(sources)}):\n{preview_list}\n\n"
            "This also updates learning memory labels.",
        ):
            return

        merged_ok = 0
        merged_failed = 0
        total_moved = 0
        total_learning_events = 0
        total_learning_with_embeddings = 0

        for source in sources:
            args = [
                "--identity-action",
                "merge",
                "--identity-source-folder",
                source,
                "--identity-target-folder",
                target,
            ]
            try:
                payload = self._run_backend_json(args)
            except Exception as exc:
                merged_failed += 1
                self._append_log(f"[IDENTITY] merge {source} -> {target} failed: {exc}\n")
                continue

            if not bool(payload.get("ok", False)):
                merged_failed += 1
                self._append_log(
                    f"[IDENTITY] merge {source} -> {target} failed: {payload.get('error', payload)}\n"
                )
                continue

            details = payload.get("details", {})
            moved_count = int(details.get("moved_count", 0) or 0) if isinstance(details, dict) else 0
            learning_events = int(details.get("learning_feedback_events", 0) or 0) if isinstance(details, dict) else 0
            learning_with_embeddings = (
                int(details.get("learning_feedback_with_embeddings", 0) or 0) if isinstance(details, dict) else 0
            )
            total_moved += moved_count
            total_learning_events += learning_events
            total_learning_with_embeddings += learning_with_embeddings
            merged_ok += 1

            self._append_log(
                f"[IDENTITY] merge {source} -> {target}: moved={moved_count} "
                f"learning_events={learning_events} with_embeddings={learning_with_embeddings}\n"
            )

        self._refresh_identity_tools()
        self._refresh_learning_stats()
        self._refresh_review_queue_status()

        messagebox.showinfo(
            APP_TITLE,
            "Merge complete.\n\n"
            f"Successful: {merged_ok}\n"
            f"Failed: {merged_failed}\n"
            f"Videos moved: {total_moved}\n"
            f"Learning events: {total_learning_events}\n"
            f"Learning events with embeddings: {total_learning_with_embeddings}",
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
        self.review_preview_label = None
        self.review_preview_image = None

    def _safe_review_label_configure(self, **kwargs: Any) -> bool:
        label = getattr(self, "review_preview_label", None)
        if label is None:
            return False
        try:
            if not bool(label.winfo_exists()):
                return False
        except Exception:
            return False
        try:
            label.configure(**kwargs)
            return True
        except tk.TclError:
            return False

    def _load_review_preview(self, video_path: Path) -> None:
        label = getattr(self, "review_preview_label", None)
        if label is None:
            return
        try:
            if not bool(label.winfo_exists()):
                return
        except Exception:
            return
        if not video_path.exists():
            self._safe_review_label_configure(text="Video not found", image="")
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
            self._safe_review_label_configure(text="Preview unavailable", image="")
            self.review_preview_image = None
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((760, 420))
        photo = ImageTk.PhotoImage(image=image)
        if not self._safe_review_label_configure(image=photo, text=""):
            self.review_preview_image = None
            return
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
                self._safe_review_label_configure(text="No pending items", image="")
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

    def _load_review_queue_async(self, *, notify_if_empty: bool) -> None:
        self.status_var.set("Loading review queue...")

        def _on_success(state: dict) -> None:
            items = state.get("items", [])
            if not isinstance(items, list):
                items = []
            self.review_items = [item for item in items if isinstance(item, dict) and item.get("status") == "pending"]
            if self.review_index >= len(self.review_items):
                self.review_index = max(0, len(self.review_items) - 1)
            self._render_review_item()
            self._refresh_review_queue_status()

            if not self.review_items:
                self.status_var.set("No pending review items")
                if notify_if_empty:
                    messagebox.showinfo(APP_TITLE, "No pending review items.")
                if self.review_window is not None and self.review_window.winfo_exists():
                    self._close_review_window()
                return

            self.status_var.set("Review queue loaded")
            if self.review_window is not None and self.review_window.winfo_exists():
                self.review_window.deiconify()
                self.review_window.lift()
                self.review_window.focus_force()

        def _on_error(exc: Exception) -> None:
            self.status_var.set("Failed to load review queue")
            messagebox.showerror(APP_TITLE, f"Failed to load review queue:\n{exc}")
            if self.review_window is not None and self.review_window.winfo_exists():
                self._close_review_window()

        self._run_backend_json_async(["--review-list-json"], _on_success, _on_error)

    def _open_review_queue(self) -> None:
        if self.process is not None:
            messagebox.showinfo(APP_TITLE, "Please wait until sorting finishes.")
            return

        self.review_index = 0
        self._build_review_window()
        if self.review_window is not None and self.review_window.winfo_exists():
            self.review_window.deiconify()
            self.review_window.lift()
            self.review_window.focus_force()
        self.review_items = []
        self._render_review_item()
        self._load_review_queue_async(notify_if_empty=True)

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

        self._load_review_queue_async(notify_if_empty=False)

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
        self._update_face_engine_status_text("starting...")
        self.progress_status_var.set("Progress: 0/0 videos")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal", text="Stop (Save Progress)")
        self.open_review_button.configure(state="disabled")
        self.identity_tools_button.configure(state="disabled")
        self.duplicate_finder_button.configure(state="disabled")
        self.reprocess_uncertain_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=100)
        self._clear_results_table()
        self._dashboard_reset()

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
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
                token, message, selected_engine = self.runtime_refresh_queue.get_nowait()
            except queue.Empty:
                break
            if token != self._runtime_refresh_token:
                continue
            self.runtime_var.set(message)
            self._update_face_engine_status_text(selected_engine)

        log_chunks: list[str] = []
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
                self._dashboard_maybe_refresh_projection(force=True)
            else:
                result_payload = parse_result_json_line(item)
                if result_payload is not None:
                    self._append_result_row(result_payload)
                    self._dashboard_update_from_result(result_payload)
                    continue
                if item.startswith("[TRACE] "):
                    self.live_status_var.set(item.replace("[TRACE] ", "Live: ").strip())
                if "[INFO] InsightFace initialized:" in item:
                    self._update_face_engine_status_text("InsightFace")
                elif "[INFO] FaceNet initialized:" in item:
                    self._update_face_engine_status_text("FaceNet (Fallback)")
                elif "InsightFace runtime failed, switching to FaceNet fallback" in item:
                    self._update_face_engine_status_text("FaceNet (Fallback)")
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
                    self._dashboard_update_from_progress(done, total)
                log_chunks.append(item)

        if log_chunks:
            self._append_log(combine_log_chunks(log_chunks))

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
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(bg=DARK_BG)

    style.configure(
        ".",
        background=DARK_BG,
        foreground=DARK_TEXT,
        fieldbackground=DARK_SURFACE,
        bordercolor=DARK_BORDER,
        lightcolor=DARK_BORDER,
        darkcolor=DARK_BORDER,
        troughcolor=DARK_SURFACE,
        focuscolor=DARK_ACCENT,
    )
    style.map(".", foreground=[("disabled", DARK_MUTED)])

    style.configure("TFrame", background=DARK_BG)
    style.configure("TLabelframe", background=DARK_BG, foreground=DARK_TEXT, bordercolor=DARK_BORDER)
    style.configure("TLabelframe.Label", background=DARK_BG, foreground=DARK_TEXT)
    style.configure("TLabel", background=DARK_BG, foreground=DARK_TEXT)
    style.configure("TCheckbutton", background=DARK_BG, foreground=DARK_TEXT)

    style.configure(
        "TButton",
        background=DARK_SURFACE,
        foreground=DARK_TEXT,
        bordercolor=DARK_BORDER,
        focusthickness=1,
        focuscolor=DARK_ACCENT,
        padding=(8, 6),
    )
    style.map(
        "TButton",
        background=[("active", DARK_ACCENT_ACTIVE), ("pressed", DARK_ACCENT)],
        foreground=[("disabled", DARK_MUTED)],
    )

    style.configure(
        "TEntry",
        fieldbackground="#11161d",
        foreground=DARK_TEXT,
        bordercolor=DARK_BORDER,
        insertcolor=DARK_TEXT,
    )
    style.configure(
        "TCombobox",
        fieldbackground="#11161d",
        background=DARK_SURFACE,
        foreground=DARK_TEXT,
        arrowcolor=DARK_TEXT,
        bordercolor=DARK_BORDER,
    )

    style.configure(
        "Vertical.TScrollbar",
        background=DARK_SURFACE,
        troughcolor="#11161d",
        bordercolor=DARK_BORDER,
        arrowcolor=DARK_TEXT,
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=DARK_SURFACE,
        troughcolor="#11161d",
        bordercolor=DARK_BORDER,
        arrowcolor=DARK_TEXT,
    )

    style.configure("TProgressbar", background=DARK_ACCENT, troughcolor="#11161d", bordercolor=DARK_BORDER)
    style.configure("Treeview", rowheight=24, background="#11161d", foreground=DARK_TEXT, fieldbackground="#11161d")
    style.map("Treeview", background=[("selected", DARK_ACCENT)], foreground=[("selected", "#ffffff")])
    style.configure(
        "Treeview.Heading",
        background=DARK_SURFACE,
        foreground=DARK_TEXT,
        bordercolor=DARK_BORDER,
        padding=(8, 4),
    )
    style.map("Treeview.Heading", background=[("active", DARK_ACCENT_ACTIVE)])

    root.option_add("*Text.Background", "#0f1318")
    root.option_add("*Text.Foreground", DARK_TEXT)
    root.option_add("*Listbox.Background", "#11161d")
    root.option_add("*Listbox.Foreground", DARK_TEXT)
    root.option_add("*Menu.Background", DARK_SURFACE)
    root.option_add("*Menu.Foreground", DARK_TEXT)
    VideoSorterGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
