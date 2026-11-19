"""Widoczny panel nadzoruje osobny proces OCR i może go zawsze zatrzymać."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

import mss

from .monitors import identify_monitors, monitor_label
from .pending import PendingQueue


def stop_worker(process):
    """Zakończ również proces potomny uruchomiony przez python.exe z venv."""
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW, check=False,
        )
    else:
        process.kill()
    process.wait(timeout=5)


class ControlPanel:
    def __init__(self, root, cfg, args, argv):
        self.root, self.cfg = root, cfg
        self.argv = [item for item in argv if item != "--youtube"]
        self.process = None
        self.events = queue.Queue()
        self.ready = False
        self.paused = True
        self.generation = 0
        self.countdown_id = None
        self.queue_path = Path(__file__).resolve().parent.parent / "kolejka.json"
        self.queue_window = None
        self.queue_text = None
        self._queue_rendered = None
        with mss.MSS() as screen:
            self.monitors = [dict(item) for item in screen.monitors[1:]]
        root.title("Lektor — sterowanie")
        root.attributes("-topmost", True)
        root.resizable(False, False)
        root.geometry("+30+35")
        root.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Lektor — sterowanie", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        self.status = tk.StringVar(value="Zatrzymany — niczego nie czytam.")
        ttk.Label(frame, textvariable=self.status, wraplength=580).pack(anchor="w", pady=(6, 12))
        self.source = ttk.Combobox(frame, state="readonly", width=48, values=["YouTube / napisy na ekranie", "Karta Elgato / gra"])
        self.source.current(0 if args.youtube or cfg.source.type == "screen" else 1)
        self.source.pack(fill="x")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 12))
        self.monitor = ttk.Combobox(row, state="readonly", width=32, values=[monitor_label(i, mon) for i, mon in enumerate(self.monitors, 1)])
        if self.monitors:
            self.monitor.current(min(max(int(cfg.source.monitor), 1), len(self.monitors)) - 1)
        self.monitor.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Pokaż numery", command=lambda: identify_monitors(root, self.monitors)).pack(side="left", padx=(8, 0))
        ttk.Label(frame, text="Włącz polskie napisy i zatrzymaj film na widocznym napisie.\nPo kliknięciu „Wybierz napisy” masz 7 sekund, aby pokazać film.\nZaznacz pas napisów, zatwierdź Enterem, potem kliknij „Czytaj”.", justify="left").pack(anchor="w")
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(14, 0))
        self.select_button = ttk.Button(buttons, text="Wybierz napisy", command=self.select)
        self.select_button.pack(side="left")
        self.play_button = ttk.Button(buttons, text="Czytaj", command=self.toggle, state="disabled")
        self.play_button.pack(side="left", padx=8)
        tk.Button(buttons, text="■ ZATRZYMAJ", bg="#b42318", fg="white", font=("Segoe UI", 10, "bold"), command=self.stop).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Zamknij", command=self.close).pack(side="left")
        queue_row = ttk.Frame(frame)
        queue_row.pack(fill="x", pady=(12, 0))
        self.queue_summary = tk.StringVar(value="Kolejka: 0 kwestii")
        ttk.Label(queue_row, textvariable=self.queue_summary).pack(side="left", padx=(0, 8))
        ttk.Button(queue_row, text="Pokaż kolejkę", command=self.show_queue).pack(side="right")
        ttk.Label(frame, text="Pauza i ZATRZYMAJ zachowują kolejkę. Zaległości będą czytane po kolei.", wraplength=580).pack(anchor="w", pady=(6, 0))
        ttk.Label(frame, text="Panel pozostaje na wierzchu. Przeciągnij go poza obszar napisów.").pack(anchor="w", pady=(12, 0))
        self.last_line = tk.StringVar()
        ttk.Label(frame, textvariable=self.last_line, wraplength=580).pack(anchor="w", pady=(4, 0))
        root.after(100, self.poll)
        self.update_queue()

    def show_queue(self):
        if self.queue_window is not None and self.queue_window.winfo_exists():
            self.queue_window.lift()
            return
        self.queue_window = tk.Toplevel(self.root)
        self.queue_window.title("Lektor — zapisane kwestie")
        self.queue_window.attributes("-topmost", True)
        frame = ttk.Frame(self.queue_window, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Kolejność czytania — pierwsza pozycja może być właśnie odtwarzana.").pack(anchor="w")
        ttk.Label(frame, text="Przerwana kwestia zostanie przeczytana od początku po wznowieniu.").pack(anchor="w", pady=(0, 8))
        self.queue_text = tk.Text(frame, width=76, height=16, wrap="word", font=("Segoe UI", 11), state="disabled")
        self.queue_text.pack(fill="both", expand=True)
        ttk.Button(frame, text="Wyczyść kolejkę…", command=self.clear_queue).pack(anchor="e", pady=(10, 0))
        self._queue_rendered = None

    def clear_queue(self):
        if messagebox.askyesno("Wyczyść kolejkę", "Zatrzymać odczyt i usunąć wszystkie zapisane, nieprzeczytane kwestie?", parent=self.queue_window):
            self.stop()
            try:
                PendingQueue(self.queue_path).clear()
            except (OSError, ValueError) as exc:
                messagebox.showerror("Kolejka", str(exc), parent=self.queue_window)

    def update_queue(self):
        try:
            lines = PendingQueue(self.queue_path).snapshot()
            self.queue_summary.set(f"Kolejka: {len(lines)} kwestii")
            content = "\n\n".join(f"{index}. {line.text}" for index, line in enumerate(lines, 1)) or "Kolejka jest pusta."
        except (OSError, ValueError, KeyError) as exc:
            self.queue_summary.set("Nie można odczytać kolejki")
            content = str(exc)
        if self.queue_text is not None and self.queue_text.winfo_exists() and content != self._queue_rendered:
            self.queue_text.configure(state="normal")
            self.queue_text.delete("1.0", "end")
            self.queue_text.insert("1.0", content)
            self.queue_text.configure(state="disabled")
            self._queue_rendered = content
        self.root.after(500, self.update_queue)

    def send(self, command):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except (OSError, ValueError):
                self.stop()

    def select(self):
        self.stop()
        self.select_button.configure(state="disabled")
        self.source.configure(state="disabled")
        self.monitor.configure(state="disabled")
        self.countdown(7)

    def countdown(self, seconds):
        if seconds:
            self.status.set(f"Pokaż film / grę. Wybór obszaru za {seconds} s. Lektor milczy.")
            self.countdown_id = self.root.after(1000, self.countdown, seconds - 1)
        else:
            self.countdown_id = None
            self.launch()

    def launch(self):
        source = "screen" if self.source.current() == 0 else "capture"
        command = [sys.executable, "-u", "-m", "lektor", *self.argv, "--worker", "--start-paused", "--select", "--source", source]
        if source == "screen":
            command += ["--monitor", str(self.monitor.current() + 1)]
        try:
            self.process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parent.parent,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as exc:
            self.stop()
            self.status.set(f"Nie można uruchomić lektora: {exc}")
            return
        self.status.set("Zaznacz napisy w osobnym oknie i naciśnij Enter. Lektor milczy.")
        process, generation = self.process, self.generation

        def read():
            for line in process.stdout:
                self.events.put((generation, line.strip()))
            self.events.put((generation, None))

        threading.Thread(target=read, daemon=True).start()

    def toggle(self):
        if not self.ready:
            return
        self.play_button.configure(state="disabled")
        self.send("pause")

    def poll(self):
        try:
            while True:
                generation, line = self.events.get_nowait()
                if generation != self.generation:
                    continue
                if line is None:
                    last_line = self.last_line.get()
                    self.stop()
                    self.last_line.set(last_line)
                    self.status.set("Lektor zakończył pracę. W razie błędu zobacz komunikat poniżej.")
                    continue
                self.last_line.set(line[:220])
                message = line.partition("] ")[2]
                if message in ("Pauza — lektor milczy.", "Wznowiono czytanie."):
                    self.ready = True
                    self.paused = message == "Pauza — lektor milczy."
                    self.status.set("PAUZA — niczego nie czytam. Kliknij Czytaj." if self.paused else "CZYTAM wybrany obszar — Pauza natychmiast ucisza głos.")
                    self.play_button.configure(text="Czytaj" if self.paused else "Pauza", state="normal")
                    self.select_button.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def stop(self):
        self.generation += 1
        if self.countdown_id is not None:
            self.root.after_cancel(self.countdown_id)
            self.countdown_id = None
        process, self.process = self.process, None
        stop_worker(process)
        self.ready = False
        self.paused = True
        self.status.set("Zatrzymany — niczego nie czytam.")
        self.last_line.set("")
        self.play_button.configure(text="Czytaj", state="disabled")
        self.select_button.configure(state="normal")
        self.source.configure(state="readonly")
        self.monitor.configure(state="readonly")

    def close(self):
        self.stop()
        self.root.destroy()


def run_panel(cfg, args, argv):
    import ctypes

    mutex = None
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        mutex = kernel.CreateMutexW(None, False, "Local\\LektorGtaVIControlPanel")
        already_running = ctypes.get_last_error() == 183
    else:
        already_running = False
    root = tk.Tk()
    if already_running:
        root.withdraw()
        messagebox.showinfo("Lektor", "Panel lektora jest już otwarty. Znajdź okno „Lektor — sterowanie” na pasku zadań.")
        root.destroy()
        if mutex:
            kernel.CloseHandle(ctypes.c_void_p(mutex))
        return 0
    panel = None
    try:
        panel = ControlPanel(root, cfg, args, argv)
        root.mainloop()
    finally:
        if panel is not None:
            stop_worker(panel.process)
        if mutex:
            kernel.CloseHandle(ctypes.c_void_p(mutex))
    return 0
