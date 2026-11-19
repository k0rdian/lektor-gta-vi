"""Oznaczenia monitorów niezależne od ich rozdzielczości."""

import tkinter as tk


def monitor_label(index, monitor):
    x, y = monitor["left"], monitor["top"]
    if x == 0 and y == 0:
        position = "główny"
    else:
        directions = []
        if x < 0:
            directions.append("po lewej")
        elif x > 0:
            directions.append("po prawej")
        if y < 0:
            directions.append("u góry")
        elif y > 0:
            directions.append("na dole")
        position = ", ".join(directions)
    return f"Monitor {index} — {position}"


def identify_monitors(root, monitors):
    for index, monitor in enumerate(monitors, 1):
        badge = tk.Toplevel(root)
        badge.overrideredirect(True)
        badge.attributes("-topmost", True)
        x = monitor["left"] + monitor["width"] // 2 - 160
        y = monitor["top"] + monitor["height"] // 2 - 120
        badge.geometry(f"320x240+{x}+{y}")
        tk.Label(badge, text=str(index), bg="#153b66", fg="white", font=("Segoe UI", 90, "bold")).pack(fill="both", expand=True)
        badge.after(3500, badge.destroy)
