"""Checkpoint -> LoRA — a small standalone Fizgig utility.

Point it at the base model a fine-tune started from and the checkpoint it produced, tick
the ranks you want, and it writes one ordinary LoRA per rank. The SVD runs once per layer
and is sliced per rank, so asking for four ranks costs about what one does.

Deliberately its own window rather than another tab: it is a post-training tool with three
inputs, and it has no business loading the trainer's models. Same venv, same look.

    venv\\Scripts\\python.exe diff_to_lora_gui.py
"""

import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Fizgig's palette, borrowed from the main GUI so the two match. Falls back to a copy if
# the trainer module can't be imported, so this tool stands alone.
try:
    from lora_trainer_gui import COLORS, FONT_FAMILY
except Exception:
    FONT_FAMILY = "Segoe UI"
    COLORS = {
        "bg_deep": "#1E2530", "bg_surface": "#252D38", "bg_hover": "#2A3542",
        "bg_header": "#1A2028", "text_primary": "#F0F4F8", "text_secondary": "#8A9BAE",
        "text_muted": "#5A6B7E", "accent": "#3B82F6", "accent_hover": "#60A5FA",
        "accent_subtle": "#1E3A5F", "border": "#3A4555", "border_focus": "#3B82F6",
        "success": "#10B981", "warning": "#F59E0B", "error": "#EF4444",
    }

RANKS = [8, 16, 32, 64, 128, 256]
DEFAULT_ON = {32, 64}
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".diff_to_lora.json")


class DiffToLoRAApp:
    def __init__(self, master):
        self.master = master
        master.title("Fizgig — Checkpoint to LoRA")
        master.geometry("880x760")
        master.minsize(760, 640)
        master.configure(bg=COLORS["bg_deep"])

        self.q = queue.Queue()
        self.worker = None
        self._stop = False
        self._load_config()
        self._build_styles()
        self._build_ui()
        self.master.after(100, self._drain)

    # ---------- persistence ----------
    def _load_config(self):
        self.cfg = {"base": "", "tuned": "", "out": "", "name": "extracted"}
        try:
            import json
            with open(CONFIG, encoding="utf-8") as f:
                self.cfg.update(json.load(f))
        except Exception:
            pass

    def _save_config(self):
        try:
            import json
            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump({"base": self.base_var.get(), "tuned": self.tuned_var.get(),
                           "out": self.out_var.get(), "name": self.name_var.get()}, f, indent=2)
        except Exception:
            pass

    # ---------- chrome ----------
    def _build_styles(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TFrame", background=COLORS["bg_surface"])
        st.configure("TLabel", background=COLORS["bg_surface"],
                     foreground=COLORS["text_secondary"], font=(FONT_FAMILY, 10))
        st.configure("TCheckbutton", background=COLORS["bg_surface"],
                     foreground=COLORS["text_primary"], font=(FONT_FAMILY, 10))
        st.map("TCheckbutton", background=[("active", COLORS["bg_surface"])])
        st.configure("TEntry", fieldbackground=COLORS["bg_deep"],
                     foreground=COLORS["text_primary"], insertcolor=COLORS["text_primary"])
        st.configure("Go.TButton", font=(FONT_FAMILY, 11, "bold"), padding=8)
        st.configure("Horizontal.TProgressbar", background=COLORS["accent"],
                     troughcolor=COLORS["bg_deep"], borderwidth=0)

    def _card(self, parent, title, desc=None):
        shell = tk.Frame(parent, bg=COLORS["bg_surface"], highlightthickness=1,
                         highlightbackground=COLORS["border"])
        shell.pack(fill=tk.X, padx=28, pady=(0, 14))
        inner = tk.Frame(shell, bg=COLORS["bg_surface"])
        inner.pack(fill=tk.X, padx=20, pady=16)
        tk.Label(inner, text=title, font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_surface"]).pack(anchor="w")
        if desc:
            tk.Label(inner, text=desc, font=(FONT_FAMILY, 9), fg=COLORS["text_secondary"],
                     bg=COLORS["bg_surface"], wraplength=760, justify="left").pack(
                anchor="w", pady=(2, 10))
        body = tk.Frame(inner, bg=COLORS["bg_surface"])
        body.pack(fill=tk.X)
        return body

    def _file_row(self, parent, label, var, row, kind="file"):
        tk.Label(parent, text=label, font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"],
                 bg=COLORS["bg_surface"]).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=var, width=58).grid(row=row, column=1, sticky="ew",
                                                           padx=8, pady=5)

        def pick():
            if kind == "dir":
                p = filedialog.askdirectory(initialdir=var.get() or None)
            else:
                p = filedialog.askopenfilename(
                    initialdir=os.path.dirname(var.get()) if var.get() else None,
                    filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")])
            if p:
                var.set(p)
        ttk.Button(parent, text="Browse", command=pick, width=9).grid(row=row, column=2, pady=5)
        parent.columnconfigure(1, weight=1)

    def _build_ui(self):
        outer = tk.Frame(self.master, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        head = tk.Frame(outer, bg=COLORS["bg_deep"])
        head.pack(fill=tk.X, padx=28, pady=(22, 16))
        tk.Label(head, text="Checkpoint to LoRA", font=(FONT_FAMILY, 22, "bold"),
                 fg=COLORS["text_primary"], bg=COLORS["bg_deep"]).pack(anchor="w")
        tk.Label(head, text="Turn a full fine-tuned checkpoint into an ordinary, shareable "
                            "LoRA by extracting its difference from the base it started from.",
                 font=(FONT_FAMILY, 11), fg=COLORS["text_secondary"],
                 bg=COLORS["bg_deep"], wraplength=800, justify="left").pack(anchor="w", pady=(4, 0))

        c = self._card(outer, "Models",
                       "Both files must be the same architecture. The base is whatever the "
                       "fine-tune STARTED from — if you continued a run, that is the checkpoint "
                       "you continued from, not the original.")
        self.base_var = tk.StringVar(value=self.cfg["base"])
        self.tuned_var = tk.StringVar(value=self.cfg["tuned"])
        self._file_row(c, "Base checkpoint:", self.base_var, 0)
        self._file_row(c, "Trained checkpoint:", self.tuned_var, 1)

        c = self._card(outer, "Ranks",
                       "One file per ticked rank, all from a single SVD pass — four ranks cost "
                       "about what one does. Higher rank = closer to the full fine-tune and a "
                       "bigger file; on a 3-subject Krea 2 run, 64 was indistinguishable.")
        rr = tk.Frame(c, bg=COLORS["bg_surface"])
        rr.pack(anchor="w")
        self.rank_vars = {}
        for r in RANKS:
            v = tk.BooleanVar(value=(r in DEFAULT_ON))
            self.rank_vars[r] = v
            ttk.Checkbutton(rr, text=str(r), variable=v).pack(side=tk.LEFT, padx=(0, 18))

        c = self._card(outer, "Output")
        self.out_var = tk.StringVar(value=self.cfg["out"])
        self.name_var = tk.StringVar(value=self.cfg["name"] or "extracted")
        self._file_row(c, "Folder:", self.out_var, 0, kind="dir")
        tk.Label(c, text="Name prefix:", font=(FONT_FAMILY, 10), fg=COLORS["text_secondary"],
                 bg=COLORS["bg_surface"]).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(c, textvariable=self.name_var, width=30).grid(row=1, column=1, sticky="w",
                                                                padx=8, pady=5)
        tk.Label(c, text="saved as  <prefix>_<timestamp>_r<rank>.safetensors",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_muted"],
                 bg=COLORS["bg_surface"]).grid(row=2, column=1, sticky="w", padx=8)

        bar = tk.Frame(outer, bg=COLORS["bg_deep"])
        bar.pack(fill=tk.X, padx=28, pady=(4, 10))
        self.go_btn = ttk.Button(bar, text="Extract LoRAs", style="Go.TButton",
                                 command=self._start)
        self.go_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self._request_stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        self.status = tk.Label(bar, text="Ready", font=(FONT_FAMILY, 10),
                               fg=COLORS["text_secondary"], bg=COLORS["bg_deep"])
        self.status.pack(side=tk.LEFT, padx=14)

        self.prog = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.prog.pack(fill=tk.X, padx=28, pady=(0, 10))

        logwrap = tk.Frame(outer, bg=COLORS["bg_surface"], highlightthickness=1,
                           highlightbackground=COLORS["border"])
        logwrap.pack(fill=tk.BOTH, expand=True, padx=28, pady=(0, 22))
        self.log = tk.Text(logwrap, height=12, bg=COLORS["bg_deep"],
                           fg=COLORS["text_primary"], insertbackground=COLORS["text_primary"],
                           font=("Consolas", 9), relief=tk.FLAT, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log.configure(state=tk.DISABLED)

    # ---------- plumbing ----------
    def _say(self, msg):
        self.q.put(("log", msg))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.configure(state=tk.NORMAL)
                    self.log.insert(tk.END, payload + "\n")
                    self.log.see(tk.END)
                    self.log.configure(state=tk.DISABLED)
                elif kind == "progress":
                    done, total, name = payload
                    self.prog["value"] = done / max(total, 1) * 100
                    self.status.config(text=f"{done}/{total}  {name[:46]}")
                elif kind == "done":
                    self._finish(payload)
        except queue.Empty:
            pass
        self.master.after(100, self._drain)

    def _finish(self, result):
        self.go_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        ok, info = result
        if ok:
            self.prog["value"] = 100
            self.status.config(text="Done", fg=COLORS["success"])
            if info and messagebox.askyesno("Extraction complete",
                                            f"{len(info)} file(s) written.\n\nOpen the folder?"):
                try:
                    os.startfile(os.path.dirname(info[0]))
                except Exception:
                    pass
        else:
            self.status.config(text="Failed", fg=COLORS["error"])
            messagebox.showerror("Extraction failed", str(info))
            self.status.config(fg=COLORS["text_secondary"])

    def _request_stop(self):
        self._stop = True
        self._say("stopping after the current layer...")

    def _start(self):
        base, tuned = self.base_var.get().strip(), self.tuned_var.get().strip()
        out, name = self.out_var.get().strip(), (self.name_var.get().strip() or "extracted")
        ranks = [r for r, v in self.rank_vars.items() if v.get()]

        for path, what in ((base, "Base checkpoint"), (tuned, "Trained checkpoint")):
            if not path or not os.path.isfile(path):
                messagebox.showerror("Missing file", f"{what} not found:\n{path or '(empty)'}")
                return
        if os.path.abspath(base) == os.path.abspath(tuned):
            messagebox.showerror("Same file", "Base and trained checkpoint are the same file.")
            return
        if not ranks:
            messagebox.showerror("No ranks", "Tick at least one rank.")
            return
        if not out:
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_loras")
            self.out_var.set(out)
        self._save_config()

        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        self.prog["value"] = 0
        self._stop = False
        self.go_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status.config(text="Loading...", fg=COLORS["text_secondary"])

        self.worker = threading.Thread(target=self._work, args=(base, tuned, out, ranks, name),
                                       daemon=True)
        self.worker.start()

    def _work(self, base, tuned, out, ranks, name):
        try:
            from fizgig.extraction.model_diff import extract_diff_loras
            paths = extract_diff_loras(
                base, tuned, out, ranks, name=name,
                progress=lambda d, t, k: self.q.put(("progress", (d, t, k))),
                log=self._say,
                should_stop=lambda: self._stop,
            )
            self.q.put(("done", (True, paths)))
        except Exception as e:
            import traceback
            self._say(traceback.format_exc())
            self.q.put(("done", (False, e)))


if __name__ == "__main__":
    root = tk.Tk()
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("fizgig.diff.to.lora")
    except Exception:
        pass
    ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    if os.path.exists(ico):
        try:
            root.iconbitmap(ico)
        except Exception:
            pass
    DiffToLoRAApp(root)
    root.mainloop()
