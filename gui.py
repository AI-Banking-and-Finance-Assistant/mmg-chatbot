
    python gui.py --employee E1234           # set your employee ID

Tabs: "Assistant" (chat, quick questions, customer buttons, Resolve / Approve dialogs) and
"Flagged customers" (colour-coded sortable table; double-click a row for the full assessment).
tkinter ships with Python; on Linux: sudo apt install python3-tk
"""
import sys

from bank_distress_bot import HELP, load_backend

QUICK = [("Customers at risk", "Find customers at risk"),
         ("In financial distress", "Find customers already in financial distress"),
         ("Late payments", "Find customers with late payments"),
         ("High credit utilisation", "Find customers with high credit utilisation"),
         ("Low balance", "Find customers with low balance"),
         ("Overdrawn", "Find customers overdrawn"),
         ("Multiple warning signs", "Show customers with multiple warning signs"),
         ("More results", "more"),
         ("How flagging works", "How does flagging work?"),
         ("Help", "help")]

CUSTOMER_ACTIONS = [("Check", "Check customer {id}"), ("Why flagged?", "Why was customer {id} flagged?"),
                    ("Support options", "What support could be relevant for customer {id}?"),
                    ("Contact details", "Show contact details for customer {id}"),
                    ("Trend", "Is customer {id} improving?")]

COLUMNS = {"id": ("Customer", 80), "name": ("Name", 160), "status": ("Status", 140), "score": ("Score", 60),
           "signals": ("Warning signs", 520), "model": ("ML est.", 70)}
BOLD = ("TkDefaultFont", 10, "bold")


def main() -> None:
    be = load_backend(__doc__)
    print(be.banner())
    try:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext, simpledialog, ttk
    except ImportError:
        sys.exit("tkinter is not installed. Windows/macOS: reinstall Python from python.org (tick 'tcl/tk').\n"
                 "Linux: sudo apt install python3-tk\nOr use the terminal version: python bank_distress_bot.py")

    root = tk.Tk()
    root.title("Financial-Distress Early-Warning Assistant")
    root.geometry("1180x740")
    root.minsize(900, 560)
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    # ---- top bar + tabs
    top = ttk.Frame(root, padding=(10, 8))
    top.pack(fill="x")
    ttk.Label(top, text="Employee ID:").pack(side="left")
    emp_var = tk.StringVar(value=be.session.employee_id or "")
    emp_entry = ttk.Entry(top, textvariable=emp_var, width=12)
    emp_entry.pack(side="left", padx=(6, 16))
    summary_var = tk.StringVar(value=be.summary())
    ttk.Label(top, textvariable=summary_var).pack(side="left")

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(root, textvariable=status_var, relief="sunken", anchor="w", padding=(8, 2)).pack(fill="x", side="bottom")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=10, pady=(0, 6))
    chat_tab, table_tab = ttk.Frame(nb, padding=6), ttk.Frame(nb, padding=6)
    nb.add(chat_tab, text="  Assistant  ")
    nb.add(table_tab, text="  Flagged customers  ")

    # ---- chat tab: left panel
    left = ttk.Frame(chat_tab, width=240)
    left.pack(side="left", fill="y", padx=(0, 8))
    ttk.Label(left, text="Quick questions", font=BOLD).pack(anchor="w", pady=(0, 4))
    for label, cmd in QUICK:
        ttk.Button(left, text=label, command=lambda c=cmd: send(c)).pack(fill="x", pady=1)

    ttk.Separator(left).pack(fill="x", pady=8)
    ttk.Label(left, text="Customer", font=BOLD).pack(anchor="w")
    cid_var = tk.StringVar()
    cid_entry = ttk.Entry(left, textvariable=cid_var)
    cid_entry.pack(fill="x", pady=(2, 4))
    ttk.Label(left, text="Enter a customer ID, then:", foreground="#555").pack(anchor="w")
    for label, tpl in CUSTOMER_ACTIONS:
        ttk.Button(left, text=label, command=lambda t=tpl: with_customer(lambda cid: send(t.format(id=cid)))).pack(fill="x", pady=1)

    ttk.Separator(left).pack(fill="x", pady=8)
    ttk.Label(left, text="Flag reviews", font=BOLD).pack(anchor="w")
    ttk.Button(left, text="Resolve flag...", command=lambda: with_customer(resolve)).pack(fill="x", pady=1)
    ttk.Button(left, text="Supervisor approve...", command=lambda: with_customer(approve)).pack(fill="x", pady=1)

    # ---- chat tab: transcript + input
    right = ttk.Frame(chat_tab)
    right.pack(side="left", fill="both", expand=True)
    font = ("Consolas", 10) if sys.platform == "win32" else ("TkFixedFont", 10)
    log = scrolledtext.ScrolledText(right, wrap="word", state="disabled", font=font, padx=8, pady=8)
    log.pack(fill="both", expand=True)
    log.tag_configure("user", foreground="#0b5cad", font=BOLD)
    log.tag_configure("sys", foreground="#777777", font=("TkDefaultFont", 9, "italic"))

    row = ttk.Frame(right, padding=(0, 6, 0, 0))
    row.pack(fill="x")
    msg_var = tk.StringVar()
    msg_entry = ttk.Entry(row, textvariable=msg_var)
    msg_entry.pack(side="left", fill="x", expand=True)
    ttk.Button(row, text="Send", command=lambda: on_send()).pack(side="left", padx=(6, 0))
    ttk.Button(row, text="Clear", command=lambda: clear_log()).pack(side="left", padx=(6, 0))

    # ---- flagged-customers tab
    bar = ttk.Frame(table_tab)
    bar.pack(fill="x", pady=(0, 6))
    only_distress = tk.BooleanVar(value=False)
    ttk.Checkbutton(bar, text="Financial distress only", variable=only_distress,
                    command=lambda: refresh_table()).pack(side="left")
    ttk.Button(bar, text="Refresh", command=lambda: refresh_table()).pack(side="left", padx=8)
    count_var = tk.StringVar()
    ttk.Label(bar, textvariable=count_var).pack(side="left", padx=8)
    ttk.Label(bar, text="Double-click a row to open the full assessment.", foreground="#555").pack(side="right")

    tree = ttk.Treeview(table_tab, columns=list(COLUMNS), show="headings", selectmode="browse")
    sort_desc: dict = {}
    for c, (title, width) in COLUMNS.items():
        tree.heading(c, text=title, command=lambda c=c: sort_by(c))
        tree.column(c, width=width, anchor="w" if c in ("name", "signals", "status") else "center")
    scroll = ttk.Scrollbar(table_tab, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True)
    tree.tag_configure("FINANCIAL DISTRESS", background="#fde2e2")
    tree.tag_configure("EARLY WARNING", background="#fff4d6")

    # ---- behaviour
    def write(text: str, tag: str = "") -> None:
        log.configure(state="normal")
        log.insert("end", text + "\n\n", tag)
        log.configure(state="disabled")
        log.see("end")

    def clear_log() -> None:
        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")

    def send(text: str) -> None:
        text = text.strip()
        if not text:
            return
        be.set_employee(emp_var.get())
        write(f"You: {text}", "user")
        status_var.set("Working...")
        root.update_idletasks()
        write(be.ask(text))
        status_var.set("Ready.")
        summary_var.set(be.summary())
        refresh_table()

    def on_send(_event=None) -> None:
        text, _ = msg_var.get(), msg_var.set("")
        send(text)

    def with_customer(action) -> None:
        """Run action(customer_id) once a numeric customer ID has been typed in the side panel."""
        cid = cid_var.get().strip().lstrip("#")
        if cid.isdigit():
            action(cid)
        else:
            messagebox.showinfo("Customer ID needed", "Please type a numeric customer ID first, e.g. 10452.")
            cid_entry.focus_set()

    def resolve(cid: str) -> None:
        emp = emp_var.get().strip()
        if not emp:
            messagebox.showinfo("Employee ID needed", "Enter your employee ID in the top bar first.")
            return emp_entry.focus_set()
        note = simpledialog.askstring("Resolve flag", f"Review note for customer {cid}\n"
                                      "(what was reviewed and what action was taken):", parent=root)
        if note and note.strip():
            send(f"resolve flag for {cid} by {emp}: {note.strip()}")

    def approve(cid: str) -> None:
        sup = simpledialog.askstring("Supervisor approval", "Supervisor ID:", parent=root)
        if sup and sup.strip():
            send(f"approve {cid} by {sup.strip()}")

    def refresh_table() -> None:
        rows = be.flagged_rows(only_distress.get())
        tree.delete(*tree.get_children())
        for r in rows:
            tree.insert("", "end", values=r, tags=(r[2],))
        count_var.set(f"{len(rows):,} active flag(s)")

    def sort_by(col: str) -> None:
        numeric = col in ("id", "score", "model")
        key = (lambda t: float(t[0].strip("%-") or -1)) if numeric else (lambda t: t[0].lower())
        items = sorted(((tree.set(k, col), k) for k in tree.get_children("")), key=key, reverse=sort_desc.get(col, False))
        for i, (_, k) in enumerate(items):
            tree.move(k, "", i)
        sort_desc[col] = not sort_desc.get(col, False)

    def open_selected(_event=None) -> None:
        sel = tree.selection()
        if sel:
            cid = str(tree.set(sel[0], "id"))
            nb.select(chat_tab)
            cid_var.set(cid)
            send(f"Check customer {cid}")

    tree.bind("<Double-1>", open_selected)
    msg_entry.bind("<Return>", on_send)
    cid_entry.bind("<Return>", lambda e: with_customer(lambda cid: send(CUSTOMER_ACTIONS[0][1].format(id=cid))))

    write(be.banner(), "sys")
    write(HELP)
    refresh_table()
    msg_entry.focus_set()
    root.mainloop()


if __name__ == "__main__":
    main()
