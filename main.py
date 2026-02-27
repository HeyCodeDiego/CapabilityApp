# -*- coding: utf-8 -*-
"""
Capabilidade – Relatório (Standalone, PyQt5)

Correções/Novos recursos:
- Menu "Info" na barra superior e botão "Info" no painel (mensagem:
  "Aplicativo desenvolvido por Diego Siste Barbosa.").
- Resumo – Visao Geral: título desenhado dentro do bloco de duas colunas
  (sem sobreposição), com quebra de linha por palavra e paginação segura.
- Resumo – por Amostra: mais espaço após o título e cabeçalho redesenhado a
  cada quebra de página.
- Mantidos: JSON/CSV/índice, carregamento e exportação PDF, figuras.
- Rótulos ASCII no PDF por compatibilidade de fonte.
"""

import sys, os, math, tempfile, shutil, datetime, csv, json, re
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QComboBox, QSpinBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QGroupBox
)
import matplotlib
matplotlib.use("Agg")  # gerar imagens para PDF
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth

# --- Constantes CEP por K (K=2..10 e 25) ---
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308, 25: 0.153}
D3 = {2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223, 25: 0.459}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777, 25: 1.541}
d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 25: 3.931}
K_ALLOWED = sorted(A2.keys())

# --- Pastas do Arquivário ---
ARQ_ROOT = os.path.join(os.getcwd(), "Capabilidade_Arquivario")
ARQ_DATA = os.path.join(ARQ_ROOT, "dados_editaveis")
ARQ_PDF  = os.path.join(ARQ_ROOT, "pdf")
ARQ_IDX  = os.path.join(ARQ_ROOT, "arquivario_index.csv")
os.makedirs(ARQ_DATA, exist_ok=True)
os.makedirs(ARQ_PDF,  exist_ok=True)

ASCII_PDF = True  # manter rótulos ASCII

# ---------------- Utilidades -----------------
def is_num(v):
    try:
        return v is not None and (v == v)
    except Exception:
        return False

def parse_float(txt: str):
    if txt is None:
        raise ValueError("valor vazio")
    return float(str(txt).strip().replace(",", "."))

def slugify(text: str, maxlen=80):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text[:maxlen] if text else "relatorio"

# ---------------- Cálculo -----------------
def compute_stats(samples_RxK, lie=None, lse=None):
    if not samples_RxK or not isinstance(samples_RxK, list):
        raise ValueError("Matriz vazia (R×K)")
    R = len(samples_RxK)
    K = len(samples_RxK[0]) if R > 0 and isinstance(samples_RxK[0], list) else 0
    if R <= 0 or K <= 0:
        raise ValueError("Dimensoes invalidas")
    if K not in K_ALLOWED:
        raise ValueError(f"K={K} nao suportado: {K_ALLOWED}")
    for r, row in enumerate(samples_RxK, 1):
        if not isinstance(row, list) or len(row) != K:
            raise ValueError(f"Linha {r} invalida")
        for c, v in enumerate(row, 1):
            if not is_num(v):
                raise ValueError(f"Valor invalido em Amostra {r}, Sub {c}")

    xbars   = [float(sum(row)) / K for row in samples_RxK]
    rvals   = [float(max(row) - min(row)) for row in samples_RxK]
    xbarbar = float(sum(xbars)) / R
    rbar    = float(sum(rvals)) / R

    sigma_within = (rbar / d2[K]) if d2.get(K, 0) > 0 else float("nan")
    LICx = xbarbar - (A2.get(K, 0.0) * rbar)
    LSCx = xbarbar + (A2.get(K, 0.0) * rbar)
    LICr = D3.get(K, 0.0) * rbar
    LSCr = D4.get(K, 0.0) * rbar

    if is_num(lie) and is_num(lse) and is_num(sigma_within) and sigma_within > 0:
        Cp  = (lse - lie) / (6.0 * sigma_within)
        Cpk = min((lse - xbarbar) / (3.0 * sigma_within),
                  (xbarbar - lie) / (3.0 * sigma_within))
    else:
        Cp, Cpk = float("nan"), float("nan")

    flat = [float(v) for row in samples_RxK for v in row]
    sigma_overall = float(np.std(flat, ddof=1)) if len(flat) >= 2 else float("nan")
    if is_num(lie) and is_num(lse) and is_num(sigma_overall) and sigma_overall > 0:
        Pp  = (lse - lie) / (6.0 * sigma_overall)
        Ppk = min((lse - xbarbar) / (3.0 * sigma_overall),
                  (xbarbar - lie) / (3.0 * sigma_overall))
    else:
        Pp, Ppk = float("nan"), float("nan")

    return {
        "R": R, "K": K, "xbars": xbars, "rvals": rvals,
        "xbarbar": xbarbar, "rbar": rbar,
        "sigma_within": sigma_within, "sigma_overall": sigma_overall,
        "Cp": Cp, "Cpk": Cpk, "Pp": Pp, "Ppk": Ppk,
        "LICx": LICx, "LSCx": LSCx, "LICr": LICr, "LSCr": LSCr,
        "lie": lie, "lse": lse, "flat": flat
    }

# ---------------- Figuras -----------------
def build_figures(stats):
    R = stats["R"]; K = stats["K"]
    xbars = stats["xbars"]; rvals = stats["rvals"]
    xbarbar = stats["xbarbar"]; rbar = stats["rbar"]
    LICx, LSCx = stats["LICx"], stats["LSCx"]
    LICr, LSCr = stats["LICr"], stats["LSCr"]
    sigma_within = stats["sigma_within"]
    Cp, Cpk = stats["Cp"], stats["Cpk"]; Pp, Ppk = stats["Pp"], stats["Ppk"]
    lie, lse = stats["lie"], stats["lse"]
    flat = stats["flat"]

    xs = list(range(1, R + 1))
    figs = []

    # X-barra
    fig1, ax1 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    ax1.plot(xs, xbars, marker="o", linestyle="-", color="#0057B7", label="X-barra (por amostra)")
    ax1.axhline(xbarbar, color="green", linestyle=":", label="Centro (X-barra-barra)")
    ax1.axhline(LICx, color="red", linestyle="--", label="LICx")
    ax1.axhline(LSCx, color="red", linestyle="--", label="LSCx")
    if is_num(lie): ax1.axhline(lie, color="#9E9E9E", linestyle="--", linewidth=1, label="LIE")
    if is_num(lse): ax1.axhline(lse, color="#9E9E9E", linestyle="--", linewidth=1, label="LSE")
    ax1.set_title(f"Grafico X-barra (K={K})")
    ax1.set_xlabel("Amostras"); ax1.set_ylabel("Media")
    ax1.grid(True, linestyle="--", alpha=0.35); ax1.legend(); ax1.set_xticks(xs)
    figs.append(fig1)

    # R
    fig2, ax2 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    ax2.plot(xs, rvals, marker="o", linestyle="-", color="#FF6F00", label="R (por amostra)")
    ax2.axhline(rbar, color="green", linestyle=":", label="Centro (R-barra)")
    ax2.axhline(LICr, color="red", linestyle="--", label="LICr")
    ax2.axhline(LSCr, color="red", linestyle="--", label="LSCr")
    ax2.set_title("Grafico das Amplitudes (R)")
    ax2.set_xlabel("Amostras"); ax2.set_ylabel("Amplitude")
    ax2.grid(True, linestyle="--", alpha=0.35); ax2.legend(); ax2.set_xticks(xs)
    figs.append(fig2)

    # Histograma + Normal (sigma_within) + LIE/LSE
    fig3, ax3 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    if flat:
        bins = max(10, int(len(flat) ** 0.5))
        ax3.hist(flat, bins=bins, density=True, color="#E0E0E0", edgecolor="#9E9E9E", label="Dados (hist.)")
        if is_num(sigma_within) and sigma_within > 0 and is_num(xbarbar):
            x_min = xbarbar - 4 * sigma_within; x_max = xbarbar + 4 * sigma_within
            xs_pdf = np.linspace(x_min, x_max, 400)
            pdf = (1.0 / (sigma_within * math.sqrt(2 * math.pi))) * np.exp(
                -0.5 * ((xs_pdf - xbarbar) / sigma_within) ** 2
            )
            ax3.plot(xs_pdf, pdf, color="#1565C0", linewidth=2, label="Normal ajustada (sigma_within)")
        if is_num(lie): ax3.axvline(lie, color="red", linestyle="--", linewidth=1.5, label="LIE")
        if is_num(lse): ax3.axvline(lse, color="red", linestyle="--", linewidth=1.5, label="LSE")
    ax3.set_title("Distribuicao vs Especificacoes (Normal com sigma_within)")
    ax3.set_xlabel("Valor medido"); ax3.set_ylabel("Densidade")
    ax3.grid(True, linestyle="--", alpha=0.35); ax3.legend()
    figs.append(fig3)

    # Cp / Cpk
    fig4, ax4 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    vals = [Cp, Cpk]
    top = max(1.5, (max([v for v in vals if is_num(v)] or [1.5]) * 1.2))
    ax4.bar(["Cp", "Cpk"], vals, color=["#2E7D32", "#1565C0"])
    ax4.axhline(1.33, color="red", linestyle="--", label="Meta 1,33")
    ax4.set_ylim(0, top); ax4.set_title("Capacidade (Cp, Cpk) – sigma_within")
    ax4.set_ylabel("Indice"); ax4.grid(True, axis="y", linestyle="--", alpha=0.35); ax4.legend()
    figs.append(fig4)

    # Pp / Ppk
    fig5, ax5 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    vals_pp = [Pp, Ppk]
    top_pp = max(1.5, (max([v for v in vals_pp if is_num(v)] or [1.5]) * 1.2))
    ax5.bar(["Pp", "Ppk"], vals_pp, color=["#00897B", "#6A1B9A"])
    ax5.axhline(1.33, color="red", linestyle="--", label="Meta 1,33")
    ax5.set_ylim(0, top_pp); ax5.set_title("Performance (Pp, Ppk) – sigma_overall")
    ax5.set_ylabel("Indice"); ax5.grid(True, axis="y", linestyle="--", alpha=0.35); ax5.legend()
    figs.append(fig5)

    return figs

# -------- Helpers de PDF (layout robusto) ---------
def _fmt(v, nd=6):
    try:
        if v is None: return "—"
        if isinstance(v, (float, np.floating)):
            if math.isnan(v) or math.isinf(v): return "—"
            return f"{v:.{nd}f}"
        if isinstance(v, int): return str(v)
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)

def _wrap_text(txt, max_w, font_name="Helvetica", font_size=9):
    words = str(txt).split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if stringWidth(test, font_name, font_size) <= max_w:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines or [""]

def _draw_kv_two_columns(c, kv_items, margin, page_w, page_h, header_txt=None):
    """
    Duas colunas com bloco de largura fixa por coluna, quebra por palavra,
    valores alinhados à direita e paginação segura. Se header_txt for
    informado, desenha o título e recua antes de começar os pares.
    """
    usable_w = page_w - 2*margin
    gap = 10*mm
    col_w = (usable_w - gap) / 2.0
    left_x  = margin
    right_x = margin + col_w + gap

    label_w = col_w * 0.65
    val_right_left  = left_x  + col_w
    val_right_right = right_x + col_w

    line_h = 5.0*mm
    font_name = "Helvetica"; fs_lbl = 9; fs_val = 9

    y_left = c._pagesize[1] - margin
    if header_txt:
        c.setFont("Helvetica-Bold", 13); c.drawString(margin, y_left, header_txt)
        y_left -= 7*mm
    else:
        y_left -= 7*mm
    y_right = y_left

    def draw_block(side, items, y):
        for label, val in items:
            lines = _wrap_text(str(label)+":", label_w, font_name, fs_lbl)
            block_h = line_h * max(1, len(lines))
            if y - block_h < margin:
                c.showPage()
                y = c._pagesize[1] - margin
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin, y, "Resumo de Parametros Calculados - continuacao")
                y -= 6*mm
            base_x = left_x if side=="left" else right_x
            v_right = val_right_left if side=="left" else val_right_right
            c.setFont(font_name, fs_lbl); yy = y
            for ln in lines:
                c.drawString(base_x, yy, ln); yy -= line_h
            c.setFont(font_name, fs_val)
            c.drawRightString(v_right, y, _fmt(val))
            y -= block_h
        return y

    half = (len(kv_items) + 1)//2
    y_left  = draw_block("left",  kv_items[:half], y_left)
    y_right = draw_block("right", kv_items[half:],  y_right)
    return min(y_left, y_right)

def _draw_table_xbar_r(c, stats, margin=15*mm, page_w=A4[0], page_h=A4[1]):
    R = stats["R"]; xbars = stats["xbars"]; rvals = stats["rvals"]
    x = margin
    y = page_h - margin
    c.showPage()
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x, y, "Resumo de Parametros Calculados - por Amostra")
    y -= 9*mm  # mais espaço após o título

    def head(y0):
        c.setFont("Helvetica-Bold", 10)
        col1, col2, col3 = x, x + 40*mm, x + 90*mm
        c.drawString(col1, y0, "Amostra")
        c.drawString(col2, y0, "X-barra (media da amostra)")
        c.drawString(col3, y0, "R (amplitude da amostra)")
        return y0 - 6*mm, col1, col2, col3

    y, col1, col2, col3 = head(y)
    c.setFont("Courier", 9)
    line_h = 4.8*mm

    for i in range(R):
        if y - line_h < margin:
            c.showPage()
            y = page_h - margin
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x, y, "Resumo - por Amostra (continuacao)")
            y -= 7*mm
            y, col1, col2, col3 = head(y)
            c.setFont("Courier", 9)
        c.drawString(col1, y, f"{i+1:>5d}")
        c.drawRightString(col2 + 35*mm, y, _fmt(xbars[i]))
        c.drawRightString(col3 + 35*mm, y, _fmt(rvals[i]))
        y -= line_h

# --------------- Exportação PDF ---------------
def export_pdf(figs, out_path, header, stats=None):
    tmpdir = tempfile.mkdtemp(prefix="capab_report_")
    try:
        imgs = []
        for i, fig in enumerate(figs, 1):
            p = os.path.join(tmpdir, f"fig_{i}.png")
            fig.savefig(p, dpi=200, bbox_inches="tight")
            imgs.append(p)

        page_w, page_h = A4
        c = pdfcanvas.Canvas(out_path, pagesize=A4)
        margin = 15*mm
        usable_w = page_w - 2*margin
        y = page_h - margin

        title = "Relatorio de Capabilidade"
        subtitle = header

        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, title); y -= 10*mm

        # Subtítulo com quebra automática
        c.setFont("Helvetica", 9)
        def wrap(text, y0):
            words = text.split(); cur = ""; lines = []
            for w in words:
                t = (cur + " " + w).strip()
                if stringWidth(t, "Helvetica", 9) <= usable_w: cur = t
                else: lines.append(cur); cur = w
            if cur: lines.append(cur)
            for ln in lines:
                c.drawString(margin, y0, ln); y0 -= 4.8*mm
            return y0

        y = wrap(subtitle, y); y -= 2*mm
        c.setFont("Helvetica", 8)
        c.drawString(margin, y, "Graficos: X-barra  |  R  |  Histograma + Normal (sigma_within)  |  Cp/Cpk  |  Pp/Ppk")
        y -= 8*mm

        # Desenho das figuras
        for img in imgs:
            ir = ImageReader(img); iw, ih = ir.getSize()
            scale = (usable_w) / float(iw); h = ih * scale
            if y - h < margin:
                c.showPage(); y = page_h - margin
                c.setFont("Helvetica-Bold", 14); c.drawString(margin, y, title); y -= 8*mm
            c.drawImage(ir, margin, y - h, width=usable_w, height=h)
            y -= (h + 8*mm)

        # Resumo
        if stats is not None:
            c.showPage()
            kv = [
                ("R (numero de amostras)", stats["R"]),
                ("K (tamanho do subgrupo)", stats["K"]),
                ("LIE", stats["lie"]),
                ("LSE", stats["lse"]),
                ("X-barra-barra (media das medias)", stats["xbarbar"]),
                ("R-barra (media das amplitudes)", stats["rbar"]),
                ("sigma_within (= R-barra/d2)", stats["sigma_within"]),
                ("sigma_overall (desv.pad. total)", stats["sigma_overall"]),
                ("LICx", stats["LICx"]), ("LSCx", stats["LSCx"]),
                ("LICr", stats["LICr"]), ("LSCr", stats["LSCr"]),
                ("Cp (sigma_within)", stats["Cp"]),
                ("Cpk (sigma_within)", stats["Cpk"]),
                ("Pp (sigma_overall)", stats["Pp"]),
                ("Ppk (sigma_overall)", stats["Ppk"]),
            ]
            _draw_kv_two_columns(c, kv, margin, page_w, page_h,
                                 header_txt="Resumo de Parametros Calculados - Visao Geral")
            _draw_table_xbar_r(c, stats, margin=margin, page_w=page_w, page_h=page_h)

        c.save()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# --------------- Arquivário ---------------
def save_editable_archive(title, timestamp, stats, samples):
    R = stats["R"]; K = stats["K"]; lie = stats["lie"]; lse = stats["lse"]
    base = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{slugify(title)}_R{R}_K{K}"
    jp = os.path.join(ARQ_DATA, base + ".json")
    cp = os.path.join(ARQ_DATA, base + ".csv")
    payload = {
        "title": title,
        "timestamp": timestamp.strftime("%d/%m/%Y %H:%M:%S"),
        "R": R, "K": K, "lie": lie, "lse": lse,
        "samples": samples,
        "metrics": {
            "xbarbar": stats["xbarbar"], "rbar": stats["rbar"],
            "sigma_within": stats["sigma_within"], "sigma_overall": stats["sigma_overall"],
            "Cp": stats["Cp"], "Cpk": stats["Cpk"], "Pp": stats["Pp"], "Ppk": stats["Ppk"],
            "LICx": stats["LICx"], "LSCx": stats["LSCx"], "LICr": stats["LICr"], "LSCr": stats["LSCr"],
        },
        "version": "capabilidade-standalone-final"
    }
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["amostra", "subgrupo", "valor"])
        for r in range(R):
            for c in range(K):
                w.writerow([r + 1, c + 1, samples[r][c]])

    need_header = not os.path.exists(ARQ_IDX)
    with open(ARQ_IDX, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=';')
        if need_header:
            w.writerow(["timestamp", "titulo", "R", "K", "LIE", "LSE", "json", "csv", "pdf"])
        w.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), title, R, K, lie, lse, jp, cp, ""])
    return jp, cp, base

def update_index_with_pdf(pdf_path, base_row_match):
    if not os.path.exists(ARQ_IDX):
        return
    with open(ARQ_IDX, "r", encoding="utf-8") as f:
        rows = [line.rstrip("\n") for line in f]
    for i in range(len(rows) - 1, -1, -1):
        if i == 0: break
        parts = rows[i].split(";")
        if len(parts) >= 9:
            jp = parts[6]; cp = parts[7]
            if (base_row_match in os.path.basename(jp)) or (base_row_match in os.path.basename(cp)):
                parts[8] = pdf_path
                rows[i] = ";".join(map(str, parts))
                break
    with open(ARQ_IDX, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(r + "\n")

# --------------- Loaders ---------------
def load_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    title = obj.get("title", "Estudo de Capabilidade")
    R = int(obj.get("R", 0))
    K = int(obj.get("K", 0))
    lie = obj.get("lie", None)
    lse = obj.get("lse", None)
    samples = obj.get("samples", [])
    if not samples or R <= 0 or K <= 0:
        raise ValueError("JSON invalido: matriz/dimensoes ausentes")
    if any((len(row) != K) for row in samples):
        raise ValueError("JSON invalido: linhas != K")
    return title, R, K, lie, lse, samples

def load_from_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter=';'))
    start = 1 if rows and rows[0] and rows[0][0].lower() == "amostra" else 0
    max_r, max_c = 0, 0
    entries = []
    for line in rows[start:]:
        if len(line) < 3:
            continue
        try:
            r = int(str(line[0]).strip())
            c = int(str(line[1]).strip())
            v = parse_float(line[2])
        except Exception:
            continue
        entries.append((r, c, v))
        max_r = max(max_r, r)
        max_c = max(max_c, c)
    if max_r <= 0 or max_c <= 0:
        raise ValueError("CSV invalido ou vazio")
    samples = [[0.0 for _ in range(max_c)] for __ in range(max_r)]
    for r, c, v in entries:
        samples[r-1][c-1] = v
    base = os.path.splitext(os.path.basename(path))[0]
    title = re.sub(r"_R\d+_K\d+$", "", base).replace("_", " ") or "Estudo de Capabilidade (CSV)"
    return title, max_r, max_c, None, None, samples

# --------------- UI ---------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sigma")
        self.resize(1100, 740)

        # Menu Info
        menubar = self.menuBar()
        info_menu = menubar.addMenu("Info")
        act = info_menu.addAction("Sobre este app…")
        act.triggered.connect(self.show_info)

        # Painel superior
        top_box = QGroupBox("Parametros do Estudo (R×K)")
        lay_top = QHBoxLayout(top_box)
        self.spin_R = QSpinBox(); self.spin_R.setRange(2, 999); self.spin_R.setValue(25)
        self.cmb_K  = QComboBox(); self.cmb_K.addItems([str(k) for k in K_ALLOWED]); self.cmb_K.setCurrentText("5")
        self.edt_LIE = QLineEdit(); self.edt_LIE.setPlaceholderText("LIE (min)"); self.edt_LIE.setFixedWidth(120)
        self.edt_LSE = QLineEdit(); self.edt_LSE.setPlaceholderText("LSE (max)"); self.edt_LSE.setFixedWidth(120)
        self.edt_title = QLineEdit(); self.edt_title.setPlaceholderText("Identificacao (Produto / Plano / Item)")
        self.btn_load = QPushButton("Carregar…")
        self.btn_gen  = QPushButton("Gerar grade")
        self.btn_pdf  = QPushButton("Exportar PDF")

        lay_top.addWidget(QLabel("Amostras (R):"));   lay_top.addWidget(self.spin_R)
        lay_top.addWidget(QLabel("Subgrupo (K):"));   lay_top.addWidget(self.cmb_K)
        lay_top.addSpacing(16)
        lay_top.addWidget(QLabel("LIE:"));            lay_top.addWidget(self.edt_LIE)
        lay_top.addWidget(QLabel("LSE:"));            lay_top.addWidget(self.edt_LSE)
        lay_top.addSpacing(16)
        lay_top.addWidget(QLabel("Identificacao:"));  lay_top.addWidget(self.edt_title, 1)
        lay_top.addWidget(self.btn_load); lay_top.addWidget(self.btn_gen); lay_top.addWidget(self.btn_pdf)

        # Tabela R×K
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.table.installEventFilter(self)

        # Layout principal
        central = QWidget(); v = QVBoxLayout(central)
        v.addWidget(top_box)
        v.addWidget(QLabel("Digite as medicoes no grid (aceita virgula ou ponto). Use 'Carregar…' para reabrir um JSON/CSV do arquivario."))
        v.addWidget(self.table, 1)
        self.setCentralWidget(central)

        # Sinais
        self.btn_load.clicked.connect(self.on_load_editable)
        self.btn_gen.clicked.connect(self.generate_grid)
        self.btn_pdf.clicked.connect(self.on_export_pdf)

        self.generate_grid()

    def show_info(self):
        QMessageBox.information(self, "Info", "Aplicativo desenvolvido por Diego Siste Barbosa.")

    def generate_grid(self):
        R = int(self.spin_R.value())
        K = int(self.cmb_K.currentText())
        self.table.clear()
        self.table.setRowCount(R); self.table.setColumnCount(K)
        self.table.setHorizontalHeaderLabels([f"Sub {c+1}" for c in range(K)])
        self.table.setVerticalHeaderLabels([f"Amostra {r+1}" for r in range(R)])
        self.table.resizeColumnsToContents()

    def fill_table(self, samples):
        R = len(samples); K = len(samples[0]) if samples else 0
        self.spin_R.setValue(R)
        if str(K) in [self.cmb_K.itemText(i) for i in range(self.cmb_K.count())]:
            self.cmb_K.setCurrentText(str(K))
        else:
            raise ValueError(f"K={K} nao suportado: {K_ALLOWED}")
        self.generate_grid()
        for r in range(R):
            for c in range(K):
                self.table.setItem(r, c, QTableWidgetItem(str(samples[r][c])))

    def read_matrix(self):
        R = self.table.rowCount(); K = self.table.columnCount()
        if K not in K_ALLOWED:
            raise ValueError(f"K={K} nao suportado: {K_ALLOWED}")
        data = []
        for r in range(R):
            row = []
            for c in range(K):
                it = self.table.item(r, c)
                txt = it.text() if it else ""
                try:
                    v = parse_float(txt)
                except Exception:
                    raise ValueError(f"Valor invalido em Amostra {r+1}, Sub {c+1}.")
                row.append(v)
            data.append(row)
        return data

    def on_load_editable(self):
        start = ARQ_DATA if os.path.isdir(ARQ_DATA) else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Carregar estudo (JSON/CSV)", start, "Arquivos (*.json *.csv)")
        if not path: return
        try:
            if path.lower().endswith(".json"):
                title, R, K, lie, lse, samples = load_from_json(path)
                self.edt_title.setText(title); self.fill_table(samples)
                self.edt_LIE.setText("" if lie is None else f"{float(lie):.6f}")
                self.edt_LSE.setText("" if lse is None else f"{float(lse):.6f}")
            else:
                title, R, K, lie, lse, samples = load_from_csv(path)
                self.edt_title.setText(title); self.fill_table(samples)
                self.edt_LIE.clear(); self.edt_LSE.clear()
                QMessageBox.information(self, "Carregar CSV",
                    "O CSV foi carregado. Informe LIE e LSE antes de exportar o PDF.")
        except Exception as ex:
            QMessageBox.warning(self, "Carregar", f"Falha ao carregar: {ex}")

    def on_export_pdf(self):
        title = self.edt_title.text().strip() or "Estudo de Capabilidade"
        try:
            lie = parse_float(self.edt_LIE.text()); lse = parse_float(self.edt_LSE.text())
            if lse <= lie:
                QMessageBox.warning(self, "Parametros", "LSE deve ser maior que LIE.")
                return
        except Exception:
            QMessageBox.warning(self, "Parametros", "Informe LIE e LSE numericos.")
            return

        try:
            samples = self.read_matrix()
        except Exception as ex:
            QMessageBox.warning(self, "Grid", str(ex)); return

        try:
            stats = compute_stats(samples, lie=lie, lse=lse)
        except Exception as ex:
            QMessageBox.warning(self, "Calculo", str(ex)); return

        ts = datetime.datetime.now()
        jp, cp, base = save_editable_archive(title, ts, stats, samples)
        figs = build_figures(stats)

        fn = f"{base}.pdf"
        out, _ = QFileDialog.getSaveFileName(self, "Salvar PDF", fn, "PDF (*.pdf)")
        if not out: return

        hdr = (f"{title}  |  R={stats['R']} K={stats['K']}  "
               f"|  LIE={lie:.6f} LSE={lse:.6f}  "
               f"|  Data: {ts.strftime('%d/%m/%Y %H:%M')}")

        try:
            export_pdf(figs, out, hdr, stats)
            pdf_arch = os.path.join(ARQ_PDF, os.path.basename(fn))
            try:
                shutil.copy2(out, pdf_arch)
            except Exception:
                pdf_arch = os.path.join(ARQ_PDF, os.path.basename(out))
                shutil.copy2(out, pdf_arch)

            update_index_with_pdf(pdf_arch, base)

            QMessageBox.information(
                self, "Exportar",
                f"PDF salvo com sucesso:\n{out}\n\n"
                f"Arquivario atualizado:\n- JSON: {jp}\n- CSV: {cp}\n- PDF: {pdf_arch}\n- Indice: {ARQ_IDX}"
            )
        except Exception as ex:
            QMessageBox.warning(self, "Exportar", f"Falha ao exportar PDF:\n{ex}")

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QtCore.QEvent.KeyPress:
            if event.matches(QtGui.QKeySequence.Paste):
                self.paste_from_clipboard(); return True
        return super().eventFilter(obj, event)

    def paste_from_clipboard(self):
        cb = QApplication.clipboard(); text = cb.text()
        if not text: return
        sr = self.table.currentRow() if self.table.currentRow() >= 0 else 0
        sc = self.table.currentColumn() if self.table.currentColumn() >= 0 else 0
        for r_off, line in enumerate(text.splitlines()):
            for c_off, cell in enumerate(line.split("\t")):
                r = sr + r_off; c = sc + c_off
                if r < self.table.rowCount() and c < self.table.columnCount():
                    self.table.setItem(r, c, QTableWidgetItem(cell))

def main():
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec_())

if __name__ == "__main__":
    main()