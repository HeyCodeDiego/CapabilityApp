# -*- coding: utf-8 -*-
"""
Capabilidade – Relatório (Standalone, PyQt5)
- Não carrega dados: tudo é inserido na interface.
- Você define R (amostras) e K (tamanho do subgrupo), digita o grid R×K, informa LIE/LSE e exporta o PDF.
- Arquivário automático:
    - dados_editaveis/  → JSON (completo) + CSV (formato longo, editável no Excel)
    - pdf/              → cópia do PDF
    - arquivario_index.csv → histórico de relatórios

Cálculos:
- X̄(r), R(r), X̄̄, R̄, σ_within = R̄/d2(K), LICx/LSCx, LICr/LSCr (mesmas tabelas A2/D3/D4/d2 do seu pop-up).
- Cp/Cpk com σ_within e Pp/Ppk com σ_overall (desvio-padrão de todas as observações).
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
matplotlib.use("Agg")  # sem preview; apenas geramos figuras para PDF
import matplotlib.pyplot as plt

from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# === Constantes CEP por K (iguais às do seu pop-up X̄–R) ===
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308, 25: 0.153}
D3 = {2: 0.000, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.000, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223, 25: 0.459}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777, 25: 1.541}
d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 25: 3.931}
K_ALLOWED = sorted(A2.keys())  # K suportados (2..10 e 25)

# === Pastas do Arquivário ===
ARQ_ROOT = os.path.join(os.getcwd(), "Capabilidade_Arquivario")
ARQ_DATA = os.path.join(ARQ_ROOT, "dados_editaveis")
ARQ_PDF  = os.path.join(ARQ_ROOT, "pdf")
ARQ_IDX  = os.path.join(ARQ_ROOT, "arquivario_index.csv")
os.makedirs(ARQ_DATA, exist_ok=True)
os.makedirs(ARQ_PDF, exist_ok=True)


def is_num(v):
    try:
        return v is not None and (v == v)
    except Exception:
        return False


def parse_float(txt: str):
    if txt is None:
        raise ValueError("valor vazio")
    t = str(txt).strip().replace(",", ".")
    return float(t)


def slugify(text: str, maxlen=80):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text[:maxlen] if text else "relatorio"


def compute_stats(samples_RxK, lie=None, lse=None):
    """
    Matriz R×K (linhas = amostras; colunas = subgrupo K).
    X̄(r), R(r), X̄̄, R̄, σ_within (R̄/d2(K)), limites, Cp/Cpk (σ_within) e Pp/Ppk (σ_overall).
    """
    if not samples_RxK or not isinstance(samples_RxK, list):
        raise ValueError("Matriz de amostras vazia/ inválida (esperado R×K).")
    R = len(samples_RxK)
    K = len(samples_RxK[0]) if R > 0 and isinstance(samples_RxK[0], list) else 0
    if R <= 0 or K <= 0:
        raise ValueError("Matriz de amostras inválida (linhas/colunas <= 0).")
    if K not in K_ALLOWED:
        raise ValueError(f"K={K} não suportado. Suportados: {K_ALLOWED}")

    for r, row in enumerate(samples_RxK, start=1):
        if not isinstance(row, list) or len(row) != K:
            raise ValueError(f"Linha {r} inválida: cada linha deve ter {K} valores.")
        for c, v in enumerate(row, start=1):
            if not is_num(v):
                raise ValueError(f"Valor inválido em Amostra {r}, Sub {c}.")

    xbars = [float(sum(row)) / K for row in samples_RxK]
    rvals = [float(max(row) - min(row)) for row in samples_RxK]
    xbarbar = float(sum(xbars)) / R
    rbar = float(sum(rvals)) / R

    sigma_within = (rbar / d2[K]) if d2.get(K, 0) > 0 else float("nan")

    LICx = xbarbar - (A2.get(K, 0.0) * rbar)
    LSCx = xbarbar + (A2.get(K, 0.0) * rbar)
    LICr = D3.get(K, 0.0) * rbar
    LSCr = D4.get(K, 0.0) * rbar

    if is_num(lie) and is_num(lse) and is_num(sigma_within) and sigma_within > 0:
        Cp  = (lse - lie) / (6.0 * sigma_within)
        Cpk = min((lse - xbarbar) / (3.0 * sigma_within), (xbarbar - lie) / (3.0 * sigma_within))
    else:
        Cp, Cpk = float("nan"), float("nan")

    flat = [float(v) for row in samples_RxK for v in row]
    sigma_overall = float(np.std(flat, ddof=1)) if len(flat) >= 2 else float("nan")
    if is_num(lie) and is_num(lse) and is_num(sigma_overall) and sigma_overall > 0:
        Pp  = (lse - lie) / (6.0 * sigma_overall)
        Ppk = min((lse - xbarbar) / (3.0 * sigma_overall), (xbarbar - lie) / (3.0 * sigma_overall))
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


def build_figures(stats):
    """
    Gera figuras: X̄, R, Hist+Normal (σ_within) com LIE/LSE, Cp/Cpk, Pp/Ppk.
    """
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
    ax1.plot(xs, xbars, marker="o", linestyle="-", color="#0057B7", label="X̄ (por amostra)")
    ax1.axhline(xbarbar, color="green", linestyle=":", label="Centro (X̄̄)")
    ax1.axhline(LICx, color="red", linestyle="--", label="LICx")
    ax1.axhline(LSCx, color="red", linestyle="--", label="LSCx")
    if is_num(lie): ax1.axhline(lie, color="#9E9E9E", linestyle="--", linewidth=1, label="LIE")
    if is_num(lse): ax1.axhline(lse, color="#9E9E9E", linestyle="--", linewidth=1, label="LSE")
    ax1.set_title(f"Gráfico X-barra (K={K})")
    ax1.set_xlabel("Amostras"); ax1.set_ylabel("Média")
    ax1.grid(True, linestyle="--", alpha=0.35); ax1.legend(); ax1.set_xticks(xs)
    figs.append(fig1)

    # R
    fig2, ax2 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    ax2.plot(xs, rvals, marker="o", linestyle="-", color="#FF6F00", label="R (por amostra)")
    ax2.axhline(rbar, color="green", linestyle=":", label="Centro (R̄)")
    ax2.axhline(LICr, color="red", linestyle="--", label="LICr")
    ax2.axhline(LSCr, color="red", linestyle="--", label="LSCr")
    ax2.set_title("Gráfico das Amplitudes (R)")
    ax2.set_xlabel("Amostras"); ax2.set_ylabel("Amplitude")
    ax2.grid(True, linestyle="--", alpha=0.35); ax2.legend(); ax2.set_xticks(xs)
    figs.append(fig2)

    # Histograma + Normal (σ_within) + LIE/LSE
    fig3, ax3 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    if flat:
        bins = max(10, int(len(flat) ** 0.5))
        ax3.hist(flat, bins=bins, density=True, color="#E0E0E0", edgecolor="#9E9E9E", label="Dados (hist.)")
        if is_num(sigma_within) and sigma_within > 0 and is_num(xbarbar):
            from math import sqrt, pi
            x_min = xbarbar - 4 * sigma_within; x_max = xbarbar + 4 * sigma_within
            xs_pdf = np.linspace(x_min, x_max, 400)
            pdf = (1.0 / (sigma_within * math.sqrt(2 * math.pi))) * np.exp(
                -0.5 * ((xs_pdf - xbarbar) / sigma_within) ** 2
            )
            ax3.plot(xs_pdf, pdf, color="#1565C0", linewidth=2, label="Normal ajustada (σ_within)")
            if is_num(lie):
                ax3.fill_between(xs_pdf, 0, pdf, where=(xs_pdf < lie), color="#FFCDD2", alpha=0.5, label="Fora (LIE)")
                ax3.axvline(lie, color="red", linestyle="--", linewidth=1.5, label="LIE")
            if is_num(lse):
                ax3.fill_between(xs_pdf, 0, pdf, where=(xs_pdf > lse), color="#FFE0B2", alpha=0.5, label="Fora (LSE)")
                ax3.axvline(lse, color="red", linestyle="--", linewidth=1.5, label="LSE")
    ax3.set_title("Distribuição vs Especificações (Normal com σ_within)")
    ax3.set_xlabel("Valor medido"); ax3.set_ylabel("Densidade")
    ax3.grid(True, linestyle="--", alpha=0.35); ax3.legend()
    figs.append(fig3)

    # Cp / Cpk
    fig4, ax4 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    vals = [Cp, Cpk]
    top = max(1.5, (max([v for v in vals if is_num(v)] or [1.5]) * 1.2))
    ax4.bar(["Cp", "Cpk"], vals, color=["#2E7D32", "#1565C0"])
    ax4.axhline(1.33, color="red", linestyle="--", label="Meta 1,33")
    ax4.set_ylim(0, top); ax4.set_title("Capacidade (Cp, Cpk) – σ_within")
    ax4.set_ylabel("Índice"); ax4.grid(True, axis="y", linestyle="--", alpha=0.35); ax4.legend()
    figs.append(fig4)

    # Pp / Ppk
    fig5, ax5 = plt.subplots(figsize=(10.5, 2.8), dpi=110, constrained_layout=True)
    vals_pp = [Pp, Ppk]
    top_pp = max(1.5, (max([v for v in vals_pp if is_num(v)] or [1.5]) * 1.2))
    ax5.bar(["Pp", "Ppk"], vals_pp, color=["#00897B", "#6A1B9A"])
    ax5.axhline(1.33, color="red", linestyle="--", label="Meta 1,33")
    ax5.set_ylim(0, top_pp); ax5.set_title("Performance (Pp, Ppk) – σ_overall")
    ax5.set_ylabel("Índice"); ax5.grid(True, axis="y", linestyle="--", alpha=0.35); ax5.legend()
    figs.append(fig5)

    return figs


def export_pdf(figs, out_path, header):
    """
    Salva figuras em PNG temporários e compõe um PDF A4 com cabeçalho.
    Título alterado para 'Relatório de Capabilidade'.
    """
    tmpdir = tempfile.mkdtemp(prefix="capab_report_")
    try:
        imgs = []
        for i, fig in enumerate(figs, start=1):
            img_path = os.path.join(tmpdir, f"fig_{i}.png")
            fig.savefig(img_path, dpi=200, bbox_inches="tight")
            imgs.append(img_path)

        page_w, page_h = A4
        c = pdfcanvas.Canvas(out_path, pagesize=A4)
        margin = 15 * mm
        usable_w = page_w - 2 * margin
        y = page_h - margin

        title = "Relatório de Capabilidade"
        subtitle = header

        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, title)
        y -= 10 * mm
        c.setFont("Helvetica", 10)
        c.drawString(margin, y, subtitle)
        y -= 7 * mm

        c.setFont("Helvetica", 8)
        c.drawString(margin, y, "Gráficos: X̄ | R | Histograma + Normal (σ_within) | Cp/Cpk | Pp/Ppk")
        y -= 8 * mm

        for img in imgs:
            ir = ImageReader(img)
            iw, ih = ir.getSize()
            scale = usable_w / float(iw)
            h_draw = ih * scale
            if y - h_draw < margin:
                c.showPage()
                y = page_h - margin
                c.setFont("Helvetica-Bold", 16); c.drawString(margin, y, title)
                y -= 10 * mm
                c.setFont("Helvetica", 10); c.drawString(margin, y, subtitle)
                y -= 7 * mm

            c.drawImage(ir, margin, y - h_draw, width=usable_w, height=h_draw)
            y -= (h_draw + 8 * mm)

        c.showPage()
        c.save()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def save_editable_archive(title, timestamp, stats, samples):
    """
    Salva o relatório em formato editável:
      - JSON completo (metadados + matriz + métricas)
      - CSV 'longo' (amostra, subgrupo, valor)
      - Atualiza o index CSV do arquivário
    Retorna caminhos (json_path, csv_path, basename_sugerido).
    """
    R = stats["R"]; K = stats["K"]
    lie = stats["lie"]; lse = stats["lse"]

    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{slugify(title)}_R{R}_K{K}"
    json_path = os.path.join(ARQ_DATA, base + ".json")
    csv_path  = os.path.join(ARQ_DATA, base + ".csv")

    # JSON completo
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
        "version": "capabilidade-standalone-1.0"
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CSV 'longo'
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(["amostra", "subgrupo", "valor"])
        for r in range(R):
            for c in range(K):
                w.writerow([r + 1, c + 1, samples[r][c]])

    # Atualiza índice
    need_header = not os.path.exists(ARQ_IDX)
    with open(ARQ_IDX, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=';')
        if need_header:
            w.writerow(["timestamp", "titulo", "R", "K", "LIE", "LSE", "json", "csv", "pdf"])
        # pdf ficará vazio aqui; será preenchido após salvar o PDF (retorno em on_export_pdf)
        w.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), title, R, K, lie, lse, json_path, csv_path, ""])
    return json_path, csv_path, base


def update_index_with_pdf(pdf_path, base_row_match):
    """
    Abre o índice e escreve o caminho do PDF na última linha que bate com o 'base' gerado.
    Simplesmente regrava o arquivo substituindo a célula 'pdf' vazia mais recente com esse base.
    """
    if not os.path.exists(ARQ_IDX):
        return
    rows = []
    with open(ARQ_IDX, "r", encoding="utf-8") as f:
        rows = [line.rstrip("\n") for line in f.readlines()]

    # procura de trás pra frente a primeira linha que não é header e não tem pdf preenchido
    for i in range(len(rows) - 1, -1, -1):
        if i == 0:
            # header
            break
        parts = rows[i].split(";")
        if len(parts) >= 9:
            json_path = parts[6]
            csv_path  = parts[7]
            pdf_cell  = parts[8]
            # tenta casar pela base do nome (sem extensão) contida no json ou csv
            if (base_row_match in os.path.basename(json_path)) or (base_row_match in os.path.basename(csv_path)):
                parts[8] = pdf_path
                rows[i] = ";".join(map(str, parts))
                break

    with open(ARQ_IDX, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(r + "\n")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Capabilidade – Relatório (PyQt5)")
        self.resize(1000, 720)

        # ---- Painel superior ----
        top_box = QGroupBox("Parâmetros do Estudo (R×K)")
        lay_top = QHBoxLayout(top_box)

        self.spin_R = QSpinBox(); self.spin_R.setRange(2, 999); self.spin_R.setValue(25)
        self.cmb_K = QComboBox(); self.cmb_K.addItems([str(k) for k in K_ALLOWED]); self.cmb_K.setCurrentText("5")
        self.edt_LIE = QLineEdit(); self.edt_LIE.setPlaceholderText("LIE (min)"); self.edt_LIE.setFixedWidth(120)
        self.edt_LSE = QLineEdit(); self.edt_LSE.setPlaceholderText("LSE (max)"); self.edt_LSE.setFixedWidth(120)

        self.edt_title = QLineEdit(); self.edt_title.setPlaceholderText("Identificação (Produto / Plano / Item)")
        self.btn_gen = QPushButton("Gerar grade")
        self.btn_pdf = QPushButton("Exportar PDF")

        lay_top.addWidget(QLabel("Amostras (R):"))
        lay_top.addWidget(self.spin_R)
        lay_top.addWidget(QLabel("Subgrupo (K):"))
        lay_top.addWidget(self.cmb_K)
        lay_top.addSpacing(16)
        lay_top.addWidget(QLabel("LIE:"))
        lay_top.addWidget(self.edt_LIE)
        lay_top.addWidget(QLabel("LSE:"))
        lay_top.addWidget(self.edt_LSE)
        lay_top.addSpacing(16)
        lay_top.addWidget(QLabel("Identificação:"))
        lay_top.addWidget(self.edt_title, 1)
        lay_top.addWidget(self.btn_gen)
        lay_top.addWidget(self.btn_pdf)

        # ---- Tabela R×K ----
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        self.table.installEventFilter(self)  # permite colar do Excel

        # ---- Layout principal ----
        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(top_box)
        v.addWidget(QLabel("Digite as medições no grid (aceita vírgula ou ponto)."))
        v.addWidget(self.table, 1)
        self.setCentralWidget(central)

        # ---- Sinais ----
        self.btn_gen.clicked.connect(self.generate_grid)
        self.btn_pdf.clicked.connect(self.on_export_pdf)

        # Gera grade inicial 25×5
        self.generate_grid()

    def generate_grid(self):
        R = int(self.spin_R.value())
        K = int(self.cmb_K.currentText())
        self.table.clear()
        self.table.setRowCount(R)
        self.table.setColumnCount(K)
        self.table.setHorizontalHeaderLabels([f"Sub {c+1}" for c in range(K)])
        self.table.setVerticalHeaderLabels([f"Amostra {r+1}" for r in range(R)])
        self.table.resizeColumnsToContents()

    def read_matrix(self):
        R = self.table.rowCount()
        K = self.table.columnCount()
        if K not in K_ALLOWED:
            raise ValueError(f"K={K} não suportado. Suportados: {K_ALLOWED}")
        data = []
        for r in range(R):
            row = []
            for c in range(K):
                it = self.table.item(r, c)
                txt = it.text() if it else ""
                try:
                    v = parse_float(txt)
                except Exception:
                    raise ValueError(f"Valor inválido em Amostra {r+1}, Sub {c+1}.")
                row.append(v)
            data.append(row)
        return data

    def on_export_pdf(self):
        # Parâmetros
        title = self.edt_title.text().strip() or "Estudo de Capabilidade"
        try:
            lie = parse_float(self.edt_LIE.text())
            lse = parse_float(self.edt_LSE.text())
            if lse <= lie:
                QMessageBox.warning(self, "Parâmetros", "LSE deve ser maior que LIE.")
                return
        except Exception:
            QMessageBox.warning(self, "Parâmetros", "Informe LIE e LSE numéricos.")
            return

        # Matriz R×K
        try:
            samples = self.read_matrix()
        except Exception as ex:
            QMessageBox.warning(self, "Grid", str(ex))
            return

        # Estatísticas (mesma convenção X̄–R do seu pop-up: K = nº de colunas)
        try:
            stats = compute_stats(samples, lie=lie, lse=lse)  # ← mesma lógica do seu sistema [1](https://mahle-my.sharepoint.com/personal/diego_siste_barbosa_mahle_com/Documents/Arquivos%20de%20Microsoft%20Copilot%20Chat/Trace.txt)
        except Exception as ex:
            QMessageBox.warning(self, "Cálculo", str(ex))
            return

        # 1) Arquivo editável (JSON + CSV) + índice
        timestamp = datetime.datetime.now()
        json_path, csv_path, base = save_editable_archive(title, timestamp, stats, samples)

        # 2) Geração das figuras para o PDF
        figs = build_figures(stats)

        # 3) Exportar PDF (pergunta caminho ao usuário) + arquivar cópia automática
        fn_sug = f"{base}.pdf"
        out_path, _ = QFileDialog.getSaveFileName(self, "Salvar PDF", fn_sug, "PDF (*.pdf)")
        if not out_path:
            return

        # Cabeçalho (texto do relatório atualizado para 'Capabilidade')
        hdr = (f"{title}  |  R={stats['R']}  K={stats['K']}  |  "
               f"LIE={lie:.6f}  LSE={lse:.6f}  |  "
               f"Data: {timestamp.strftime('%d/%m/%Y %H:%M')}")

        try:
            export_pdf(figs, out_path, hdr)

            # Cópia para o arquivário/pdf
            pdf_archived = os.path.join(ARQ_PDF, os.path.basename(fn_sug))
            try:
                shutil.copy2(out_path, pdf_archived)
            except Exception:
                # caso usuário exporte com nome diferente, garante um nome no arquivário
                pdf_archived = os.path.join(ARQ_PDF, os.path.basename(out_path))
                shutil.copy2(out_path, pdf_archived)

            # Atualiza o índice com caminho do PDF
            update_index_with_pdf(pdf_archived, base)

            QMessageBox.information(
                self, "Exportar",
                f"PDF salvo com sucesso:\n{out_path}\n\n"
                f"Arquivário atualizado:\n- JSON: {json_path}\n- CSV:  {csv_path}\n- PDF:  {pdf_archived}\n- Índice: {ARQ_IDX}"
            )
        except Exception as ex:
            QMessageBox.warning(self, "Exportar", f"Falha ao exportar PDF:\n{ex}")

    # Permite colar do Excel (Ctrl+V) na tabela
    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QtCore.QEvent.KeyPress:
            if event.matches(QtGui.QKeySequence.Paste):
                self.paste_from_clipboard()
                return True
        return super().eventFilter(obj, event)

    def paste_from_clipboard(self):
        cb = QApplication.clipboard()
        text = cb.text()
        if not text:
            return
        start_row = self.table.currentRow() if self.table.currentRow() >= 0 else 0
        start_col = self.table.currentColumn() if self.table.currentColumn() >= 0 else 0
        rows = text.splitlines()
        for r_offset, line in enumerate(rows):
            cols = [t for t in line.split("\t")]
            for c_offset, cell in enumerate(cols):
                r = start_row + r_offset
                c = start_col + c_offset
                if r < self.table.rowCount() and c < self.table.columnCount():
                    self.table.setItem(r, c, QTableWidgetItem(cell))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()