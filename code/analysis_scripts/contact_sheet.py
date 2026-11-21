#!/usr/bin/env python
"""figures/contact_sheet.pdf: every main-text figure (draft/Draft_2.tex) and Supplement figure (draft/Supplement_2.tex), one per page with
its caption. Parses the figure environments, so it always follows the drafts. Usage: python scripts/contact_sheet.py (needs pdflatex)."""
import re, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

def figures(path):
    t = Path(path).read_text()
    out = []
    for m in re.finditer(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", t, re.S):
        b = m.group(0)
        g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", b)
        c = re.search(r"\\caption\{(.*)\}\s*(?:\\label\{[^}]*\})?\s*(?:\\end\{figure)", b, re.S)
        l = re.search(r"\\label\{([^}]*)\}", b)
        if g and c: out.append((Path(g.group(1)).name, c.group(1), l.group(1) if l else ""))
    return out

def clean(c):
    c = re.sub(r"\\citep?(?:\[[^\]]*\])*\{([^}]*)\}", r"[\1]", c)
    c = re.sub(r"\\citet\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\cite\{([^}]*)\}", r"[\1]", c)
    c = re.sub(r"\\citeauthor\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\ref\{([^}]*)\}", lambda m: m.group(1).replace("_", r"\_"), c)
    c = re.sub(r"\\label\{[^}]*\}", "", c)
    return c.replace("\\citealp", "")

main, sup = figures(ROOT / "draft/Draft_2.tex"), figures(ROOT / "draft/Supplement_2.tex")
L = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=1.8cm]{geometry}", r"\usepackage{graphicx,amsmath,amssymb}", r"\usepackage[T1]{fontenc}",
     r"\pagestyle{empty}", r"\setlength{\parindent}{0pt}", r"\begin{document}"]
def page(tag, f, cap, lab):
    L.append(r"\begin{center}\textbf{" + tag + r"}\quad{\small\texttt{" + f.replace("_", r"\_") + (r"\quad " + lab.replace("_", r"\_") if lab else "") + r"}}\end{center}")
    L.append(r"\vspace{2mm}\begin{center}\includegraphics[width=\textwidth,height=0.62\textheight,keepaspectratio]{" + f + r"}\end{center}")
    L.append(r"\vspace{2mm}{\small " + clean(cap) + r"}\clearpage")
for i, (f, c, l) in enumerate(main, 1): page(f"Main Fig. {i}", f, c, l)
for i, (f, c, l) in enumerate(sup, 1): page(f"Supplement Fig. S{i}", f, c, l)
L.append(r"\end{document}")
(FIG / "contact_sheet.tex").write_text("\n".join(L))
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "contact_sheet.tex"], cwd=FIG, capture_output=True, text=True)
print(r.stdout[-300:])
for ext in ("aux", "log"): (FIG / f"contact_sheet.{ext}").unlink(missing_ok=True)
print(f"{len(main)} main + {len(sup)} Supplement figures -> figures/contact_sheet.pdf")
