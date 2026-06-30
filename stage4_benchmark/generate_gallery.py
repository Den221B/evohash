"""Generate attacked-image montages for the diploma (body figure + appendix gallery).

Body: visual cost vs robustness (one pair, best evolved attack per hash).
Appendix: per (hash, attack) montage of 2 fixed pairs, columns X | X_t | X' | Delta x10.
"""
import os, csv, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = "outputs/stage_bench/optimized_secret/images"
OUT = "figures/diploma/gallery"
os.makedirs(OUT, exist_ok=True)
PAIRS = ["secret_p0082", "secret_p0077"]   # success desc, mean L2 asc
AMP = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
})

HASHES = ["phash", "pdq", "photodna", "neuralhash"]
HASH_NAME = {"phash": "pHash", "pdq": "PDQ", "photodna": "PhotoDNA", "neuralhash": "NeuralHash"}
ATTACK_ORDER = {
    "phash": ["atkscopes", "nes", "prokos", "simba", "zo_signsgd", "pHash_GigaEvolve", "pHash_OpenEvolve"],
    "pdq": ["atkscopes", "nes", "prokos", "simba", "zo_signsgd", "PDQ_GigaEvolve", "PDQ_OpenEvolve"],
    "photodna": ["atkscopes", "nes", "prokos", "simba", "zo_signsgd", "PhotoDNA_GigaEvolve", "PhotoDNA_OpenEvolve"],
    "neuralhash": ["atkscopes", "NeuralHash_GigaEvolve", "NeuralHash_OpenEvolve"],
}
BEST_EVOLVED = {"phash": "pHash_OpenEvolve", "pdq": "PDQ_OpenEvolve",
                "photodna": "PhotoDNA_GigaEvolve", "neuralhash": "NeuralHash_GigaEvolve"}

def pretty(a):
    m = {"nes": "NES", "simba": "SimBA", "zo_signsgd": "ZO-SignSGD", "prokos": "Prokos",
         "atkscopes": "ATKScopes"}
    if a in m:
        return m[a]
    if a.endswith("GigaEvolve"):
        return "GigaEvolve"
    if a.endswith("OpenEvolve"):
        return "OpenEvolve"
    return a

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

def delta_img(src, adv):
    d = np.clip(np.abs(adv - src) * AMP, 0, 255).astype(np.uint8)
    return d

# metrics lookup
M = {}
for r in csv.DictReader(open(f"{BASE}/../metrics.csv", encoding="utf-8")):
    M[(r["hash_id"], r["attack"], r["pair_id"])] = (
        float(r["pixel_l2_raw"]), float(r["lpips"]), r["success"].lower() in ("true", "1"))

def show(ax, img):
    ax.imshow(img.astype(np.uint8))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)

# ---------- appendix: per (hash, attack) ----------
COLS = ["Исходное $X$", "Целевое $X_t$", "Атакованное $X'$", "Разность $\\times 10$"]
for h in HASHES:
    for a in ATTACK_ORDER[h]:
        fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.0))
        for i, pid in enumerate(PAIRS):
            src = load(f"{BASE}/{h}/source/{pid}.png")
            tgt = load(f"{BASE}/{h}/target/{pid}.png")
            adv = load(f"{BASE}/{h}/adv/{a}/{pid}.png")
            dl = delta_img(src, adv)
            for j, im in enumerate([src, tgt, adv, dl]):
                show(axes[i, j], im)
            l2, lp, ok = M.get((h, a, pid), (float("nan"), float("nan"), False))
            axes[i, 0].set_ylabel(
                f"Пара {i+1}\n$L_2$={l2:.1f}\nLPIPS={lp:.3f}".replace(".", ","),
                fontsize=8, rotation=0, ha="right", va="center", labelpad=22)
        for j, c in enumerate(COLS):
            axes[0, j].set_title(c, fontsize=9)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.05)
        fig.savefig(f"{OUT}/{h}_{a}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("appendix:", h, a)

# ---------- body: visual cost vs robustness ----------
pid = PAIRS[0]
fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.85))
src = load(f"{BASE}/phash/source/{pid}.png")
show(axes[0], src); axes[0].set_title("Исходное $X$", fontsize=9)
for k, h in enumerate(HASHES):
    a = BEST_EVOLVED[h]
    adv = load(f"{BASE}/{h}/adv/{a}/{pid}.png")
    show(axes[k + 1], adv)
    l2, lp, ok = M.get((h, a, pid))
    axes[k + 1].set_title(f"{HASH_NAME[h]}\n$L_2$={l2:.0f}, LPIPS={lp:.2f}".replace(".", ","), fontsize=8.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.06)
fig.savefig(f"{OUT}/body_cost.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("body figure done; pair", pid)
