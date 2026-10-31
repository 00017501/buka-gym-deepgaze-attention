"""
Buka Gym CW1 - predicted visual attention with DeepGaze IIE + AOI statistics.

Model: DeepGaze IIE (Linardos, Kummerer, Press & Bethge, 2021, ICCV),
       https://github.com/matthias-k/DeepGaze  (pip package: deepgaze_pytorch)
The model outputs a fixation probability density: for every pixel, the
probability that a free-viewing observer's fixation lands there. The density
sums to 1 over the image, so "the share of attention in an AOI" is simply the
sum of the density inside that AOI.

IMPORTANT CAVEAT (for the report): these are PREDICTED attention maps from a
model trained on free-viewing eye-tracking data (MIT1003 etc.). They are not
real eye-tracking of Buka Gym's audience, and they ignore goals, scrolling
speed, caption text and the Instagram UI.

Usage
-----
  # 1) run the model + stats on every post in aois.json
  python buka_attention.py --images ./images --aois aois.json --out ./output

  # 2) only some posts (e.g. a redesign added to aois.json later)
  python buka_attention.py --images ./images --aois aois.json --out ./output --only post2_taekwondo_kids post2_v1

  # 3) recompute stats/overlays after editing AOI boxes, without re-running the model
  python buka_attention.py --images ./images --aois aois.json --out ./output --skip-model

  # 4) before/after table for redesigns (pairs of keys from aois.json)
  python buka_attention.py ... --skip-model --compare post2_taekwondo_kids:post2_v1

Outputs (in --out)
  density/<key>.npy          fixation density (H x W, sums to 1)
  heatmaps/<key>_heatmap.png heatmap over the post
  heatmaps/<key>_aoi.png     heatmap + AOI boxes with % attention labels
  stats_aoi.csv              one row per post x AOI
  stats_summary.csv          one row per post (entropy, clarity, logo/CTA/offer share, first-look order)
  stats.md                   the same tables in Markdown
  compare.csv / compare.md   only with --compare

Metrics
  share_pct        % of predicted attention inside the AOI
  area_pct         % of the image area the AOI covers
  lift             share_pct / area_pct  (>1 = AOI pulls more attention than its size)
  peak             highest density value inside the AOI (x 1e6, for readability)
  first_look_rank  AOIs ranked by peak density (1 = most likely to be fixated first)
  entropy_norm     Shannon entropy of the density on a 10x10 px grid / log(cells); 0..1, lower = more focused
  clarity          1 - entropy_norm (higher = attention concentrated on fewer places)
  top10_pct        % of attention falling in the 10% most-attended pixels (another clarity measure)
  logo_share_pct   share of all AOIs of type "logo"
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID = 10  # px block size for the entropy calculation


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_centerbias(path, h, w):
    """Resize the MIT1003 centre-bias template to the image size and normalise
    it to a log-density (same as the official DeepGaze example, without scipy)."""
    if path and os.path.exists(path):
        cb = np.load(path).astype(np.float32)
        cb = np.array(Image.fromarray(cb, mode="F").resize((w, h), Image.NEAREST))
    else:
        print("  ! centre-bias file not found, using a uniform prior", file=sys.stderr)
        cb = np.zeros((h, w), np.float32)
    m = cb.max()
    cb = cb - (m + np.log(np.exp(cb - m).sum()))  # subtract logsumexp
    return cb


def run_deepgaze(img_path, centerbias_path, device):
    import torch
    import deepgaze_pytorch

    global _MODEL
    if "_MODEL" not in globals():
        print("  loading DeepGaze IIE (pretrained)...")
        _MODEL = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device).eval()
    image = np.array(Image.open(img_path).convert("RGB"))
    h, w = image.shape[:2]
    cb = load_centerbias(centerbias_path, h, w)
    with torch.no_grad():
        x = torch.tensor(image.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
        c = torch.tensor(cb[None]).to(device)
        log_density = _MODEL(x, c)[0, 0].cpu().numpy().astype(np.float64)
    d = np.exp(log_density)
    return d / d.sum()


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def aoi_mask(boxes, h, w):
    m = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, int(y0)):min(h, int(y1)), max(0, int(x0)):min(w, int(x1))] = True
    return m


def entropy_norm(d):
    h, w = d.shape
    hh, ww = h // GRID * GRID, w // GRID * GRID
    p = d[:hh, :ww].reshape(hh // GRID, GRID, ww // GRID, GRID).sum(axis=(1, 3)).ravel()
    p = p / p.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(p.size))


def top_k_share(d, frac=0.10):
    flat = np.sort(d.ravel())[::-1]
    k = max(1, int(len(flat) * frac))
    return float(flat[:k].sum() / flat.sum())


def compute_stats(key, spec, d):
    h, w = d.shape
    rows, union = [], np.zeros((h, w), bool)
    for name, a in spec["aois"].items():
        m = aoi_mask(a["boxes"], h, w)
        union |= m
        rows.append({
            "post": key, "aoi": name, "type": a["type"],
            "share_pct": 100 * d[m].sum(),
            "area_pct": 100 * m.mean(),
            "peak": 1e6 * d[m].max() if m.any() else 0.0,
        })
    for r in rows:
        r["lift"] = r["share_pct"] / r["area_pct"] if r["area_pct"] else 0.0
    for rank, r in enumerate(sorted(rows, key=lambda r: -r["peak"]), 1):
        r["first_look_rank"] = rank
    rest = ~union
    rows.append({"post": key, "aoi": "(rest of image)", "type": "other",
                 "share_pct": 100 * d[rest].sum(), "area_pct": 100 * rest.mean(),
                 "peak": 1e6 * d[rest].max() if rest.any() else 0.0,
                 "lift": (d[rest].sum() / rest.mean()) if rest.any() else 0.0,
                 "first_look_rank": ""})

    def share_of(t):
        ms = [aoi_mask(a["boxes"], h, w) for a in spec["aois"].values() if a["type"] == t]
        if not ms:
            return None  # element not present in this post
        return 100 * d[np.logical_or.reduce(ms)].sum()

    en = entropy_norm(d)
    order = [r["aoi"] for r in sorted(rows[:-1], key=lambda r: r["first_look_rank"])]
    summary = {
        "post": key, "label": spec.get("label", key),
        "entropy_norm": en, "clarity": 1 - en, "top10_pct": 100 * top_k_share(d),
        "logo_share_pct": share_of("logo"), "cta_share_pct": share_of("cta"),
        "offer_share_pct": share_of("offer"), "face_share_pct": share_of("face"),
        "headline_share_pct": share_of("headline"),
        "first_look_order": " > ".join(order),
    }
    return rows, summary


# ----------------------------------------------------------------------------
# Visuals (numpy + PIL only)
# ----------------------------------------------------------------------------
_STOPS = np.array([  # inferno-like colormap
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]], float)


def colormap(v):
    v = np.clip(v, 0, 1) * (len(_STOPS) - 1)
    i = np.minimum(v.astype(int), len(_STOPS) - 2)
    f = (v - i)[..., None]
    return (_STOPS[i] * (1 - f) + _STOPS[i + 1] * f).astype(np.uint8)


def heat_overlay(img, d, alpha=0.8):
    v = d / d.max()
    v = v ** 0.4  # gamma so mid-level attention stays visible
    rgb = colormap(v).astype(float)
    base = np.asarray(img.convert("RGB")).astype(float) * 0.6
    a = (alpha * np.clip(v * 2.0, 0, 1))[..., None]
    return Image.fromarray((base * (1 - a) + rgb * a).astype(np.uint8))


def _font(size):
    for f in ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


TYPE_COLOURS = {"logo": (0, 200, 255), "cta": (0, 255, 120), "offer": (255, 230, 0),
                "face": (255, 80, 200), "headline": (255, 255, 255), "body": (180, 180, 180),
                "visual": (160, 120, 255)}


def aoi_overlay(heat, spec, rows):
    im = heat.copy()
    dr = ImageDraw.Draw(im)
    font = _font(max(16, im.width // 40))
    by_name = {r["aoi"]: r for r in rows}
    for name, a in spec["aois"].items():
        col = TYPE_COLOURS.get(a["type"], (255, 255, 255))
        r = by_name[name]
        for b in a["boxes"]:
            dr.rectangle(b, outline=col, width=4)
        x0, y0 = a["boxes"][0][:2]
        txt = f"#{r['first_look_rank']} {name} {r['share_pct']:.1f}%"
        tw, th = dr.textbbox((0, 0), txt, font=font)[2:]
        ty = y0 - th - 8 if y0 - th - 8 > 0 else y0 + 4
        dr.rectangle([x0, ty, x0 + tw + 10, ty + th + 6], fill=(0, 0, 0))
        dr.text((x0 + 5, ty + 2), txt, fill=col, font=font)
    return im


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def fmt(v, nd=1):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})


def md_table(rows, cols, nd=None):
    nd = nd or {}
    out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r[k], nd.get(k, 1)) for _, k in cols) + " |")
    return "\n".join(out)


def compare(pairs, summaries, aoi_rows):
    S = {s["post"]: s for s in summaries}
    out = []
    for p in pairs:
        before, after = p.split(":")
        if before not in S or after not in S:
            print(f"  ! compare: {p} not found", file=sys.stderr)
            continue
        for m in ["cta_share_pct", "offer_share_pct", "logo_share_pct", "face_share_pct",
                  "headline_share_pct", "clarity", "top10_pct"]:
            b, a = S[before][m], S[after][m]
            out.append({"before": before, "after": after, "metric": m, "before_val": b, "after_val": a,
                        "change": (a - b) if (a is not None and b is not None) else None})
        for k in (before, after):
            out.append({"before": before, "after": after, "metric": f"first_look_order ({k})",
                        "before_val": S[k]["first_look_order"] if k == before else "",
                        "after_val": S[k]["first_look_order"] if k == after else "", "change": None})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="images", help="folder with the post images")
    ap.add_argument("--aois", default="aois.json")
    ap.add_argument("--out", default="output")
    ap.add_argument("--centerbias", default="centerbias_mit1003.npy")
    ap.add_argument("--only", nargs="*", help="process only these keys")
    ap.add_argument("--skip-model", action="store_true", help="reuse saved density/*.npy")
    ap.add_argument("--compare", nargs="*", default=[], help="before:after key pairs")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    spec_all = json.load(open(args.aois, encoding="utf-8"))
    keys = [k for k in spec_all if not k.startswith("_") and (not args.only or k in args.only)]
    for sub in ("density", "heatmaps"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    device = args.device
    if not args.skip_model and device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_rows, summaries = [], []
    for key in keys:
        spec = spec_all[key]
        img_path = os.path.join(args.images, spec["image"])
        npy = os.path.join(args.out, "density", key + ".npy")
        print(f"[{key}]")
        if args.skip_model:
            d = np.load(npy)
        else:
            d = run_deepgaze(img_path, args.centerbias, device)
            np.save(npy, d)
        img = Image.open(img_path).convert("RGB")
        rows, summary = compute_stats(key, spec, d)
        heat = heat_overlay(img, d)
        heat.save(os.path.join(args.out, "heatmaps", f"{key}_heatmap.png"))
        aoi_overlay(heat, spec, rows).save(os.path.join(args.out, "heatmaps", f"{key}_aoi.png"))
        all_rows += rows
        summaries.append(summary)

    # Note: tables contain only the posts processed in this run (use --only with both
    # the original and its redesign keys when comparing).
    write_csv(os.path.join(args.out, "stats_aoi.csv"), all_rows)
    write_csv(os.path.join(args.out, "stats_summary.csv"), summaries)

    md = ["# Predicted attention (DeepGaze IIE) - Buka Gym posts", "",
          "_Predicted free-viewing attention from DeepGaze IIE (Linardos et al., 2021). "
          "Not real eye-tracking._", "", "## Summary per post", "",
          md_table(summaries, [("Post", "label"), ("Logo %", "logo_share_pct"), ("Headline %", "headline_share_pct"),
                               ("Face %", "face_share_pct"), ("Offer %", "offer_share_pct"), ("CTA %", "cta_share_pct"),
                               ("Clarity (0-1)", "clarity"), ("Top-10% px share", "top10_pct"),
                               ("First-look order (by peak)", "first_look_order")], {"clarity": 2}),
          "", "n/a = element not present in the post.", "", "## Per AOI", ""]
    for key in keys:
        md += [f"### {spec_all[key].get('label', key)}", "",
               md_table([r for r in all_rows if r["post"] == key],
                        [("AOI", "aoi"), ("Type", "type"), ("% attention", "share_pct"), ("% area", "area_pct"),
                         ("Lift", "lift"), ("First-look rank", "first_look_rank")], {"lift": 2}), ""]
    if args.compare:
        comp = compare(args.compare, summaries, all_rows)
        if comp:
            write_csv(os.path.join(args.out, "compare.csv"), comp)
            md += ["## Before / after", "", md_table(comp, [("Before", "before"), ("After", "after"), ("Metric", "metric"),
                                                            ("Before", "before_val"), ("After", "after_val"),
                                                            ("Change", "change")], {"before_val": 2, "after_val": 2, "change": 2}), ""]
    open(os.path.join(args.out, "stats.md"), "w", encoding="utf-8").write("\n".join(md))
    print("done ->", args.out)


if __name__ == "__main__":
    main()
