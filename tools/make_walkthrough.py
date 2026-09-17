"""Build the Session 12 walkthrough page from README.md and the EXECUTED notebook.

The page opens with the README's own explanation of the concepts - the part of the write-up
that argues understanding - then the findings, then every cell with the transcript it actually
printed on Colab. Nothing is typed here by hand: the prose comes from README.md (whose numbers
come from summary.json), the transcripts from the executed .ipynb, the figures from outputs/.

Usage:
    python tools/make_walkthrough.py                       # newest executed notebook
    python tools/make_walkthrough.py path/to/executed.ipynb
"""
import base64
import glob
import html
import io
import json
import os
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "session12_walkthrough.html")
BEGIN, END = "<!-- BEGIN MEASURED -->", "<!-- END MEASURED -->"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LIST = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
LABEL = {"dp": "DP", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}
MiB = 2**20

# Which figures belong under which cell number.
FIGURES = {"9": ["f7_loss_identical.png"],
           "11": ["f1_memory_per_rank.png", "f2_step_timeline.png"],
           "12": ["f3_memory_ladder.png", "f4_communication.png"],
           "14": ["f5_hot_rank.png"],
           "15": ["f6_v5_projection.png"]}

# README chapters that are not "the learning": figures and file lists live elsewhere on the page.
SKIP_CHAPTERS = ("Results", "Files")


def newest_executed():
    c = sorted(glob.glob(os.path.join(ROOT, "*_executed_v*.ipynb")))
    return c[-1] if c else sorted(glob.glob(os.path.join(ROOT, "*.ipynb")))[-1]


def highlight(src):
    """Python syntax highlighting via the stdlib tokenizer (strings with '#' stay strings)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return html.escape(src)
    KW = {"False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
          "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
          "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
          "raise", "return", "try", "while", "with", "yield"}
    lines = src.split("\n")
    spans = []
    for t in toks:
        if t.type == tokenize.COMMENT:
            spans.append((t.start, t.end, "c"))
        elif t.type == tokenize.STRING:
            spans.append((t.start, t.end, "s"))
        elif t.type == tokenize.NUMBER:
            spans.append((t.start, t.end, "n"))
        elif t.type == tokenize.NAME and t.string in KW:
            spans.append((t.start, t.end, "k"))
    for (sr, sc), (er, ec), cls in sorted(spans, reverse=True):
        if sr != er:
            lines[sr - 1] = lines[sr - 1][:sc] + f"\x00{cls}\x01" + lines[sr - 1][sc:]
            lines[er - 1] = lines[er - 1][:ec] + "\x02" + lines[er - 1][ec:]
        else:
            L = lines[sr - 1]
            lines[sr - 1] = L[:sc] + f"\x00{cls}\x01" + L[sc:ec] + "\x02" + L[ec:]
    out = html.escape("\n".join(lines))
    out = re.sub(r"\x00(\w)\x01", lambda m: f'<span class="{m.group(1)}">', out)
    return out.replace("\x02", "</span>")


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    # Links: external ones stay links; repo-relative ones would 404 on a published page, so they
    # keep their text only.
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?![*\w])", r"<em>\1</em>", s)
    return s


def strip_emoji(title):
    """README headings carry a leading emoji; the page's own type system does the signposting,
    so headings on the page drop it."""
    return re.sub(r"^[^\w\"'(`]+\s*", "", title.strip())


def _starts_block(ln):
    return ln.startswith(("```", "|", "#", ">")) or bool(LIST.match(ln))


def md_to_html(md):
    """The Markdown subset the notebook and README use: headings, tables, fences, lists with
    wrapped continuation lines, blockquotes, paragraphs, and inline code/bold/em/links."""
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip() or re.match(r"^-{3,}\s*$", ln):
            i += 1
            continue
        if ln.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append('<pre class="fence">' + html.escape("\n".join(block)) + "</pre>")
            i += 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            t = ['<div class="tw"><table><thead><tr>']
            t += [f"<th>{inline(c)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = min(len(m.group(1)) + 1, 6)
            out.append(f"<h{lvl}>{inline(strip_emoji(m.group(2)))}</h{lvl}>")
            i += 1
            continue
        if ln.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(quote)) + "</blockquote>")
            continue
        if LIST.match(ln):
            items, ordered = [], bool(re.match(r"^\s*\d+\.\s+", ln))
            while i < len(lines):
                if LIST.match(lines[i]):
                    items.append(LIST.sub("", lines[i], count=1))
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and (not para or not _starts_block(lines[i])):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline(" ".join(para)) + "</p>")
    return "\n".join(out)


def cell_output(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append("".join(o.get("text", [])))
        elif o.get("output_type") in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "text/plain" in d:
                parts.append("".join(d["text/plain"]))
        elif o.get("output_type") == "error":
            parts.append("\n".join(o.get("traceback", [])))
    return ANSI.sub("", "".join(parts)).rstrip()


def data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def readme_parts():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    head, rest = text.split(BEGIN, 1)
    measured, tail = rest.split(END, 1)
    chapters = []
    for block in re.split(r"^## ", head + "\n" + tail, flags=re.M)[1:]:
        title, _, body = block.partition("\n")
        title = strip_emoji(title)
        if title.startswith(SKIP_CHAPTERS):
            continue
        chapters.append((title, body))
    said = re.split(r"^###\s+\W*What the run said\s*$", measured, maxsplit=1, flags=re.M)[1]
    said = said.split("\n### ", 1)[0]
    return chapters, said


def facts(R):
    n = R["config"]["sim"]["world"]
    m32, sw, gpu = R["memory_n32"], R["sweep"], R["gpu_check"]
    C = R["gates"]["C_identical"]
    worst = max(C[s]["max_abs_master"] for s in C)
    same = all(C[s]["identical"] for s in C)
    dp, z3 = m32["dp"]["persistent_per_rank"][0], m32["zero3"]["persistent_per_rank"][0]
    comm = {s: sw[s][str(n)]["comm_over_P_rank0"] for s in LABEL}
    wt = R["hot_rank"]["whole-tensor"]
    fit = R["projection"]["smallest_fitting_world"]
    out = [
        ("same mathematics", "ZeRO-1/2/3 master weights vs DP after training",
         "bit-identical" if same else f"max |Δ| {worst:.1e}",
         f"{n} ranks, {R['config']['sim']['steps']} steps; loss fell {R['train']['loss_drop']:.2f} nats"),
        ("memory per rank", "persistent state, DP → ZeRO-3",
         f"{dp / MiB:.1f} → {z3 / MiB:.2f} MiB",
         f"{dp / z3:.0f}× less at N = {n}; counted bytes equal the formula on every rank"),
        ("bytes on the wire", "sent per rank per step, in P = 2Ψ",
         f"{comm['dp']:.3f} P · {comm['zero3']:.3f} P",
         f"DP = ZeRO-1 = ZeRO-2 exactly; ZeRO-3 {comm['zero3'] / comm['dp']:.2f}×"),
    ]
    if gpu.get("stages"):
        rs = [g["persistent_ratio"] for g in gpu["stages"].values()]
        out.append(("real GPU allocator", f"measured ÷ counted persistent, {gpu['device']}",
                    f"{min(rs):.3f}–{max(rs):.3f}×", "torch.cuda.memory_allocated() against the ledger, all four arrangements"))
    else:
        out.append(("real GPU allocator", "measured ÷ counted persistent", "not run",
                    "no CUDA in this run"))
    out += [
        ("keeping tensors whole", "ZeRO-3 hottest rank ÷ mean",
         f"{wt['max_over_mean']:.1f}×",
         " and ".join(f"rank {k} holds {'+'.join(v)}" for k, v in wt["hot_units"].items())
         + f"; {wt['empty_ranks']} of {n} ranks hold nothing"),
        ("30B on 80 GB cards", "smallest world size that fits",
         " · ".join(f"{LABEL[s]} {min(v)}" for s, v in fit.items() if v),
         " · ".join(f"{LABEL[s]} never" for s, v in fit.items() if not v)
         + f" (ZeRO-1 floor {R['projection']['zero1_floor_gib']:.1f} GiB); activations excluded"),
    ]
    return out


def build(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    R = json.load(open(os.path.join(ROOT, "outputs", "summary.json"), encoding="utf-8"))
    meta, acc = R["meta"], R["acceptance"]
    chapters, said = readme_parts()

    # ---- the learning: README chapters, in README order
    learn = []
    for k, (title, body) in enumerate(chapters):
        kicker = re.match(r"^(Part \d+)", title)
        heading = re.sub(r"^Part \d+\s+—\s+", "", title)
        heading = heading[0].upper() + heading[1:]
        learn.append(f"""
<section class="chapter" id="{slug(title)}">
  <div class="kicker">{html.escape(kicker.group(1).lower() if kicker else "")}</div>
  <div>
    <h2>{inline(heading)}</h2>
    <div class="prose">{md_to_html(body)}</div>
  </div>
</section>""")

    # ---- cells
    pairs, pending, seen_intro = [], None, False
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])
            if not seen_intro and not src.lstrip().startswith("## "):
                seen_intro = True
            else:
                pending = src
        elif c["cell_type"] == "code":
            pairs.append((pending, "".join(c["source"]), cell_output(c)))
            pending = None

    sections, toc = [], []
    for idx, (md, code, out) in enumerate(pairs, start=1):
        md = md or ""
        m = re.search(r"^##\s+Cell\s+([0-9a-z]+)\s+—\s+(.*)$", md, re.M)
        num = m.group(1) if m else str(idx)
        title = m.group(2) if m else f"Cell {idx}"
        body_md = re.sub(r"^##\s+Cell.*$", "", md, count=1, flags=re.M)
        g = re.match(r"^GATE\s+(\S+?):\s*(.*)$", title)
        gate = g.group(1) if g else None
        clean = g.group(2) if g else title
        clean = clean[0].upper() + clean[1:]
        anchor = f"cell-{num}"
        toc.append((anchor, num, clean, gate))

        figs = ""
        for fn in FIGURES.get(num, []):
            p = os.path.join(ROOT, "outputs", fn)
            if os.path.exists(p):
                figs += (f'<figure class="fig"><img src="{data_uri(p)}" alt="{html.escape(fn)}">'
                         f'<figcaption>{html.escape(fn)} — drawn by Cell 17 from the results dictionary.'
                         f'</figcaption></figure>')

        chips = []
        if gate:
            chips.append(f'<span class="chip gate">gate {html.escape(gate)}</span>')
        if not out.strip():
            chips.append('<span class="chip quiet">no output</span>')

        sections.append(f"""
<section class="cell" id="{anchor}">
  <div class="rail">
    <div class="cellno">{html.escape(num)}</div>
    <div class="chips">{''.join(chips)}</div>
  </div>
  <div class="main">
    <h2>{inline(clean)}</h2>
    <div class="prose">{md_to_html(body_md)}</div>
    <details class="src">
      <summary><span>source</span><span class="loc">{len(code.splitlines())} lines</span></summary>
      <pre class="code">{highlight(code)}</pre>
    </details>
    <div class="outwrap">
      <div class="outlabel">what it printed</div>
      <pre class="out">{html.escape(out) if out.strip() else '(no output)'}</pre>
    </div>
    {figs}
  </div>
</section>""")

    factcards = "".join(
        f'<div class="fact"><div class="ft">{html.escape(t)}</div><div class="fq">{html.escape(q)}</div>'
        f'<div class="fv">{html.escape(v)}</div><div class="fn">{html.escape(n)}</div></div>'
        for t, q, v, n in facts(R))
    toc_html = "".join(
        f'<a class="tocrow" href="#{a}"><span class="tocno">{html.escape(n)}</span>'
        f'<span class="toctitle">{inline(t)}</span>'
        f'<span class="toctags">{f"<b>gate {html.escape(gt)}</b>" if gt else ""}</span></a>'
        for a, n, t, gt in toc)

    gpu = R["gpu_check"]
    css = open(os.path.join(HERE, "walkthrough.css"), encoding="utf-8").read()
    page = f"""<title>The ZeRO Ledger</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{css}</style>

<header class="hero">
  <div class="wrap">
    <p class="eyebrow">ERA V5 · Session 12 · Distributed training: data parallel and ZeRO</p>
    <h1>The ZeRO<br>Ledger</h1>
    <p class="lede">{R['config']['sim']['world']} virtual GPUs in one process, four ways of storing
    the same model, and a gate that demands they train to identical bits. The page opens with the
    write-up's explanation of the concepts, then what the run measured, then every cell with the
    transcript it printed.</p>
    <dl class="runmeta">
      <div><dt>simulator</dt><dd>{html.escape(meta['sim_device'])} · {R['config']['sim']['world']} ranks</dd></div>
      <div><dt>memory check</dt><dd>{html.escape(gpu.get('device') or 'not run')}</dd></div>
      <div><dt>torch</dt><dd>{html.escape(meta['torch'])}</dd></div>
      <div><dt>model</dt><dd>{R['model']['params'] / 1e6:.2f}M params</dd></div>
      <div><dt>acceptance</dt><dd class="pass">{acc['passed']}/{acc['total']}</dd></div>
    </dl>
  </div>
</header>

<main class="wrap">
  <section class="learn">
    <h2 class="secttl">The learning — from the README</h2>
    {''.join(learn)}
  </section>

  <section class="facts">
    <h2 class="secttl">What the run measured</h2>
    <div class="factgrid six">{factcards}</div>
  </section>

  <section class="findings">
    <h2 class="secttl">What the run said — generated from summary.json</h2>
    <div class="prose">{md_to_html(said)}</div>
  </section>

  <section class="toc">
    <h2 class="secttl">Every cell</h2>
    <nav>{toc_html}</nav>
  </section>

  <div class="cells">{''.join(sections)}</div>
</main>

<footer class="foot">
  <div class="wrap">
    <p>Built from <code>README.md</code> and <code>{html.escape(os.path.basename(nb_path))}</code> by
    <code>tools/make_walkthrough.py</code>. The concept chapters are the README's own text; the findings
    and every number in them come from <code>outputs/summary.json</code> via <code>tools/make_readme.py</code>;
    the transcripts are the executed notebook's own output. Nothing on this page was typed by hand.</p>
  </div>
</footer>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    return OUT, len(pairs), len(chapters)


if __name__ == "__main__":
    nb = sys.argv[1] if len(sys.argv) > 1 else newest_executed()
    path, n, k = build(nb)
    print(f"{os.path.basename(nb)} -> {os.path.basename(path)}")
    print(f"  {k} README chapters · {n} code cells · {os.path.getsize(path) / 1e6:.2f} MB")
