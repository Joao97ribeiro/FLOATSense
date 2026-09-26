# pylint: disable=line-too-long  # HTML fragments
"""Build the anonymized copy of the project page under review/.

Run from the root of the `gh-pages` branch after any change to the site:

    python scripts/build_review.py

The copy drops everything that identifies the authors (names,
affiliations, e-mails, repository links, citation, resources) and points
to the anonymized code and data. The data are embedded as scripts
(`review/static/data/*.js`): the sandboxed viewer of
anonymous.4open.science blocks `fetch()` of sibling files and times out
on large text files, so each simulation goes in its own small file with
a vowel-free encoding that the anonymizer's term replacement cannot hit.
The script fails if an identifying term survives.
"""

import base64
import json
import pathlib
import re
import shutil

SITE = pathlib.Path(__file__).resolve().parents[1]
OUT = SITE / "review"
B52 = "0123456789bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
# Id of the anonymous.4open.science copy of branch `review` (e.g.
# "FLOATSense-ABCD"); empty until the code is anonymized.
CODE_ID = "FLOATSense-FEB2"
CODE_URL = f"https://anonymous.4open.science/r/{CODE_ID}/" if CODE_ID else ""
CODE_ZIP = (f"https://anonymous.4open.science/api/repo/{CODE_ID}/zip"
            if CODE_ID else "")
DATA_URL = "https://osf.io/h54t6/?view_only=73f55c8d86214fe1b943ede5260ddf4e"
DATA_ZIP = ("https://osf.io/download/6ab737febca12c92231409b9/"
            "?view_only=73f55c8d86214fe1b943ede5260ddf4e")
FORBIDDEN = [
    "joao", "ribeiro", "tavares", "faez", "ahmed", "decodelab", "mit.edu",
    "delft", "brown univ", "university of porto", "aveiro", "2605.25717",
    "arxiv", "github.io", "github.com/joao97", "huggingface.co/datasets",
    "iclr", "neurips", "openreview"
]


def cut(html, start, end):
    """Remove html from marker `start` to the end of marker `end`."""
    i = html.index(start)
    j = html.index(end, i) + len(end)
    return html[:i] + html[j:]


def section(html, name):
    """Remove one <!-- ==== NAME ==== --> section block."""
    return cut(html, f"<!-- ============ {name} ============ -->",
               "</section>\n")


def buttons():
    """Hero buttons of the anonymized page."""
    code = (f'''        <a href="{CODE_URL}" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-code"></i></span><span>Code</span>
        </a>
''' if CODE_URL else "")
    return f'''      <div class="publication-links" style="margin-top:1.5rem;">
{code}        <a href="{DATA_URL}" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-database"></i></span><span>Data</span>
        </a>
        <a href="#start" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-download"></i></span><span>Get started</span>
        </a>
        <a href="#explorer" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-cube"></i></span><span>Dataset explorer</span>
        </a>
        <a href="#leaderboard" class="button is-rounded is-dark">
          <span class="icon"><i class="fas fa-trophy"></i></span><span>Leaderboard</span>
        </a>
      </div>
'''


def start_section():
    """Get Started section: anonymized code and data, with commands."""
    code_line = (f'<a href="{CODE_URL}">Code (anonymized)</a>'
                 if CODE_URL else "Code (anonymized link in the paper)")
    get_code = (f'curl -L -o code.zip "{CODE_ZIP}"\n'
                "unzip code.zip -d FLOATSense &amp;&amp; cd FLOATSense\n"
                if CODE_ZIP else
                "# download the code from its anonymized link and enter it\n")
    zip_link = (f' <a href="{CODE_ZIP}">Download it as a zip</a>.'
                if CODE_ZIP else "")
    code_steps = ("# 1. code\n" + get_code +
                  "conda env create -f environment.yml &amp;&amp; "
                  "conda activate floatsense\n\n")
    return f'''<!-- ============ GET STARTED ============ -->
<section class="section" id="start">
  <div class="container is-max-desktop">
    <h2 class="title is-3">Get Started</h2>
    <div class="columns">
      <div class="column is-half">
        <div class="box">
          <p class="title is-5"><i class="fas fa-code"></i> &nbsp;{code_line}</p>
          <p>Data loaders, physics baseline, the 20 learned models, the evaluation harness and the scripts of every experiment of the paper.{zip_link}</p>
        </div>
      </div>
      <div class="column is-half">
        <div class="box">
          <p class="title is-5"><i class="fas fa-database"></i> &nbsp;<a href="{DATA_URL}">Review subset (anonymized)</a></p>
          <p>22 simulations per tower in the release format, the complete tabular files, the trained checkpoints and the reference per-simulation results (355 MB). <a href="{DATA_ZIP}">Direct download</a>. The complete dataset (24.0 GB) is released on publication.</p>
        </div>
      </div>
    </div>
<pre><code>{code_steps}# 2. data: download the review subset into data/FLOATSense
python scripts/download/run.py --flagfile=scripts/download/config.cfg

# 3. run the released checkpoints on the subset (CPU, seconds) and
#    compare with the paper's per-simulation results
mkdir -p outputs/review &amp;&amp; cp -r data/FLOATSense/checkpoints/opt2/seed0 outputs/review/opt2
python scripts/train/run.py --flagfile=scripts/train/config.cfg \\
    --tower=opt2 --test_split=review/test --run_training=False \\
    --models=tcn,mamba,naive --output_dir=outputs/review/opt2
python data/FLOATSense/compare.py outputs/review/opt2 opt2</code></pre>
  </div>
</section>

'''


def alternate_backgrounds(html):
    """White / light-grey alternation of the sections after the hero."""
    count = [0]

    def repl(_match):
        light = count[0] % 2 == 1
        count[0] += 1
        return ('<section class="section has-background-light"'
                if light else '<section class="section"')

    return re.sub(r'<section class="section(?: has-background-light)?"', repl,
                  html)


def build_html(html):
    """Anonymized index.html."""
    i = html.index('      <div class="is-size-5 publication-authors">')
    j = html.index("    </div>\n  </div>\n</section>", i)
    html = html[:i] + buttons() + html[j:]
    html = cut(html,
               '    <div class="has-text-centered" style="margin:0 0 1.5rem;',
               "    </div>\n\n")
    for name in ("RESOURCES", "CITATION"):
        html = section(html, name)
    html = html.replace(
        "<!-- ============ SCOPE ============ -->",
        start_section() + "<!-- ============ SCOPE ============ -->", 1)
    html = re.sub(
        r'<a href="https://joao97ribeiro\.github\.io/FLOATBench/">FLOATBench</a>',
        "FLOATBench (cited in the paper)", html)
    html = re.sub(
        r'<a href="https://joao97ribeiro\.github\.io/FLOAT/">FLOAT</a>',
        "FLOAT (cited in the paper)", html)
    html = re.sub(r'<meta property="og:image"[^>]*>\n', "", html)
    html = re.sub(
        r'<footer class="footer">.*?</footer>',
        '<footer class="footer">\n  <div class="content has-text-centered">\n'
        '    <p>Anonymized project page for review. Page template adapted from '
        'the Academic Project Page Template.</p>\n  </div>\n</footer>',
        html,
        flags=re.S)
    html = re.sub(r"<script>/\* anon-redirect \*/.*?</script>\n",
                  "",
                  html,
                  flags=re.S)
    html = re.sub(
        r'(<script defer src="static/js/app\.js[^"]*"></script>)', lambda m:
        '<script src="static/data/data.js"></script>\n  ' + m.group(1), html)
    return alternate_backgrounds(html)


def series_scripts(data):
    """One vowel-free script per (tower, simulation) of the explorer."""
    per = len(data["channels"]) * data["n"]
    out = {}
    for tower in data["towers"]:
        raw = base64.b64decode((SITE / "static" / "data" /
                                f"series_{tower}.txt").read_text().strip())
        vals = memoryview(raw).cast("h")
        for i in range(len(data["ids"])):
            block = vals[i * per:(i + 1) * per]
            codes = (round((v + 32768) / 65535 * 2703) for v in block)
            enc = "".join(B52[c // 52] + B52[c % 52] for c in codes)
            out[f"s_{tower}_{i}.js"] = (
                "window.FS_SER = window.FS_SER || {};\n"
                f'window.FS_SER["{tower}/{i}"] = "{enc}";\n')
    return out


def check(path, text):
    """Fail if an identifying term is left."""
    low = text.lower()
    hits = [t for t in FORBIDDEN if t in low]
    if hits:
        raise SystemExit(f"{path}: identifying terms left: {hits}")


def main():
    """Write review/ from the current site."""
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("static/css", "static/js", "static/data", "static/images"):
        (OUT / sub).mkdir(parents=True)
    data = json.loads((SITE / "static" / "data" / "data.json").read_text())
    files = {
        "index.html":
            build_html((SITE / "index.html").read_text()),
        "static/js/app.js": (SITE / "static" / "js" / "app.js").read_text(),
        "static/css/style.css":
            (SITE / "static" / "css" / "style.css").read_text(),
        "static/data/data.js":
            "window.FS_DATA = " + json.dumps(data, separators=(",", ":")) +
            ";\n",
    }
    files.update({
        f"static/data/{k}": v for k, v in series_scripts(data).items()
    })
    for path, text in files.items():
        check(path, text)
        (OUT / path).write_text(text)
    for img in (SITE / "static" / "images").iterdir():
        shutil.copy(img, OUT / "static" / "images" / img.name)
    print(f"wrote {OUT} ({len(files)} files)")


if __name__ == "__main__":
    main()
