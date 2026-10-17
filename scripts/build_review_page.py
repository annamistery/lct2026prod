"""Page for manual answer check of a photo set: photo vs the service's top candidates -> labels.tsv.

Without correct answers the accuracy of a set (e.g. the 100 control photos in queries/) cannot be measured.
The page shows every photo next to the top-3 candidates of the running cascade; a person marks the right one,
types another catalog wine, or marks «нет в каталоге» / «не уверен». The marks are exported as labels.tsv
(image_path, expected_slug, status, alt_slugs) — the format of scripts/eval_recognition.py.

    python scripts/build_review_page.py --api http://127.0.0.1:8030 --images-dir queries --manifest queries.tsv \
        --output tmp/review/review.html
    # open the page in a browser while the API is running, mark the photos, «Скачать labels.tsv» -> put it in queries/
    python scripts/eval_recognition.py --api http://127.0.0.1:8030 --sets control=queries
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import os
import sys
import urllib.request
import uuid
from pathlib import Path

CATALOG_CSV = Path(__file__).resolve().parent.parent / "app" / "sommelier" / "data" / "wines_integrated.csv"


def search(api: str, path: Path, k: int) -> dict:
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n").encode() + path.read_bytes() + (
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="k"\r\n\r\n{k}\r\n--{boundary}--\r\n').encode()
    request = urllib.request.Request(f"{api}/api/cascade/search", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def load_catalog() -> list[dict]:
    with CATALOG_CSV.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    name_col = next(iter(rows[0]))
    return [{"slug": r["Slug"].strip(), "name": r[name_col].strip(), "winery": (r.get("Винодельня") or "").strip()}
            for r in rows if r.get("Slug", "").strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8030")
    parser.add_argument("--images-dir", type=Path, default=Path("queries"))
    parser.add_argument("--manifest", type=Path, default=Path("queries.tsv"), help="query_id<TAB>image_path; without it — all photos of the folder")
    parser.add_argument("--output", type=Path, default=Path("tmp/review/review.html"))
    parser.add_argument("-k", type=int, default=3, help="candidates shown per photo")
    args = parser.parse_args()

    if args.manifest.is_file():
        with args.manifest.open(encoding="utf-8") as fh:
            names = [row["image_path"].strip() for row in csv.DictReader(fh, delimiter="\t")]
    else:
        names = sorted(p.name for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

    items = []
    for i, name in enumerate(names, 1):
        print(f"\r{i}/{len(names)}", end="", file=sys.stderr, flush=True)
        data = search(args.api.rstrip("/"), args.images_dir / name, args.k)
        winner = (data.get("winner") or {}).get("slug")
        cands = [dict(slug=c["slug"], title=c["title"], winery=c["manufacturer"], sim=c.get("v4_similarity"),
                      img=args.api.rstrip("/") + c["image_url"] if c.get("image_url") else "")
                 for c in data["final_results"][: args.k]]
        items.append(dict(name=name, photo=Path(os.path.relpath(args.images_dir / name, args.output.parent)).as_posix(),
                          status=data["status"], winner=winner, cands=cands))
    print(file=sys.stderr)

    catalog = load_catalog()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(PAGE.replace("__DATA__", json.dumps(items, ensure_ascii=False))
                           .replace("__CATALOG__", json.dumps(catalog, ensure_ascii=False))
                           .replace("__TITLE__", html.escape(f"Сверка ответов: {args.images_dir.name} ({len(items)} фото)")),
                           encoding="utf-8")
    print(f"page: {args.output}")
    return 0


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 0; background: #f4f4f6; color: #1a1a1a; }
header { position: sticky; top: 0; z-index: 2; background: #fff; padding: 12px 16px; box-shadow: 0 2px 10px rgba(0,0,0,.08);
         display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
header h1 { font-size: 16px; margin: 0; flex: 1 1 260px; }
#progress { font-size: 13px; color: #555; }
button { border: 0; border-radius: 8px; padding: 8px 14px; font-weight: 600; cursor: pointer; background: #800020; color: #fff; }
button.ghost { background: #eee; color: #333; }
.row { display: grid; grid-template-columns: 260px 1fr; gap: 16px; background: #fff; margin: 12px 16px; padding: 12px; border-radius: 12px;
       border-left: 6px solid #ccc; }
.row.done { border-left-color: #1f9d55; }
.row.unsure { border-left-color: #d69e2e; }
.photo img { width: 100%; max-height: 340px; object-fit: contain; background: #eee; border-radius: 8px; }
.meta { font-size: 12px; color: #666; margin-top: 6px; }
.cands { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
.cand { border: 2px solid #eee; border-radius: 10px; padding: 8px; cursor: pointer; font-size: 12px; }
.cand img { width: 100%; height: 160px; object-fit: contain; background: #fafafa; }
.cand.sel { border-color: #1f9d55; background: #eefaf2; }
.cand b { display: block; font-size: 13px; margin: 4px 0 2px; }
.slug { font-family: monospace; font-size: 11px; color: #777; word-break: break-all; }
.answer { font-size: 11px; font-weight: 700; color: #800020; }
.other { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; align-items: center; font-size: 13px; }
.other input { flex: 1 1 280px; padding: 6px 8px; border: 1px solid #ccc; border-radius: 6px; }
.other label { display: flex; gap: 4px; align-items: center; cursor: pointer; }
@media (max-width: 640px) { .row { grid-template-columns: 1fr; } }
</style></head><body>
<header><h1>__TITLE__</h1><span id="progress"></span>
<button class="ghost" id="only-open">Только неотмеченные</button><button id="export">Скачать labels.tsv</button></header>
<datalist id="catalog"></datalist>
<main id="list"></main>
<script>
const DATA = __DATA__;
const CATALOG = __CATALOG__;
const KEY = 'review:' + document.title;
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { marks = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {} };
const known = new Set(CATALOG.map(c => c.slug));
document.getElementById('catalog').innerHTML = CATALOG.map(c => `<option value="${c.slug}">${c.name} — ${c.winery}</option>`).join('');
const esc = s => String(s ?? '').replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
let onlyOpen = false;

function render() {
  const list = document.getElementById('list');
  list.innerHTML = '';
  DATA.forEach(item => {
    const m = marks[item.name];
    if (onlyOpen && m) return;
    const row = document.createElement('section');
    row.className = 'row' + (m ? (m.status === 'unsure' ? ' unsure' : ' done') : '');
    row.innerHTML = `<div class="photo"><img src="${esc(item.photo)}" loading="lazy"><div class="meta">${esc(item.name)} · ответ сервиса: ${esc(item.status)}</div></div>
      <div><div class="cands">${item.cands.map((c, i) => `<div class="cand ${m && m.slug === c.slug ? 'sel' : ''}" data-slug="${esc(c.slug)}">
        ${c.slug === item.winner ? '<div class="answer">ответ сервиса</div>' : `<div class="answer" style="color:#999">кандидат ${i + 1}</div>`}
        <img src="${esc(c.img)}" loading="lazy"><b>${esc(c.title)}</b>${esc(c.winery)}<div class="slug">${esc(c.slug)}</div>сходство ${c.sim ?? '—'}</div>`).join('')}</div>
      <div class="other"><input list="catalog" placeholder="другое вино из каталога: начните вводить slug" value="${m && m.slug && !item.cands.some(c => c.slug === m.slug) ? esc(m.slug) : ''}">
        <label><input type="radio" name="s-${esc(item.name)}" value="not_in_catalog" ${m && m.status === 'not_in_catalog' ? 'checked' : ''}>нет в каталоге</label>
        <label><input type="radio" name="s-${esc(item.name)}" value="unsure" ${m && m.status === 'unsure' ? 'checked' : ''}>не уверен(а)</label>
        <button class="ghost" data-clear>сбросить</button></div></div>`;
    row.querySelectorAll('.cand').forEach(el => el.onclick = () => { marks[item.name] = { slug: el.dataset.slug, status: 'in_catalog' }; save(); render(); });
    row.querySelector('.other input[list]').onchange = e => {
      const v = e.target.value.trim();
      if (!v) return;
      if (!known.has(v)) { alert('Такого slug нет в каталоге: ' + v); return; }
      marks[item.name] = { slug: v, status: 'in_catalog' }; save(); render();
    };
    row.querySelectorAll('input[type=radio]').forEach(r => r.onchange = () => { marks[item.name] = { slug: '', status: r.value }; save(); render(); });
    row.querySelector('[data-clear]').onclick = () => { delete marks[item.name]; save(); render(); };
    list.append(row);
  });
  const done = DATA.filter(it => marks[it.name]).length;
  const agree = DATA.filter(it => marks[it.name] && marks[it.name].status === 'in_catalog' && marks[it.name].slug === it.winner).length;
  document.getElementById('progress').textContent = `отмечено ${done} из ${DATA.length}; ответ сервиса совпал с отметкой: ${agree}`;
}

document.getElementById('only-open').onclick = e => { onlyOpen = !onlyOpen; e.target.textContent = onlyOpen ? 'Показать все' : 'Только неотмеченные'; render(); };
document.getElementById('export').onclick = () => {
  const lines = ['image_path\texpected_slug\tstatus\talt_slugs'];
  DATA.forEach(it => { const m = marks[it.name]; if (m) lines.push([it.name, m.slug || '', m.status, ''].join('\t')); });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lines.join('\n') + '\n'], { type: 'text/tab-separated-values' }));
  a.download = 'labels.tsv';
  a.click();
};
render();
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
