#!/usr/bin/env python3
"""
Generate a consolidated static HTML QA report from per-amplifier JSON sidecars.

Usage:
  python qa_report.py --qa-folder outputs/qa --out outputs/qa_report.html
  # Optionally specify a custom Jinja2 template
  python qa_report.py --qa-folder outputs/qa --out outputs/qa_report.html --template docs/qa_report_template.html

Inputs:
  - Folder containing QA_*.json files produced by qa_utils.save_amp_qa_page.

Output:
  - A single self-contained HTML file that lists all amplifiers, their overall
    status, and individual check statuses with links to plots.

Notes:
  - Requires Jinja2. If not installed, the script will emit a helpful error.
  - The table is searchable/filterable via a simple inline JavaScript function.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_records(qa_folder: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(qa_folder.glob('QA_*.json')):
        try:
            with open(path, 'r') as f:
                rec = json.load(f)
            # Ensure minimal fields exist
            rec.setdefault('amp', path.stem.replace('QA_', ''))
            rec.setdefault('__overall_status__', 'unknown')
            rec.setdefault('__checks__', {})
            rec.setdefault('__plots__', {})
            records.append(rec)
        except Exception:
            # Skip malformed JSON
            continue
    return records


def get_template(template_path: Optional[Path]) -> str:
    if template_path and template_path.exists():
        return template_path.read_text()
    # Inline fallback Jinja2 template
    return (
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Remedy QA Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 16px; }
    h1 { margin-top: 0; }
    .controls { margin: 10px 0 16px; }
    input[type="search"] { padding: 6px 8px; width: 360px; font-size: 14px; }
    table { border-collapse: collapse; width: 100%; font-size: 14px; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; }
    th { position: sticky; top: 0; background: #fafafa; z-index: 1; }
    .badge { padding: 2px 6px; border-radius: 10px; color: #fff; font-weight: 600; font-size: 12px; }
    .pass { background: #2e7d32; }
    .warn { background: #f9a825; }
    .fail { background: #c62828; }
    .unknown { background: #607d8b; }
    .small { font-size: 12px; color: #666; }
    a.plot { text-decoration: none; color: #1e88e5; }
  </style>
</head>
<body>
  <h1>Remedy QA Report</h1>
  <div class="controls">
    <input id="search" type="search" placeholder="Search amplifiers or filter by status: pass, warn, fail" oninput="filterTable()" />
    <span class="small">Total amplifiers: {{ amps|length }}</span>
  </div>
  <table id="qa-table">
    <thead>
      <tr>
        <th>Amp</th>
        <th>Overall</th>
        {% for ck in check_keys %}
          <th title="{{ thresholds.get(ck, {}) }}">{{ ck }}</th>
        {% endfor %}
        <th>Plots</th>
      </tr>
    </thead>
    <tbody>
      {% for r in amps %}
      <tr>
        <td>{{ r.amp }}</td>
        {% set overall = r.get('__overall_status__', 'unknown') %}
        <td><span class="badge {{ overall }}">{{ overall }}</span></td>
        {% for ck in check_keys %}
          {% set chk = r.get('__checks__', {}).get(ck, {}) %}
          {% set st = chk.get('status', 'unknown') %}
          {% set val = chk.get('value') %}
          <td>
            <span class="badge {{ st }}">{{ st }}</span>
            {% if val is not none %}
              <span class="small">({{ '%.3f'|format(val) }})</span>
            {% endif %}
          </td>
        {% endfor %}
        <td class="small">
          {% set p = r.get('__plots__', {}) %}
          {% if p.get('qa_page') %}<a class="plot" href="{{ p.get('qa_page') }}">QA</a>{% endif %}
          {% if p.get('specmask_overlay') %} · <a class="plot" href="{{ p.get('specmask_overlay') }}">Mask</a>{% endif %}
          {% if p.get('trace_overlay') %} · <a class="plot" href="{{ p.get('trace_overlay') }}">Trace</a>{% endif %}
          {% if p.get('fibernorm_diagnostic') %} · <a class="plot" href="{{ p.get('fibernorm_diagnostic') }}">FN</a>{% endif %}
          {% if p.get('fibernorm_compare') %} · <a class="plot" href="{{ p.get('fibernorm_compare') }}">FNcmp</a>{% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <script>
    function filterTable() {
      var input = document.getElementById('search');
      var filter = input.value.toLowerCase();
      var table = document.getElementById('qa-table');
      var trs = table.getElementsByTagName('tr');
      for (var i = 1; i < trs.length; i++) {
        var tds = trs[i].getElementsByTagName('td');
        var rowText = '';
        for (var j = 0; j < tds.length; j++) {
          rowText += tds[j].innerText.toLowerCase() + ' ';
        }
        if (rowText.indexOf(filter) > -1) {
          trs[i].style.display = '';
        } else {
          trs[i].style.display = 'none';
        }
      }
    }
  </script>
</body>
</html>
        """
    )


def render_html(records: List[Dict[str, Any]], template_str: str) -> str:
    try:
        from jinja2 import Environment, BaseLoader
    except Exception as e:
        raise SystemExit(
            "Jinja2 is required to build the QA report. Please install it (pip install Jinja2)."
        )
    # Collect union of known check keys from thresholds and present checks
    # Assume thresholds are the same across records; fallback to known keys
    default_keys = [
        'readnoise_e', 'bad_wavelength_frac', 'failed_traces', 'median_arc_rms', 'masked_spectral_frac',
        'fibernorm_blue_edge_median', 'fibernorm_central_median', 'fibernorm_red_edge_median'
    ]
    thresholds = {}
    for r in records:
        thr = r.get('__thresholds__', {}) or {}
        thresholds.update(thr)
    check_keys = []
    seen = set()
    for k in default_keys:
        # include if appears in any record's checks
        present = any(k in (ri.get('__checks__', {}) or {}) for ri in records)
        if present and k not in seen:
            check_keys.append(k)
            seen.add(k)
    # Sort records by amp id for stable display
    amps_sorted = sorted(records, key=lambda x: x.get('amp', ''))

    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(template_str)
    html = tmpl.render(amps=amps_sorted, check_keys=check_keys, thresholds=thresholds)
    return html


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build consolidated HTML QA report from QA_*.json files.")
    p.add_argument('--qa-folder', type=str, required=True, help='Folder containing QA_*.json files')
    p.add_argument('--out', type=str, default=None, help='Output HTML file (default: <qa-folder>/index.html)')
    p.add_argument('--template', type=str, default=None, help='Optional path to a custom Jinja2 template')
    args = p.parse_args(argv)

    qa_folder = Path(args.qa_folder)
    if not qa_folder.exists():
        raise SystemExit(f"QA folder not found: {qa_folder}")

    records = load_records(qa_folder)
    if not records:
        raise SystemExit(f"No QA_*.json files found in {qa_folder}")

    template_path = Path(args.template) if args.template else None
    template_str = get_template(template_path)
    html = render_html(records, template_str)

    out_path = Path(args.out) if args.out else (qa_folder / 'index.html')
    out_path.write_text(html)
    print(f"Wrote QA report: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
