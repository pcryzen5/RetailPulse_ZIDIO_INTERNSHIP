"""
Converts docs/ZIDIO_PROJECT_REPORT.md into a beautifully styled HTML report
optimized for A4 printing and PDF export.
"""
from pathlib import Path
import re


def markdown_to_html(md_text: str) -> str:
    """Basic markdown to styled HTML converter with tables and code blocks."""
    lines = md_text.splitlines()
    html_lines = []
    in_table = False
    in_code = False

    for line in lines:
        # Code blocks
        if line.startswith("```"):
            if in_code:
                html_lines.append("</pre></code>")
                in_code = False
            else:
                html_lines.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            html_lines.append(line)
            continue

        # Tables
        if "|" in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not in_table:
                html_lines.append("<table class='report-table'><thead><tr>")
                for p in parts:
                    html_lines.append(f"<th>{p}</th>")
                html_lines.append("</tr></thead><tbody>")
                in_table = True
                continue
            elif all(set(p).issubset({"-", ":", " "}) for p in parts):
                continue
            else:
                html_lines.append("<tr>")
                for p in parts:
                    html_lines.append(f"<td>{p}</td>")
                html_lines.append("</tr>")
                continue
        else:
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False

        # Headings
        if line.startswith("# "):
            html_lines.append(f"<h1 class='report-h1'>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2 class='report-h2'>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3 class='report-h3'>{line[4:]}</h3>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote class='report-quote'>{line[2:]}</blockquote>")
        elif line.strip() == "---":
            html_lines.append("<hr class='report-hr'/>")
        elif line.strip().startswith("- "):
            html_lines.append(f"<li>{line.strip()[2:]}</li>")
        elif line.strip():
            # Bold & inline code formatting
            formatted = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)
            formatted = re.sub(r"\*(.*?)\*", r"<em>\1</em>", formatted)
            formatted = re.sub(r"`(.*?)`", r"<code>\1</code>", formatted)
            html_lines.append(f"<p>{formatted}</p>")

    if in_table:
        html_lines.append("</tbody></table>")
    if in_code:
        html_lines.append("</pre></code>")

    return "\n".join(html_lines)


def generate_html_report():
    doc_path = Path("docs/ZIDIO_PROJECT_REPORT.md")
    if not doc_path.exists():
        print("docs/ZIDIO_PROJECT_REPORT.md not found.")
        return

    with open(doc_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    body_html = markdown_to_html(md_content)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RetailPulse - Zidio Project Report</title>
<style>
    @page {{
        size: A4;
        margin: 20mm;
    }}
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        line-height: 1.6;
        color: #1f2937;
        background-color: #ffffff;
        margin: 0;
        padding: 40px;
    }}
    .report-h1 {{
        color: #1e3a8a;
        font-size: 26pt;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 12px;
        margin-top: 0;
    }}
    .report-h2 {{
        color: #1e40af;
        font-size: 16pt;
        border-bottom: 1.5px solid #e5e7eb;
        padding-bottom: 6px;
        margin-top: 28px;
        page-break-after: avoid;
    }}
    .report-h3 {{
        color: #374151;
        font-size: 12pt;
        margin-top: 18px;
        page-break-after: avoid;
    }}
    .report-quote {{
        border-left: 4px solid #3b82f6;
        background: #f0fdf4;
        color: #166534;
        padding: 14px 18px;
        margin: 20px 0;
        font-style: italic;
        border-radius: 4px;
    }}
    .report-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 9.5pt;
    }}
    .report-table th {{
        background-color: #1e3a8a;
        color: white;
        text-align: left;
        padding: 8px 10px;
    }}
    .report-table td {{
        border-bottom: 1px solid #e5e7eb;
        padding: 7px 10px;
    }}
    .report-table tr:nth-child(even) {{
        background-color: #f9fafb;
    }}
    pre {{
        background-color: #f3f4f6;
        border: 1px solid #e5e7eb;
        padding: 12px;
        border-radius: 6px;
        font-size: 8.5pt;
        overflow-x: auto;
    }}
    code {{
        font-family: 'Consolas', 'Courier New', monospace;
        color: #1e3a8a;
    }}
    .report-hr {{
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 24px 0;
    }}
    li {{
        margin-bottom: 6px;
    }}
    @media print {{
        body {{
            padding: 0;
        }}
        .report-h2 {{
            page-break-before: auto;
        }}
    }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    out_file = Path("docs/ZIDIO_PROJECT_REPORT.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Generated printable report at: {out_file}")


if __name__ == "__main__":
    generate_html_report()
