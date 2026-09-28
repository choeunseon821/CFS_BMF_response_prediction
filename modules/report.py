from __future__ import annotations

from io import BytesIO
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors


def build_pdf_report(summary: dict, input_table, plot_png: bytes | None = None) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("CFS-BMF Design & Response Prediction Report", styles["Title"]),
        Spacer(1, 6*mm),
    ]

    data = [["Item", "Result"]] + [[str(k), str(v)] for k, v in summary.items()]
    t = Table(data, colWidths=[60*mm, 100*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [t, Spacer(1, 7*mm), Paragraph("Design variables", styles["Heading2"])]

    rows = [["Variable", "Value"]] + [[str(r[0]), f"{float(r[1]):.6g}"] for r in input_table]
    ti = Table(rows, colWidths=[70*mm, 60*mm], repeatRows=1)
    ti.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ]))
    story += [ti]

    if plot_png:
        story += [Spacer(1, 7*mm), Paragraph("Predicted response", styles["Heading2"])]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(plot_png)
            img_path = f.name
        story.append(Image(img_path, width=175*mm, height=42*mm))

    story += [Spacer(1, 6*mm), Paragraph("Note: Predictions are valid only within the model/database domain used for development.", styles["BodyText"])]
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
