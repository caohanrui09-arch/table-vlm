"""Generate the bundled example table image from scratch.

The image is synthetic and carries no third-party data or paper content.  It
exists only to make the Gradio demo usable immediately after cloning.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'table.png'
FONT = Path(r'C:\Windows\Fonts\arial.ttf')
FONT_BOLD = Path(r'C:\Windows\Fonts\arialbd.ttf')

ROWS = [
    ['Parameter', 'Group A', 'Group B', 'p-value', 'Note'],
    ['Age (years)', '52.4 ± 8.1', '51.9 ± 7.6', '0.61', 'mean ± SD'],
    ['BMI (kg/m²)', '24.8 ± 3.2', '25.1 ± 3.5', '0.44', 'mean ± SD'],
    ['Systolic BP', '128.6 ± 12.4', '131.2 ± 13.0', '0.07', 'mmHg'],
    ['Response rate', '42/120 (35.0%)', '51/118 (43.2%)', '0.19', 'n (%)'],
]


def main() -> None:
    size, pad = 20, 14
    regular = ImageFont.truetype(str(FONT), size)
    bold = ImageFont.truetype(str(FONT_BOLD), size + 1)
    widths = [0] * len(ROWS[0])
    for row in ROWS:
        for index, text in enumerate(row):
            font = bold if row is ROWS[0] else regular
            widths[index] = max(widths[index], int(font.getlength(text)))
    widths = [max(120, width + 2 * pad) for width in widths]
    row_height = 48
    width = sum(widths) + 2
    height = row_height * len(ROWS) + 2
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    x = 1
    for index, column_width in enumerate(widths):
        y = 1
        for row_index, row in enumerate(ROWS):
            header = row_index == 0
            if header:
                draw.rectangle([x, y, x + column_width, y + row_height], fill='#DCE6F1')
            draw.rectangle([x, y, x + column_width, y + row_height], outline='#6B7A8F')
            font = bold if header else regular
            text = row[index]
            text_width = font.getlength(text)
            draw.text((x + (column_width - text_width) / 2, y + (row_height - size) / 2 - 2),
                      text, font=font, fill='#111111')
            y += row_height
        x += column_width
    image.save(OUT, format='PNG', optimize=True)
    print(f'wrote {OUT} ({image.width}x{image.height})')


if __name__ == '__main__':
    main()
