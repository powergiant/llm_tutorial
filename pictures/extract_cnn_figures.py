from pathlib import Path

import pypdfium2 as pdfium


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = Path("/Users/xiaom/Downloads/LeeDL_Tutorial_v.1.2.4.pdf")
OUT_DIR = ROOT / "pictures"


def extract_figure(pdf: pdfium.PdfDocument, page_index: int, box, out_name: str) -> None:
    page = pdf[page_index]
    image = page.render(scale=2.4).to_pil().convert("RGB")

    # The crop boxes were measured on a scale=1.2 render. This render uses
    # scale=2.4, so multiply each coordinate by 2.
    scaled_box = tuple(int(v * 2) for v in box)
    image.crop(scaled_box).save(OUT_DIR / out_name)


def main() -> None:
    pdf = pdfium.PdfDocument(PDF_PATH)

    extract_figure(
        pdf,
        page_index=74,
        box=(150, 525, 575, 730),
        out_name="cnn_fig_4_5_local_pattern.png",
    )
    extract_figure(
        pdf,
        page_index=75,
        box=(170, 145, 555, 365),
        out_name="cnn_fig_4_6_receptive_field.png",
    )


if __name__ == "__main__":
    main()
