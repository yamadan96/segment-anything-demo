"""Gradio web application for SAM interactive segmentation."""

import logging
import os

import gradio as gr
import torch
from PIL import Image, ImageDraw

from src.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POINT_RADIUS = 6
FG_COLOR = (50, 220, 50)
BG_COLOR = (220, 50, 50)


def _draw_points(
    image: Image.Image,
    points: list[list[int]],
    labels: list[int],
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for (x, y), label in zip(points, labels):
        color = FG_COLOR if label == 1 else BG_COLOR
        r = POINT_RADIUS
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=2)
    return canvas


def on_upload(
    image: Image.Image | None,
) -> tuple[Image.Image | None, list, list, None]:
    return image, [], [], None


def on_click(
    orig: Image.Image | None,
    points: list[list[int]],
    labels: list[int],
    label_mode: str,
    evt: gr.SelectData,
) -> tuple[list, list, Image.Image | None]:
    if orig is None:
        return points, labels, None

    x, y = int(evt.index[0]), int(evt.index[1])
    label = 1 if label_mode == "foreground" else 0
    new_points = points + [[x, y]]
    new_labels = labels + [label]

    annotated = _draw_points(orig, new_points, new_labels)
    return new_points, new_labels, annotated


def on_segment(
    orig: Image.Image | None,
    points: list[list[int]],
    labels: list[int],
) -> Image.Image | None:
    if orig is None:
        return None
    return Predictor.predict(orig, points, labels)


def on_clear(
    orig: Image.Image | None,
) -> tuple[list, list, Image.Image | None, None]:
    return [], [], orig, None


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    with gr.Blocks(title="SAM - Segment Anything", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SAM - Segment Anything")
        gr.Markdown(
            "1. 画像をアップロード\n"
            "2. セグメントしたい場所をクリック（複数可）\n"
            "3. **セグメント実行** ボタンを押す\n"
            "- 🟢 **Foreground** — 含める領域\n"
            "- 🔴 **Background** — 除外する領域"
        )

        orig_state = gr.State(None)
        points_state = gr.State([])
        labels_state = gr.State([])

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="クリックして点を追加",
                    interactive=True,
                )
                with gr.Row():
                    label_radio = gr.Radio(
                        choices=["foreground", "background"],
                        value="foreground",
                        label="点の種類",
                    )
                    clear_btn = gr.Button("🗑 クリア", size="sm")
                segment_btn = gr.Button("▶ セグメント実行", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="セグメンテーション結果")

        image_input.upload(
            fn=on_upload,
            inputs=[image_input],
            outputs=[orig_state, points_state, labels_state, output_image],
        )

        image_input.select(
            fn=on_click,
            inputs=[orig_state, points_state, labels_state, label_radio],
            outputs=[points_state, labels_state, image_input],
        )

        segment_btn.click(
            fn=on_segment,
            inputs=[orig_state, points_state, labels_state],
            outputs=[output_image],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[orig_state],
            outputs=[points_state, labels_state, image_input, output_image],
        )

    return demo


# Module-level init and demo for HF Spaces
Predictor.initialize(device=DEFAULT_DEVICE)
demo = build_app()

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
