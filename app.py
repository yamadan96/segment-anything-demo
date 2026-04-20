"""Gradio web application for SAM interactive segmentation."""

import logging
import os

import gradio as gr
import torch
from PIL import Image

from src.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_points(raw_text: str, label_mode: str) -> tuple[list[list[int]], list[int]]:
    """Parse a coordinate string into points and labels.

    Args:
        raw_text: Semicolon-separated "x,y" pairs (e.g., "100,200;300,400").
        label_mode: Either "foreground" or "background"; applied to all points.

    Returns:
        A tuple of (points, labels).
    """
    points: list[list[int]] = []
    labels: list[int] = []
    label_value = 1 if label_mode == "foreground" else 0

    if not raw_text or not raw_text.strip():
        return points, labels

    for pair in raw_text.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(",")
        if len(parts) != 2:
            logger.warning("Skipping malformed coordinate pair: '%s'", pair)
            continue
        x, y = int(parts[0].strip()), int(parts[1].strip())
        points.append([x, y])
        labels.append(label_value)

    return points, labels


def segment(
    image: Image.Image | None, coords_text: str, label_mode: str
) -> Image.Image | None:
    """Run segmentation on the uploaded image.

    Args:
        image: Uploaded PIL image (or None).
        coords_text: Raw coordinate string from the textbox.
        label_mode: "foreground" or "background".

    Returns:
        Mask-overlaid image, or None if no image was provided.
    """
    if image is None:
        return None

    points, labels = parse_points(coords_text, label_mode)
    return Predictor.predict(image, points, labels)


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    with gr.Blocks(title="SAM - Segment Anything") as demo:
        gr.Markdown("# SAM - Segment Anything")
        gr.Markdown(
            "Upload an image and optionally specify click coordinates "
            "to segment objects. Coordinates use the format `x1,y1;x2,y2`."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                coords_input = gr.Textbox(
                    label="Coordinates (optional)",
                    placeholder="x1,y1;x2,y2",
                )
                label_radio = gr.Radio(
                    choices=["foreground", "background"],
                    value="foreground",
                    label="Point Label",
                )
                run_button = gr.Button("Segment", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Segmentation Result")

        run_button.click(
            fn=segment,
            inputs=[image_input, coords_input, label_radio],
            outputs=output_image,
        )

    return demo


def main() -> None:
    """Entry point: initialize the predictor and launch the Gradio app."""
    logger.info("Initializing Predictor on '%s'...", DEFAULT_DEVICE)
    Predictor.initialize(device=DEFAULT_DEVICE)

    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    logger.info("Launching Gradio app on port %d...", port)

    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
