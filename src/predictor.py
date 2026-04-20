"""Singleton predictor for SAM segmentation."""

import logging

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from .model import load_model

logger = logging.getLogger(__name__)

MASK_COLOR = np.array([30, 100, 255, 100], dtype=np.uint8)
"""Blue semi-transparent overlay color (RGBA)."""


class Predictor:
    """Singleton wrapper around the SAM model for interactive segmentation.

    Usage::

        Predictor.initialize(device="cuda")
        result = Predictor.predict(image, points=[[100, 200]], labels=[1])
    """

    _model: SamModel | None = None
    _processor: SamProcessor | None = None
    _device: str = "cpu"

    @classmethod
    def initialize(cls, device: str = "cuda") -> None:
        """Load the SAM model onto the specified device.

        This is idempotent; calling it twice with the same device is a no-op.

        Args:
            device: Target device (e.g., "cuda" or "cpu").
        """
        if cls._model is not None:
            logger.info("Predictor already initialized; skipping.")
            return

        cls._device = device
        cls._model, cls._processor = load_model(device)
        logger.info("Predictor initialized on '%s'.", device)

    @classmethod
    def predict(
        cls,
        image: Image.Image,
        points: list[list[int]],
        labels: list[int],
    ) -> Image.Image:
        """Run SAM segmentation and return a mask-overlaid image.

        Args:
            image: Input PIL image.
            points: Click coordinates as ``[[x, y], ...]``.
            labels: Per-point labels (1 = foreground, 0 = background).

        Returns:
            A copy of the input image with a semi-transparent blue mask overlay.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        if cls._model is None or cls._processor is None:
            raise RuntimeError(
                "Predictor is not initialized. Call Predictor.initialize() first."
            )

        # Default to center point if no points are provided
        if not points or not labels:
            cx = image.width // 2
            cy = image.height // 2
            points = [[cx, cy]]
            labels = [1]
            logger.info("No points provided; using image center (%d, %d).", cx, cy)

        # Prepare inputs for the model
        inputs = cls._processor(
            image,
            input_points=[points],
            input_labels=[labels],
            return_tensors="pt",
        )
        inputs = {k: v.to(cls._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = cls._model(**inputs)

        # Post-process masks
        masks = cls._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        # Pick the mask with the highest IoU score
        iou_scores = outputs.iou_scores.cpu()
        best_idx = int(iou_scores[0, 0].argmax())
        mask = masks[0][0, best_idx].numpy().astype(bool)

        # Create overlay
        result = image.convert("RGBA")
        overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
        overlay_array = np.array(overlay)
        overlay_array[mask] = MASK_COLOR
        overlay = Image.fromarray(overlay_array)

        composited = Image.alpha_composite(result, overlay)
        return composited.convert("RGB")
