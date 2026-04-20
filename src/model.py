"""SAM model loading utilities."""

import logging

from transformers import SamModel, SamProcessor

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/sam-vit-base"


def load_model(device: str) -> tuple[SamModel, SamProcessor]:
    """Load the SAM model and processor from HuggingFace.

    Args:
        device: Target device for the model (e.g., "cuda" or "cpu").

    Returns:
        A tuple of (SamModel, SamProcessor) loaded on the specified device.

    Raises:
        RuntimeError: If the model fails to load.
    """
    logger.info("Loading SAM model '%s' on device '%s'...", MODEL_NAME, device)

    processor = SamProcessor.from_pretrained(MODEL_NAME)
    model = SamModel.from_pretrained(MODEL_NAME)
    model = model.to(device)  # pyright: ignore[reportAttributeAccessIssue]
    model.eval()  # pyright: ignore[reportAttributeAccessIssue]

    logger.info("SAM model loaded successfully.")
    return model, processor  # pyright: ignore[reportReturnType]
