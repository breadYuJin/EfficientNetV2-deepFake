"""Model factory wrapper around timm."""

import timm


def create_model(name: str = "tf_efficientnetv2_l",
                 pretrained: bool = True,
                 num_classes: int = 1):
    """Single-logit binary classifier by default (use with BCEWithLogitsLoss)."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
