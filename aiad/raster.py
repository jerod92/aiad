"""Low-level rasterization utilities backed by OpenCV."""

import cv2
import numpy as np


def rasterize_line(img, p1, p2, color=1.0, thickness=1):
    """Draw a line segment on a float32 numpy image."""
    cv2.line(
        img,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        color,
        thickness,
    )
    return img


def rasterize_circle(img, center, radius, color=1.0, thickness=1):
    """Draw a circle on a float32 numpy image."""
    cv2.circle(
        img,
        (int(center[0]), int(center[1])),
        int(max(radius, 1)),
        color,
        thickness,
    )
    return img


def draw_cursor_np(img, cx, cy, size=10, color=1.0, thickness=1):
    """Draw a '+' crosshair cursor on a numpy image."""
    cv2.line(img, (int(cx), int(cy) - size), (int(cx), int(cy) + size), color, thickness)
    cv2.line(img, (int(cx) - size, int(cy)), (int(cx) + size, int(cy)), color, thickness)
    return img


def gaussian_blur(img, ksize=5):
    """Apply Gaussian blur to soften rasterized lines."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)
