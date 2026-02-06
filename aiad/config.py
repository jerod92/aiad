import torch

TOOLS = [
    "None", "Line", "Arc", "Circle", "Rectangle",
    "Spline", "Select", "Rotate", "Extend",
]
TOOL_MAP = {name: i for i, name in enumerate(TOOLS)}
NUM_TOOLS = len(TOOLS)

IMG_SIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
