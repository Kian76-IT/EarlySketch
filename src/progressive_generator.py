
import math
COMPLETION_LEVELS = [20, 40, 60, 80, 100]

def generate_partial_sketch(drawing, percentage):
    """
    Generate a progressive sketch based on
    the original stroke sequence.

    Parameters
    ----------
    drawing : list
        Original Quick, Draw! drawing.
    percentage : int
        Completion percentage: 20, 40, 60, 80, or 100.

    Returns
    -------
    list
        Partial drawing containing the first N strokes.
    """
    if percentage not in COMPLETION_LEVELS:
        raise ValueError(
            f"Percentage must be one of {COMPLETION_LEVELS}"
        )
    if not drawing:
        return []
    total_strokes = len(drawing)

    selected_strokes = math.floor(
        total_strokes * percentage / 100
    )
    # Keep at least one stroke for non-empty drawings
    if total_strokes > 0:
        selected_strokes = max(1, selected_strokes)
    return drawing[:selected_strokes]



