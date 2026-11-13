
from PIL import Image, ImageDraw
import numpy as np


def get_valid_strokes(drawing):
    """Collect valid X and Y coordinates from strokes."""

    valid_strokes = []
    for stroke in drawing:
        if not isinstance(stroke, (list, tuple)) or len(stroke) < 2:
            continue
        try:
            x = np.asarray(stroke[0], dtype=np.float64)
            y = np.asarray(stroke[1], dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if len(x) == 0 or len(x) != len(y):
            continue
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            continue
        valid_strokes.append((x, y))
    return valid_strokes


def render_sketch(
    drawing,
    image_size=224,
    padding=20,
    stroke_width=3,
    background_color=(255, 255, 255),
    line_color=(0, 0, 0),
    reference_drawing=None
):
    """
    Convert stroke coordinates into an RGB image.

    reference_drawing:
        Original complete drawing used as the coordinate reference.
        This keeps scale and position consistent across completion levels.
    """

    image = Image.new(
        "RGB",
        (image_size, image_size),
        background_color
    )

    draw = ImageDraw.Draw(image)
    # Collect strokes to render
    valid_strokes = get_valid_strokes(drawing)
    if not valid_strokes:
        return image
    # Use complete original drawing as coordinate reference
    if reference_drawing is not None:
        reference_strokes = get_valid_strokes(reference_drawing)
    else:
        reference_strokes = valid_strokes
    if not reference_strokes:
        return image

    # Calculate bounds from reference drawing
    all_x = np.concatenate([s[0] for s in reference_strokes])
    all_y = np.concatenate([s[1] for s in reference_strokes])
    min_x, max_x = all_x.min(), all_x.max()
    min_y, max_y = all_y.min(), all_y.max()
    drawing_width = max_x - min_x
    drawing_height = max_y - min_y
    available_size = image_size - (2 * padding)
    scale = min(
        available_size / max(drawing_width, 1e-8),
        available_size / max(drawing_height, 1e-8)
    )

    # Calculate fixed position using reference bounds
    scaled_width = drawing_width * scale
    scaled_height = drawing_height * scale
    offset_x = (image_size - scaled_width) / 2
    offset_y = (image_size - scaled_height) / 2

    # Draw partial strokes in their original order
    for x_coords, y_coords in valid_strokes:
        points = []
        for x, y in zip(x_coords, y_coords):
            px = (x - min_x) * scale + offset_x
            py = (y - min_y) * scale + offset_y
            points.append((round(px), round(py)))
        radius = max(1, stroke_width // 2)
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse(
                (x-radius, y-radius, x+radius, y+radius),
                fill=line_color
            )
        else:
            draw.line(
                points,
                fill=line_color,
                width=stroke_width,
                joint="curve"
            )
            for x, y in [points[0], points[-1]]:
                draw.ellipse(
                    (x-radius, y-radius, x+radius, y+radius),
                    fill=line_color
                )
    return image


def save_sketch(
    drawing,
    output_path,
    image_size=224,
    padding=20,
    stroke_width=3,
    reference_drawing=None
):
    """Render and save sketch as RGB PNG."""

    image = render_sketch(
        drawing=drawing,
        image_size=image_size,
        padding=padding,
        stroke_width=stroke_width,
        reference_drawing=reference_drawing
    )
    image.save(output_path, format="PNG")
    return image
