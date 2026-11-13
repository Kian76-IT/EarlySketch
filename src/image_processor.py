
from pathlib import Path
from PIL import Image
from torchvision import transforms
# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_SIZE = 224
# ImageNet normalization used by the selected
# pretrained DINOv2 checkpoint.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================
# IMAGE VALIDATION
# ==========================================

def validate_image(image_path, image_size=IMAGE_SIZE):
    """
    Validate image format, color mode, and dimensions.

    Returns:
        bool: True if image is valid.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            raise ValueError(
                f"Expected RGB, got {image.mode}"
            )
        if image.size != (image_size, image_size):
            raise ValueError(
                f"Expected {(image_size, image_size)}, "
                f"got {image.size}"
            )

    return True

# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def get_dinov2_transform():
    """
    Create image transformation for DINOv2.

    Input:
        PIL RGB image, 224 x 224

    Output:
        Normalized tensor, shape (3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])
    return transform
# ==========================================
# LOAD AND TRANSFORM IMAGE
# ==========================================

def preprocess_image(image_path, transform=None):
    """
    Load an image and convert it into a tensor.

    Returns:
        torch.Tensor: Image tensor.
    """

    image_path = Path(image_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Expected 224x224, got {image.size}"
            )
        if transform is None:
            transform = get_dinov2_transform()
        image_tensor = transform(image)
    return image_tensor