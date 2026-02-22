import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import rembg

MODNET_MODEL_PATH = "modnet.onnx"
MAX_IMAGE_SIZE = 1200

# Initialize global ONNX session
_modnet_session = None

def get_modnet_session():
    global _modnet_session
    if _modnet_session is None:
        try:
            # Optimize for CPU
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _modnet_session = ort.InferenceSession(MODNET_MODEL_PATH, options, providers=['CPUExecutionProvider'])
            print("Successfully loaded MODNet model.")
        except Exception as e:
            print(f"Error loading MODNet: {e}")
            _modnet_session = False  # Mark as failed so we don't keep retrying
    return _modnet_session

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """Resizes the image so its longest edge is at most `max_size`."""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def run_modnet(image: np.ndarray) -> np.ndarray:
    """
    Runs MODNet on an image (RGB numpy array).
    Returns the alpha matte (2D numpy array, values 0-255).
    """
    session = get_modnet_session()
    if not session:
        return None

    # Pre-process
    im_h, im_w, _ = image.shape
    ref_size = 512

    # Resize to MODNet's expected input size
    im_resize = cv2.resize(image, (ref_size, ref_size), interpolation=cv2.INTER_AREA)

    # Normalize image to [-1, 1]
    im = (im_resize.astype(np.float32) - 127.5) / 127.5

    # Convert to NCHW format (Batch, Channel, Height, Width)
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 3, 1, 2))

    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: im})

    # Output matte
    matte_tensor = outputs[0]

    # Post-process matte
    matte = matte_tensor[0, 0] * 255.0
    matte = np.clip(matte, 0, 255).astype(np.uint8)

    # Resize matte back to original dimensions
    matte_full = cv2.resize(matte, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

    return matte_full

def process_image(image: Image.Image) -> Image.Image:
    """
    Main processing pipeline.
    1. Resizes to <= 1200px
    2. Runs MODNet
    3. If MODNet fails or confidence is too low, falls back to rembg
    4. Returns transparent PNG as PIL RGBA Image
    """
    # 1. Resize
    image = resize_image(image, MAX_IMAGE_SIZE)

    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)

    # 2. Try MODNet
    alpha_matte = run_modnet(img_array)

    # Confidence Heuristic:
    # If the matte is entirely black (or almost), it might not have found a portrait
    use_fallback = False

    if alpha_matte is not None:
        # Simple check: if less than 5% of pixels are foreground, it might be a failure
        fg_ratio = np.count_nonzero(alpha_matte > 128) / alpha_matte.size
        print(f"MODNet foreground ratio: {fg_ratio:.4f}")
        if fg_ratio < 0.05:
            use_fallback = True
    else:
        use_fallback = True

    if not use_fallback:
        print("Using MODNet extraction.")
        # Combine the original RGB image with the alpha matte to get RGBA
        result_rgba = np.dstack((img_array, alpha_matte))
        return Image.fromarray(result_rgba, 'RGBA')

    # 3. Fallback to rembg
    print("Falling back to rembg (u2net).")
    try:
        session = rembg.new_session("u2net")
        result = rembg.remove(image, session=session)
        # rembg.remove() returns a PIL Image in RGBA mode
        if isinstance(result, Image.Image):
            return result
        # If it somehow returns bytes, convert back to Image
        return Image.open(io.BytesIO(result))
    except Exception as e:
        print(f"Fallback failed: {e}")
        # Return original with full-opaque alpha as a last resort
        return image.convert("RGBA")
