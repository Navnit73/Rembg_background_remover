import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure basic logging
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from PIL import Image
import io
import engine
import traceback

# ---- Fixed API Token ----
API_TOKEN = "wbg_sk_2026_prod_xK9mN3pQ7rT1vW5y"
security = HTTPBearer()

app = FastAPI(title="AI Image Processing Engine")

# --- Scaling Configuration ---
# Max concurrent image processing tasks (prevents CPU overload)
MAX_CONCURRENT = 10
_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Thread pool for CPU-bound image processing
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)


def _process_sync(image_bytes: bytes) -> bytes:
    """
    Synchronous image processing function.
    Runs in a thread pool so it doesn't block the async event loop.
    """
    pillow_img = Image.open(io.BytesIO(image_bytes))
    result_img = engine.process_image(pillow_img)

    if isinstance(result_img, Image.Image):
        output = io.BytesIO()
        result_img.save(output, format="PNG")
        return output.getvalue()
    return result_img


@app.post("/api/remove-bg")
async def remove_background(
    image: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Accepts an image upload and returns a transparent PNG with the background removed.
    Offloads CPU-heavy work to a thread pool for concurrency.
    """
    try:
        # Verify token
        if credentials.credentials != API_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid API token")

        if image.content_type is not None and not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await image.read()

        # Acquire semaphore to limit concurrent processing
        async with _semaphore:
            loop = asyncio.get_event_loop()
            final_bytes = await loop.run_in_executor(_executor, _process_sync, image_bytes)

        return Response(content=final_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint for load balancers."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import multiprocessing

    # Use CPU count for workers (cap at 8 to avoid memory issues)
    workers = min(multiprocessing.cpu_count(), 8)
    print(f"Starting with {workers} workers, {MAX_CONCURRENT} threads each")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=workers,
        log_level="info",
    )
