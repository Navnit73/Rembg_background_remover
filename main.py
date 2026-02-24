import logging
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from PIL import Image
import io
import engine
import traceback
from download_model import download_modnet_model

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- SECURITY ----------------
API_TOKEN = "wbg_sk_2026_prod_xK9mN3pQ7rT1vW5y"
security = HTTPBearer()

# ---------------- APP ----------------
app = FastAPI(title="AI Image Processing Engine")

# ============================================================
# PERFORMANCE CONFIG  (SAFE FOR SMALL CLOUD VPS)
# ============================================================

CPU_COUNT = os.cpu_count() or 1

# For AI workloads, too many threads = slower
MAX_CONCURRENT = max(1, CPU_COUNT // 2)

# Semaphore protects CPU overload
_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Threadpool for blocking image processing
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

logging.info(f"CPU={CPU_COUNT}  MAX_CONCURRENT={MAX_CONCURRENT}")

# ============================================================
# MODEL WARMUP (VERY IMPORTANT FOR REMBG / AI)
# ============================================================

@app.on_event("startup")
def warmup_model():
    """
    Pre-loads the AI model into RAM so the first user request
    is not painfully slow.
    """
    try:
        logging.info("Checking and downloading MODNet model if missing...")
        download_modnet_model()
        logging.info("Warming up AI models...")
        # Note: we use an RGB dummy image instead of RGBA to prevent issues in intermediate array conversions if any
        img = Image.new("RGB", (10, 10))
        engine.process_image(img)
        logging.info("Model warmup complete")
    except Exception as e:
        logging.error(f"Warmup failed: {e}")

# ============================================================
# SYNC PROCESSOR
# ============================================================

def _process_sync(image_bytes: bytes) -> bytes:
    """
    Runs in worker thread.
    Handles full decoding + processing.
    """
    pillow_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    result_img = engine.process_image(pillow_img)

    if isinstance(result_img, Image.Image):
        output = io.BytesIO()
        result_img.save(output, format="PNG", optimize=True)
        return output.getvalue()

    return result_img

# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/api/remove-bg")
async def remove_background(
    image: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:

        # -------- TOKEN CHECK --------
        if credentials.credentials != API_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid API token")

        # -------- FILE TYPE CHECK --------
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await image.read()

        # -------- CONTROLLED CPU ACCESS --------
        async with _semaphore:

            loop = asyncio.get_running_loop()

            final_bytes = await loop.run_in_executor(
                _executor,
                _process_sync,
                image_bytes
            )

        return Response(content=final_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Processing error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Processing failed")

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok"}

# ============================================================
# LOCAL RUN (FOR DEVELOPMENT ONLY)
# IMPORTANT: USE ONE WORKER FOR AI SERVERS
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,   # NEVER increase for rembg / AI
        log_level="info",
    )