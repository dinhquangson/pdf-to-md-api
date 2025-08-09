import asyncio
import os
import re
import uuid
from multiprocessing import get_context, Queue
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import APIKeyHeader
from scalar_fastapi import get_scalar_api_reference, Theme
import hashlib
import shutil
import base64
import json
import multiprocessing

# Load environment variables
load_dotenv()

# Get configuration from environment variables
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
API_KEY = os.getenv("API_KEY")
PORT = int(os.getenv("PORT", "8000"))

# Note: we import create_model_dict and other marker modules inside the worker process.
# Download marker model on main process if you want to warm it up (optional).
try:
    from marker.models import create_model_dict  # type: ignore
    # Warm-up (optional) - comment out if you want worker to handle model loading
    artifact_dict = create_model_dict()
except (ImportError, RuntimeError, OSError):
    # If import fails at startup we still proceed; worker will try to import/initialize.
    artifact_dict = None  # type: ignore

# Job tracking
# We store a dict per job: { "task": asyncio.Task, "process": multiprocessing.Process, "queue": multiprocessing.Queue }
jobs: Dict[str, Dict[str, Any]] = {}

# Map from file hash -> job_id (prevents duplicate processing of same file at the same time)
file_jobs: dict[str, str] = {}

def hash_file(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Initialize FastAPI
app = FastAPI(
    title="PDF to Markdown/JSON Converter",
    description="API to convert PDF files to various formats using datalab-to/marker (LLM optional)",
    version="1.0.0",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "PDF Conversion",
            "description": "Endpoints for converting PDF files to Markdown, JSON, HTML, or Chunks."
        },
        {
            "name": "API Info",
            "description": "Endpoint to retrieve API metadata."
        },
        {
            "name": "Job Control",
            "description": "Endpoints for cancelling running jobs."
        }
    ],
    docs_url=None,
    redoc_url=None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    return api_key

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """
    Normalize a filename by replacing invalid characters and spaces with underscores.
    Preserves the file extension.
    """
    base, ext = os.path.splitext(filename)
    safe_base = re.sub(r'[^\w\-]', '_', base.strip())
    safe_base = re.sub(r'_+', '_', safe_base)
    if not safe_base:
        safe_base = "unnamed"
    return f"{safe_base}{ext}"

# ------------------------------
# Worker function run in child process
# ------------------------------
def _worker_convert(file_path_str: str, safe_filename: str, config: dict, result_queue: Queue):
    """
    This function runs inside a separate process. It performs the Marker conversion
    synchronously and puts the resulting zip path (or an error dict) into result_queue.
    """
    try:
        # Imports inside worker to avoid pickling heavy objects
        from marker.config.parser import ConfigParser  # type: ignore
        from marker.converters.pdf import PdfConverter  # type: ignore
        from marker.models import create_model_dict as _create_model_dict  # type: ignore
        from marker.output import text_from_rendered  # type: ignore
        from pathlib import Path as _Path

        # Recreate artifact dict inside worker (this may take time but isolates process)
        artifact = _create_model_dict()

        # Build converter and run conversion
        cfg_parser = ConfigParser(config)
        converter = PdfConverter(
            config=cfg_parser.generate_config_dict(),
            artifact_dict=artifact,
            processor_list=cfg_parser.get_processors(),
            renderer=cfg_parser.get_renderer(),
            llm_service=cfg_parser.get_llm_service() if config.get("use_llm") else None
        )

        rendered = converter(str(file_path_str))
        text, metadata, images = text_from_rendered(rendered)

        zip_file_name = safe_filename.removesuffix(".pdf")
        output_dir = _Path("output") / zip_file_name
        output_dir.mkdir(parents=True, exist_ok=True)

        text_file = output_dir / f"output.{ 'md' if config['output_format'] == 'markdown' else config['output_format'] }"
        text_file.write_text(text, encoding="utf-8")

        metadata_file = output_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        for img_name, img_data in images.items():
            safe_img_name = re.sub(r'[^\w\-]', '_', img_name)
            safe_img_name = re.sub(r'_+', '_', safe_img_name)
            if not safe_img_name:
                safe_img_name = "image"
            img_path = output_dir / f"{safe_img_name}.jpeg"
            if isinstance(img_data, Image.Image):
                img_data.save(img_path, format="JPEG")
            elif isinstance(img_data, (bytes, bytearray)):
                img_path.write_bytes(img_data)
            elif isinstance(img_data, str):
                img_path.write_bytes(base64.b64decode(img_data))

        zip_path = _Path("output") / f"{zip_file_name}.zip"
        # Create archive (shutil in worker)
        shutil.make_archive(str(zip_path.with_suffix("")), 'zip', output_dir)

        # Put resulting zip path back to parent
        result_queue.put({"zip_path": str(zip_path)})
    except (OSError, RuntimeError, ValueError) as os_err:
        # Put error back to parent
        try:
            result_queue.put({"error": str(os_err)})
        except (OSError, RuntimeError):
            pass

# ------------------------------
# Async job runner (monitors worker process)
# ------------------------------
async def process_pdf_job(job_id: str, file_path: Path, safe_filename: str, config: dict, file_hash: str):
    """
    Launch worker process to perform conversion. Monitor process so cancellation requests
    terminate the child process. Return io.BytesIO of zip on success.
    """
    ctx = get_context("spawn")
    result_queue: Queue = ctx.Queue()
    p = ctx.Process(target=_worker_convert, args=(str(file_path), safe_filename, config, result_queue))
    p.start()

    try:
        while True:
            entry = jobs.get(job_id)
            if entry is None:
                if p.is_alive():
                    try:
                        p.terminate()
                    except (OSError, RuntimeError):
                        pass
                raise asyncio.CancelledError()

            if entry.get("cancellation_requested"):
                if p.is_alive():
                    p.terminate()
                raise asyncio.CancelledError()

            if not p.is_alive():
                break
            await asyncio.sleep(0.2)

        try:
            result = result_queue.get(timeout=1)
        except multiprocessing.queues.Queue:
            result = None

        if not result:
            raise HTTPException(status_code=500, detail="Worker finished but no result was reported.")

        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Worker error: {result['error']}")

        zip_path_str = result.get("zip_path")
        if not zip_path_str:
            raise HTTPException(status_code=500, detail="Worker did not return a valid zip path.")

        zip_path = Path(zip_path_str)
        if not zip_path.exists():
            raise HTTPException(status_code=500, detail="Output ZIP missing after conversion.")

        jobs[job_id]["zip_path"] = zip_path_str
        return zip_path_str

    finally:
        # Clean up mapping so same file can be processed again in future
        file_jobs.pop(file_hash, None)
        try:
            if p.is_alive():
                p.terminate()
        except (OSError, RuntimeError):
            pass
        try:
            p.join(timeout=1)
        except (OSError, RuntimeError):
            pass

# ------------------------------
# API endpoints
# ------------------------------
@app.get("/", tags=["/"], summary="Get API metadata")
async def get_api_info():
    """
    Retrieve metadata about the API, including title, description, version, and tags.
    """
    try:
        return JSONResponse(content={
            "title": app.title,
            "description": app.description,
            "version": app.version,
            "openapi_url": app.openapi_url,
            "openapi_tags": app.openapi_tags
        })
    except (ValueError, TypeError, RuntimeError) as get_api_error:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(get_api_error)}")

@app.post("/convert", tags=["Job Conversion Control"], summary="Start PDF conversion job", dependencies=[Depends(verify_api_key)])
async def convert_pdf(
        file: UploadFile = File(...),
        output_format: str = Form("markdown"),
        force_ocr: bool = Form(False),
        strip_existing_ocr: bool = Form(False),
        disable_image_extraction: bool = Form(False),
        page_range: Optional[str] = Form(None),
        langs: Optional[str] = Form(None),
        use_llm: bool = Form(False),
        block_correction_prompt: Optional[str] = Form(None),
        llm_service: Optional[str] = Form(None),
        redo_inline_math: bool = Form(False)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    safe_filename = sanitize_filename(file.filename)
    file_path = UPLOAD_DIR / safe_filename
    with file_path.open("wb") as f:
        f.write(await file.read())

    # Hash file to detect duplicates
    file_hash = hash_file(file_path)
    if file_hash in file_jobs:
        existing_job_id = file_jobs[file_hash]
        return {"job_id": existing_job_id, "status": "already_processing"}

    config = {
        "output_format": output_format,
        "force_ocr": force_ocr,
        "strip_existing_ocr": strip_existing_ocr,
        "disable_image_extraction": disable_image_extraction,
        "page_range": page_range,
        "langs": langs.split(",") if langs else None,
        "use_llm": use_llm,
        "block_correction_prompt": block_correction_prompt,
        "llm_service": llm_service or "marker.services.gemini.GoogleGeminiService",
        "redo_inline_math": redo_inline_math
    }

    job_id = str(uuid.uuid4())
    file_jobs[file_hash] = job_id  # Track this file's hash to prevent duplicates

    task = asyncio.create_task(process_pdf_job(job_id, file_path, safe_filename, config, file_hash))
    jobs[job_id] = {"task": task, "cancellation_requested": False}

    return {"job_id": job_id, "status": "processing"}

@app.get("/result/{job_id}", tags=["Job Conversion Control"], summary="Get the converted pdf file",  dependencies=[Depends(verify_api_key)])
async def get_result(job_id: str):
    entry = jobs.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    task: asyncio.Task = entry.get("task")
    if not task or not task.done():
        return {"job_id": job_id, "status": "processing"}

    if task.cancelled():
        raise HTTPException(status_code=499, detail="Conversion cancelled")

    exc = task.exception()
    if exc:
        raise HTTPException(status_code=500, detail=str(exc))

    zip_path_str = task.result()
    zip_path = Path(zip_path_str)
    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="ZIP file missing")

    response = FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{zip_path.stem}.zip"
    )

    async def cleanup():
        try:
            import shutil
            output_folder = Path("output") / zip_path.stem
            zip_file_path = Path("output") / f"{zip_path.stem}.zip"  # fixed path to point directly to the zip in output/

            # Remove the extracted output folder if it exists
            if output_folder.exists():
                shutil.rmtree(output_folder, ignore_errors=True)

            # Remove the generated zip file if it exists
            if zip_file_path.exists():
                zip_file_path.unlink()  # unlink because it's a file, not a folder

            # Remove the uploaded PDF if it exists
            uploaded_pdf = UPLOAD_DIR / f"{zip_path.stem}.pdf"
            if uploaded_pdf.exists():
                uploaded_pdf.unlink()

        except (OSError, RuntimeError) as e:
            print(f"[CLEANUP ERROR] {e}")

        jobs.pop(job_id, None)

    asyncio.create_task(cleanup())
    return response

@app.post("/cancel/{job_id}", tags=["Job Conversion Control"], summary="Cancel a running job", dependencies=[Depends(verify_api_key)])
async def cancel_job(job_id: str):
    entry = jobs.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found")

    # Mark cancellation requested so monitor loop can terminate the process
    entry["cancellation_requested"] = True

    # Cancel the asyncio task (so task.cancelled() becomes True)
    task: asyncio.Task = entry.get("task")
    if task and not task.done():
        task.cancel()

    return {"status": "cancellation_requested", "job_id": job_id}

@app.get("/docs", include_in_schema=False)
async def scalar_html():
    """
    Serve Scalar documentation.
    """
    try:
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=app.title,
            theme=Theme.DEEP_SPACE,
            scalar_favicon_url="🍕"
        )
    except (ValueError, TypeError, RuntimeError) as scalar_doc_error:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(scalar_doc_error)}")
