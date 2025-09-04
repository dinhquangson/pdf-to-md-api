import asyncio
import os
import queue
import re
import uuid
from multiprocessing import get_context, Queue
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Security, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import APIKeyHeader
from scalar_fastapi import get_scalar_api_reference, Theme
import hashlib
import shutil
import base64
import json
import datetime

# Load environment variables
load_dotenv()

# Get configuration from environment variables
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
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
            "description": "Endpoints to retrieve API metadata and supported model information."
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
    import sys
    import psutil
    import os
    
    try:
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Worker started. Initial memory usage: {initial_memory:.1f} MB", flush=True)
        
        # Imports inside worker to avoid pickling heavy objects
        try:
            from marker.config.parser import ConfigParser  # type: ignore
            from marker.converters.pdf import PdfConverter  # type: ignore
            from marker.models import create_model_dict as _create_model_dict  # type: ignore
            from marker.output import text_from_rendered  # type: ignore
            from pathlib import Path as _Path
        except ImportError as e:
            raise RuntimeError(f"Failed to import required marker modules: {e}")

        # Check if input file exists and is readable
        file_path = _Path(file_path_str)
        if not file_path.exists():
            raise FileNotFoundError(f"Input PDF file not found: {file_path_str}")
        
        if not file_path.is_file():
            raise ValueError(f"Input path is not a file: {file_path_str}")
            
        # Check PDF file size
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"Processing PDF file: {file_size_mb:.1f} MB", flush=True)
        
        if file_size_mb > 100:  # Warn for large files
            print(f"Warning: Large PDF file ({file_size_mb:.1f} MB) may require significant memory and time", flush=True)

        # Recreate artifact dict inside worker (this may take time but isolates process)
        try:
            print("Loading marker models...", flush=True)
            artifact = _create_model_dict()
            
            # Check memory after model loading
            memory_after_models = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Models loaded. Memory usage: {memory_after_models:.1f} MB", flush=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load marker models: {e}")

        # Build converter and run conversion
        try:
            cfg_parser = ConfigParser(config)
            converter = PdfConverter(
                config=cfg_parser.generate_config_dict(),
                artifact_dict=artifact,
                processor_list=cfg_parser.get_processors(),
                renderer=cfg_parser.get_renderer(),
                llm_service=cfg_parser.get_llm_service() if config.get("use_llm") else None
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PDF converter: {e}")

        # Process the PDF
        try:
            print("Starting PDF processing...", flush=True)
            rendered = converter(str(file_path_str))
            
            # Check memory after processing
            memory_after_processing = process.memory_info().rss / 1024 / 1024  # MB
            print(f"PDF processed. Memory usage: {memory_after_processing:.1f} MB", flush=True)
            
            text, metadata, images = text_from_rendered(rendered)
            print(f"Text extraction complete. Found {len(images)} images", flush=True)
        except MemoryError:
            raise RuntimeError(f"Out of memory while processing PDF file ({file_size_mb:.1f} MB). Try processing a smaller file or increase system memory.")
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF file (may be corrupted or unsupported format): {e}")

        # Create output directory and files
        zip_file_name = safe_filename.removesuffix(".pdf")
        output_dir = _Path("output") / zip_file_name
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {e}")

        try:
            text_file = output_dir / f"output.{ 'md' if config['output_format'] == 'markdown' else config['output_format'] }"
            text_file.write_text(text, encoding="utf-8")

            metadata_file = output_dir / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to write text/metadata files: {e}")

        # Save images
        try:
            for img_name, img_data in images.items():
                # Strip extension if present before sanitizing
                base_name, _ = os.path.splitext(img_name)
                safe_img_name = re.sub(r'[^\w\-]', '_', base_name)
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
        except Exception as e:
            raise RuntimeError(f"Failed to save images: {e}")

        # Create ZIP archive
        try:
            zip_path = _Path("output") / f"{zip_file_name}.zip"
            shutil.make_archive(str(zip_path.with_suffix("")), 'zip', output_dir)
            
            if not zip_path.exists():
                raise RuntimeError("ZIP file was not created successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to create ZIP archive: {e}")

        # Put resulting zip path back to parent
        result_queue.put({"zip_path": str(zip_path)})
    except Exception as e:
        # Capture all exceptions and provide detailed error information
        import traceback
        error_details = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        try:
            result_queue.put(error_details)
        except Exception as queue_error:
            # If we can't put to queue, at least try to log
            print(f"Worker error (failed to report): {e}", flush=True)
            print(f"Queue error: {queue_error}", flush=True)

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

        # Wait for process to fully terminate and get exit code
        p.join(timeout=1)
        exit_code = p.exitcode

        try:
            result = result_queue.get(timeout=5)
        except queue.Empty:
            result = None

        if not result:
            if exit_code is not None and exit_code != 0:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Worker process failed with exit code {exit_code}. This usually indicates a crash during PDF processing. Check if the PDF file is corrupted or too large."
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Worker finished but no result was reported. The process may have been terminated unexpectedly or encountered a silent failure."
                )

        if "error" in result:
            error_detail = f"Worker error ({result.get('error_type', 'Unknown')}): {result['error']}"
            if "traceback" in result:
                # Log full traceback for debugging but provide clean error to user
                print(f"Worker traceback:\n{result['traceback']}", flush=True)
            raise HTTPException(status_code=500, detail=error_detail)

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
# API Endpoints
# ------------------------------
@app.get(
    "/",
    tags=["API Info"],
    summary="Retrieve API metadata",
    description="Returns metadata about the API, including title, description, version, and available tags.",
    responses={
        200: {
            "description": "Successful response with API metadata",
            "content": {
                "application/json": {
                    "example": {
                        "title": "PDF to Markdown/JSON Converter",
                        "description": "API to convert PDF files to various formats using datalab-to/marker (LLM optional)",
                        "version": "1.0.0",
                        "openapi_url": "/openapi.json",
                        "openapi_tags": [
                            {"name": "PDF Conversion", "description": "Endpoints for converting PDF files."},
                            {"name": "API Info", "description": "Endpoints for API metadata and model information."},
                            {"name": "Job Control", "description": "Endpoints for cancelling jobs."}
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal Server Error: [error message]"}
                }
            }
        }
    }
)
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

@app.post(
    "/convert",
    tags=["PDF Conversion"],
    summary="Start a PDF conversion job",
    description="Uploads a PDF file and initiates an asynchronous conversion job to convert it to the specified format (Markdown, JSON, HTML, or Chunks). Returns a job ID to track the conversion status.",
    dependencies=[Depends(verify_api_key)],
    responses={
        201: {
            "description": "Job successfully started",
            "content": {
                "application/json": {
                    "example": {"job_id": "123e4567-e89b-12d3-a456-426614174000", "status": "processing"}
                }
            }
        },
        400: {
            "description": "Invalid file format (non-PDF file uploaded)",
            "content": {
                "application/json": {
                    "example": {"detail": "Only PDF files are supported"}
                }
            }
        },
        401: {
            "description": "Invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or missing API key"}
                }
            }
        },
        409: {
            "description": "File is already being processed",
            "content": {
                "application/json": {
                    "example": {"job_id": "existing-job-id", "status": "already_processing"}
                }
            }
        }
    }
)
# Modify the convert_pdf endpoint to store additional job metadata
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
    # Add filename and creation time to job info
    jobs[job_id] = {
        "task": task, 
        "cancellation_requested": False,
        "filename": safe_filename,
        "created_at": datetime.datetime.now(datetime.timezone.utc)
    }

    return {"job_id": job_id, "status": "processing"}

@app.get(
    "/result/{job_id}",
    tags=["PDF Conversion"],
    summary="Retrieve conversion job result",
    description="Fetches the result of a completed PDF conversion job as a ZIP file containing the converted output, metadata, and images (if applicable).",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {
            "description": "Successful response with the converted ZIP file",
            "content": {
                "application/zip": {
                    "example": "Binary ZIP file containing output.md/json/html, metadata.json, and images"
                }
            }
        },
        202: {
            "description": "Job is still processing",
            "content": {
                "application/json": {
                    "example": {"job_id": "123e4567-e89b-12d3-a456-426614174000", "status": "processing"}
                }
            }
        },
        401: {
            "description": "Invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or missing API key"}
                }
            }
        },
        404: {
            "description": "Job not found or expired",
            "content": {
                "application/json": {
                    "example": {"detail": "Job not found or expired"}
                }
            }
        },
        499: {
            "description": "Job was cancelled",
            "content": {
                "application/json": {
                    "example": {"detail": "Conversion cancelled"}
                }
            }
        },
        500: {
            "description": "Internal server error or missing ZIP file",
            "content": {
                "application/json": {
                    "example": {"detail": "ZIP file missing"}
                }
            }
        }
    }
)
# Modify the get_result endpoint to not automatically clean up files
async def get_result(job_id: str):
    entry = jobs.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    task: asyncio.Task = entry.get("task")
    if not task or not task.done():
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "processing"
            }
        )
    if task.cancelled():
        raise HTTPException(status_code=499, detail="Conversion cancelled")

    exc = task.exception()
    if exc:
        raise HTTPException(status_code=500, detail=str(exc))

    zip_path_str = task.result()
    zip_path = Path(zip_path_str)
    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="ZIP file missing")

    # Store the zip path in the job info for future downloads
    jobs[job_id]["zip_path"] = zip_path_str

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{zip_path.stem}.zip"
    )


@app.post(
    "/cancel/{job_id}",
    tags=["Job Control"],
    summary="Cancel a running conversion job",
    description="Cancels a running PDF conversion job by its job ID.",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {
            "description": "Cancellation request accepted",
            "content": {
                "application/json": {
                    "example": {"status": "cancellation_requested", "job_id": "123e4567-e89b-12d3-a456-426614174000"}
                }
            }
        },
        401: {
            "description": "Invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or missing API key"}
                }
            }
        },
        404: {
            "description": "Job not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Job not found"}
                }
            }
        }
    }
)
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
@app.get(
    "/jobs",
    tags=["Job Control"],
    summary="List all current jobs",
    description="Returns a list of all current jobs with their status and metadata.",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {
            "description": "Successful response with job list",
            "content": {
                "application/json": {
                    "example": {
                        "jobs": [
                            {
                                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                                "status": "processing",
                                "filename": "document.pdf",
                                "created_at": "2023-10-01T12:00:00Z"
                            }
                        ]
                    }
                }
            }
        },
        401: {
            "description": "Invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or missing API key"}
                }
            }
        }
    }
)
async def list_jobs():
    """
    Retrieve a list of all current jobs with their status and metadata.
    """
    try:
        job_list = []
        for job_id, job_info in jobs.items():
            job_data = {
                "job_id": job_id,
                "status": "processing" if not job_info.get("task").done() else 
                         "completed" if job_info.get("zip_path") else 
                         "error" if job_info.get("task").exception() else 
                         "cancelled" if job_info.get("cancellation_requested") else "unknown",
                "filename": job_info.get("filename", "unknown"),
                "error": str(job_info.get("task").exception()),
                "created_at": job_info.get("created_at", "").isoformat() if job_info.get("created_at") else ""
            }
            job_list.append(job_data)
        
        return JSONResponse(content={"jobs": job_list})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job list: {str(e)}")

# Add this new endpoint after the list_jobs endpoint
@app.get(
    "/download/{job_id}",
    tags=["PDF Conversion"],
    summary="Download conversion result",
    description="Download the ZIP file containing the conversion results for a completed job.",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {
            "description": "Successful response with ZIP file",
            "content": {
                "application/zip": {
                    "example": "Binary ZIP file containing output.md/json/html, metadata.json, and images"
                }
            }
        },
        401: {
            "description": "Invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or missing API key"}
                }
            }
        },
        404: {
            "description": "Job not found or not completed",
            "content": {
                "application/json": {
                    "example": {"detail": "Job not found or not completed"}
                }
            }
        }
    }
)
async def download_result(job_id: str):
    """
    Download the ZIP file containing the conversion results for a completed job.
    """
    entry = jobs.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if not entry.get("zip_path"):
        raise HTTPException(status_code=404, detail="Job not completed yet")
    
    zip_path = Path(entry["zip_path"])
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP file not found")
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{zip_path.stem}.zip"
    )




# Add a new endpoint to clean up job files
@app.delete(
    "/job/{job_id}",
    tags=["Job Control"],
    summary="Delete a job and its files",
    description="Deletes a job and all associated files (uploaded PDF and output ZIP).",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {
            "description": "Job successfully deleted",
            "content": {
                "application/json": {
                    "example": {"status": "deleted", "job_id": "123e4567-e89b-12d3-a456-426614174000"}
                }
            }
        },
        401: {
            "description": "Invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or missing API key"}
                }
            }
        },
        404: {
            "description": "Job not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Job not found"}
                }
            }
        }
    }
)
async def delete_job(job_id: str):
    entry = jobs.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        # Clean up files
        if "zip_path" in entry:
            zip_path = Path(entry["zip_path"])
            if zip_path.exists():
                zip_path.unlink()
            
            # Also remove the extracted folder
            output_folder = Path("output") / zip_path.stem
            if output_folder.exists():
                shutil.rmtree(output_folder, ignore_errors=True)
        
        # Remove the uploaded file if it exists
        filename = entry.get("filename")
        if filename:
            uploaded_file = UPLOAD_DIR / filename
            if uploaded_file.exists():
                uploaded_file.unlink()
        
        # Remove the job from tracking
        jobs.pop(job_id, None)
        
        return {"status": "deleted", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")
@app.get(
    "/docs",
    include_in_schema=False
)
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