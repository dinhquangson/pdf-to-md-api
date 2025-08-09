import base64
import json
import shutil
import re
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from scalar_fastapi import get_scalar_api_reference, Theme
from PIL import Image
from dotenv import load_dotenv
import os
import io

# Load environment variables
load_dotenv()

# Get configuration from environment variables
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
API_KEY = os.getenv("API_KEY")
PORT = int(os.getenv("PORT", "8000"))  # Default to 8000 if PORT is not set

# Download marker model first
artifact_dict = create_model_dict()

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
        }
    ],
    docs_url=None,
    redoc_url=None
)

# Add CORS middleware with dynamic origins
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

# Function to normalize file names
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
    except Exception as get_api_error:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(get_api_error)}")

@app.post("/convert", tags=["PDF Conversion"], summary="Convert PDF and return ZIP", dependencies=[Depends(verify_api_key)])
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
    file_path = None
    output_dir = None
    zip_path = None

    try:
        # Validate that the uploaded file is a PDF
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        valid_formats = ["markdown", "json", "html", "chunks"]
        if output_format not in valid_formats:
            raise HTTPException(status_code=400, detail=f"Invalid output_format. Must be one of {valid_formats}")

        valid_llm_services = [
            "marker.services.gemini.GoogleGeminiService",
            "marker.services.vertex.GoogleVertexService",
            "marker.services.ollama.OllamaService",
            "marker.services.claude.ClaudeService",
            "marker.services.openai.OpenAIService",
            "marker.services.azure_openai.AzureOpenAIService"
        ]
        if use_llm and llm_service and llm_service not in valid_llm_services:
            raise HTTPException(status_code=400, detail=f"Invalid llm_service. Must be one of {valid_llm_services}")

        # Sanitize the uploaded file name
        safe_filename = sanitize_filename(file.filename)
        file_path = UPLOAD_DIR / safe_filename
        with file_path.open("wb") as f:
            f.write(await file.read())

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
        config_parser = ConfigParser(config)

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service() if use_llm else None
        )

        rendered = converter(str(file_path))
        text, metadata, images = text_from_rendered(rendered)

        # Create unique output folder using sanitized name without extension
        zip_file_name = safe_filename.removesuffix(".pdf")
        output_dir = Path("output") / zip_file_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown or chosen format
        text_file = output_dir / f"output.{ 'md' if output_format == 'markdown' else output_format }"
        text_file.write_text(text, encoding="utf-8")

        # Save metadata as JSON
        metadata_file = output_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Save images safely
        for img_name, img_data in images.items():
            safe_img_name = sanitize_filename(img_name)
            img_path = output_dir / f"{safe_img_name}.jpeg"
            if isinstance(img_data, Image.Image):
                img_data.save(img_path, format="JPG")
            elif isinstance(img_data, (bytes, bytearray)):
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
            elif isinstance(img_data, str):
                with open(img_path, "wb") as img_file:
                    img_file.write(base64.b64decode(img_data))
            else:
                raise TypeError(f"Unsupported image data type for {safe_img_name}: {type(img_data)}")

        # Create ZIP using sanitized name
        zip_path = Path("output") / f"{zip_file_name}.zip"
        shutil.make_archive(str(zip_path.with_suffix("")), 'zip', output_dir)

        # Read ZIP file into memory for StreamingResponse
        with open(zip_path, "rb") as zip_file:
            zip_content = zip_file.read()

        # Cleanup uploaded file and output immediately
        if file_path and file_path.exists():
            file_path.unlink(missing_ok=True)
        if output_dir and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        if zip_path and zip_path.exists():
            zip_path.unlink(missing_ok=True)

        # Serve the ZIP file as a StreamingResponse with the original filename
        return StreamingResponse(
            io.BytesIO(zip_content),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={file.filename.removesuffix('.pdf')}.zip"}
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (e.g., 400 for non-PDF files) without wrapping
        if file_path and file_path.exists():
            file_path.unlink(missing_ok=True)
        if output_dir and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        if zip_path and zip_path.exists():
            zip_path.unlink(missing_ok=True)
        raise http_exc
    except Exception:
        # Cleanup for unexpected errors and return generic 500 error
        if file_path and file_path.exists():
            file_path.unlink(missing_ok=True)
        if output_dir and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        if zip_path and zip_path.exists():
            zip_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/docs", include_in_schema=False)
async def scalar_html():
    """
    Serve Scalar documentation.
    """
    try:
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=app.title,
            theme=Theme.DEEP_SPACE
        )
    except Exception as scalar_doc_error:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(scalar_doc_error)}")