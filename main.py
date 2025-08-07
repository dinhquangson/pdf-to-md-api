from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from pydantic import BaseModel
from scalar_fastapi import get_scalar_api_reference, Theme

# Download marker model first
artifact_dict = create_model_dict()

# Initialize FastAPI with correct metadata parameters and disable Swagger UI
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
    docs_url=None,  # Disable default Swagger UI
    redoc_url=None  # Disable default ReDoc UI
)

# Add CORS middleware for debugging (remove in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def custom_exception_handler(exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# Custom handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
try:
    UPLOAD_DIR.mkdir(exist_ok=True)
except Exception as e:
    raise

class ConversionResponse(BaseModel):
    content: str
    metadata: dict
    images: dict

    class Config:
        json_schema_extra = {
            "example": {
                "content": "# Sample Markdown\nThis is a converted PDF.",
                "metadata": {
                    "table_of_contents": [{"title": "Introduction", "heading_level": 1, "page_id": 0}],
                    "page_stats": [{"page_id": 0, "text_extraction_method": "pdftext", "block_counts": [["Span", 200]]}]
                },
                "images": {"image_1": "base64_encoded_image_data"}
            }
        }

class APIInfoResponse(BaseModel):
    title: str
    description: str
    version: str
    openapi_url: str
    openapi_tags: List[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "title": "PDF to Markdown/JSON Converter",
                "description": "API to convert PDF files to various formats using datalab-to/marker (LLM optional)",
                "version": "1.0.0",
                "openapi_url": "/openapi.json",
                "openapi_tags": [
                    {
                        "name": "PDF Conversion",
                        "description": "Endpoints for converting PDF files to Markdown, JSON, HTML, or Chunks."
                    },
                    {
                        "name": "API Info",
                        "description": "Endpoint to retrieve API metadata."
                    }
                ]
            }
        }

@app.get("/", tags=["/"], summary="Get API metadata", response_model=APIInfoResponse)
async def get_api_info():
    """
    Retrieve metadata about the API, including title, description, version, and tags.
    """
    try:
        return JSONResponse(content={
            "title": app.title,  # type: ignore
            "description": app.description,  # type: ignore
            "version": app.version,  # type: ignore
            "openapi_url": app.openapi_url,  # type: ignore
            "openapi_tags": app.openapi_tags  # type: ignore
        })
    except Exception as get_api_error:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(get_api_error)}")

@app.post("/convert", tags=["PDF Conversion"], summary="Convert PDF to specified format", response_model=ConversionResponse)
async def convert_pdf(
        file: UploadFile = File(..., description="PDF file to convert"),
        output_format: str = Form("markdown", description="Output format: markdown, json, html, or chunks"),
        force_ocr: bool = Form(False, description="Force OCR on all pages, including digital text"),
        strip_existing_ocr: bool = Form(False, description="Remove existing OCR text and re-OCR"),
        disable_image_extraction: bool = Form(False, description="Disable image extraction from PDF"),
        page_range: Optional[str] = Form(None, description="Pages to process, e.g., '0,5-10,20'"),
        langs: Optional[str] = Form(None, description="Comma-separated languages for OCR, e.g., 'en,fr'"),
        use_llm: bool = Form(False, description="Use LLM for improved accuracy (requires LLM service configuration)"),
        block_correction_prompt: Optional[str] = Form(None, description="Custom prompt for LLM block correction (if use_llm is true)"),
        llm_service: Optional[str] = Form(None, description="LLM service: gemini, vertex, ollama, claude, openai, azure_openai (if use_llm is true)"),
        redo_inline_math: bool = Form(False, description="Enhance inline math conversion (requires use_llm)")
):
    """
    Convert a PDF file to the specified format using datalab-to/marker.
    - **file**: The PDF file to convert.
    - **output_format**: Desired output format (markdown, json, html, chunks).
    - **force_ocr**: Force OCR on all pages, useful for bad text or inline math.
    - **strip_existing_ocr**: Remove existing OCR text and re-OCR with surya.
    - **disable_image_extraction**: Skip image extraction; with use_llm, images are described.
    - **page_range**: Specific pages to process (e.g., '0,5-10,20').
    - **langs**: Languages for OCR (e.g., 'en,fr').
    - **use_llm**: Enable LLM for improved accuracy (requires API key).
    - **block_correction_prompt**: Custom prompt for LLM output correction.
    - **llm_service**: Specify LLM service (e.g., 'gemini') if use_llm is true.
    - **redo_inline_math**: Improve inline math conversion (requires use_llm).
    """
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Validate output format
        valid_formats = ["markdown", "json", "html", "chunks"]
        if output_format not in valid_formats:
            raise HTTPException(status_code=400, detail=f"Invalid output_format. Must be one of {valid_formats}")

        # Validate llm_service if provided
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

        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as f:
            f.write(await file.read())

        # Configure marker settings
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

        # Initialize converter
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service() if use_llm else None
        )

        # Convert PDF
        rendered = converter(str(file_path))
        text, metadata, images = text_from_rendered(rendered)

        # Clean up
        file_path.unlink()

        return JSONResponse(content={
            "content": text,
            "metadata": metadata,
            "images": images
        })

    except Exception as convert_error:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(convert_error)}")

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

