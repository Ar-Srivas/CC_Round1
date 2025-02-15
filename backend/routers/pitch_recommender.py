from fastapi import APIRouter, File, UploadFile, HTTPException
from services.pitch_services import PitchDeckAnalyzer
import shutil
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pitch", tags=["pitch"])
analyzer = PitchDeckAnalyzer()

# Create temp directory if it doesn't exist
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

@router.post("/analyze")
async def analyze_pitch_deck(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    temp_path = TEMP_DIR / f"upload_{file.filename}"
    try:
        logger.debug(f"Saving file to: {temp_path}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.debug("File saved, starting analysis")
        result = analyzer.analyze_pdf(str(temp_path))
        logger.debug("Analysis completed")
        return result
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug("Temporary file cleaned up")
            except Exception as e:
                logger.error(f"Failed to delete temp file: {str(e)}")