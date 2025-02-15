import json
import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from services.roadmap_services import add_roadmap, load_roadmaps, RoadmapResponse
from services.pitch_services import PitchDeckAnalyzer
import shutil
import google.generativeai as genai
from dotenv import load_dotenv
from database import add_roadmap
from pydantic import BaseModel

DB_FILE = "roadmaps.json"

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

def generate_roadmap(startup_name: str, timeline: int):
    prompt = f"Generate a {timeline}-month startup roadmap for {startup_name}. Include key milestones."
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text

def save_roadmap(user_id: int, startup_name: str, timeline: int):
    roadmap_text = generate_roadmap(startup_name, timeline)
    return add_roadmap(user_id, startup_name, timeline,roadmap_text)


# Initialize the JSON file if it doesn't exist
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as file:
        json.dump([], file)

def load_roadmaps():
    with open(DB_FILE, "r") as file:
        return json.load(file)

def save_roadmaps(roadmaps):
    with open(DB_FILE, "w") as file:
        json.dump(roadmaps, file, indent=4)

def add_roadmap(user_id: int, startup_name: str, timeline: int, roadmap_text: str):
    roadmaps = load_roadmaps()
    new_roadmap = {
        "user_id": user_id,
        "startup_name": startup_name,
        "timeline": timeline,
        "roadmap_text": roadmap_text
    }
    roadmaps.append(new_roadmap)
    save_roadmaps(roadmaps)
    return {"message": "Roadmap saved successfully", "roadmap": new_roadmap}


router = APIRouter(prefix="/api/roadmap", tags=["roadmap"])
analyzer = PitchDeckAnalyzer()

@router.post("/pitch/analyze")
async def analyze_pitch_deck(file: UploadFile = File(...)):
    temp_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = analyzer.analyze_pdf(temp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

class RoadmapRequest(BaseModel):
    user_id: int
    startup_name: str
    timeline: int

@router.post("/create", response_model=RoadmapResponse)
async def create_roadmap(request: RoadmapRequest):
    try:
        return add_roadmap(
            user_id=request.user_id,
            startup_name=request.startup_name,
            timeline=request.timeline
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all")
async def get_all_roadmaps():
    return {"roadmaps": load_roadmaps()}
