import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

DB_FILE = "roadmaps.json"

# Load API key and configure Gemini
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

class RoadmapResponse(BaseModel):
    message: str
    roadmap: dict

def generate_roadmap(startup_name: str, timeline: int) -> str:
    prompt = f"Generate a {timeline}-month startup roadmap for {startup_name}. Include key milestones."
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text

def load_roadmaps():
    """Load all roadmaps from JSON file"""
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as file:
            json.dump([], file)
    with open(DB_FILE, "r") as file:
        return json.load(file)

def save_roadmaps(roadmaps):
    """Save roadmaps to JSON file"""
    with open(DB_FILE, "w") as file:
        json.dump(roadmaps, file, indent=4)

def add_roadmap(user_id: int, startup_name: str, timeline: int) -> RoadmapResponse:
    """Add a new roadmap entry"""
    roadmap_text = generate_roadmap(startup_name, timeline)
    roadmaps = load_roadmaps()
    new_roadmap = {
        "user_id": user_id,
        "startup_name": startup_name,
        "timeline": timeline,
        "roadmap_text": roadmap_text
    }
    roadmaps.append(new_roadmap)
    save_roadmaps(roadmaps)
    return RoadmapResponse(
        message="Roadmap saved successfully",
        roadmap=new_roadmap
    )