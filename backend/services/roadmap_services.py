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
    prompt = f"""Generate a structured {timeline}-month startup roadmap for solo founders.  
        Output only the following format, with no title or extra text:  
        Month 1: [Step]  
        Month 2: [Step]  
        ...until Month {timeline}  

        Make each step clear, actionable, and concise.  
        Use only plain text without special characters or formatting.  
        Start directly with Month 1, no headers or titles.  
        Do not include any additional information or notes and stick to one line for each month.  
        Ensure that the output does not contain any bold text or special formatting.  
        """
    
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text.strip()

def load_roadmaps():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as file:
            json.dump([], file)
    with open(DB_FILE, "r") as file:
        return json.load(file)

def save_roadmaps(roadmaps):
    with open(DB_FILE, "w") as file:
        json.dump(roadmaps, file, indent=4)

def add_roadmap(user_id: int, startup_name: str, timeline: int) -> RoadmapResponse:
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