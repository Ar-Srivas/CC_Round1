import json
import os

DB_FILE = "roadmaps.json"

# Initialize the JSON file if it doesn't exist
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as file:
        json.dump([], file)

def load_roadmaps():
    """Load all roadmaps from JSON file"""
    with open(DB_FILE, "r") as file:
        return json.load(file)

def save_roadmaps(roadmaps):
    """Save roadmaps to JSON file"""
    with open(DB_FILE, "w") as file:
        json.dump(roadmaps, file, indent=4)

def add_roadmap(user_id: int, startup_name: str, timeline: int, roadmap_text: str):
    """Add a new roadmap entry"""
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

def get_all_roadmaps():
    """Retrieve all saved roadmaps"""
    return load_roadmaps()