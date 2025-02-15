from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import roadmap_generator, pitch_recommender
from services.pitch_services import PitchDeckAnalyzer

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = PitchDeckAnalyzer()

# Include the roadmap generator router
app.include_router(roadmap_generator.router)
app.include_router(pitch_recommender.router)

@app.get("/")
def root():
    return {"message": "API is running"}