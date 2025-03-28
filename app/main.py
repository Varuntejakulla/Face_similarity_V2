# app/main.py

from fastapi import FastAPI, File, UploadFile
from app.face_comparison import compare_faces
from fastapi.middleware.cors import CORSMiddleware
from app.face_comparison import router as compare_faces_router
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
     allow_origins=["*"],  # Replace with Angular frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add the route for comparing faces
app.include_router(compare_faces_router)
