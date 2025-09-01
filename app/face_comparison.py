# app/face_comparison.py

import os
import cv2
import numpy as np
import tempfile
from fastapi import APIRouter, UploadFile, File
from sklearn.metrics.pairwise import cosine_similarity
from app.utils import save_uploaded_file, extract_face_embedding

router = APIRouter()

@router.post("/compare_faces/")
async def compare_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    """Compare two face images and return a similarity score."""

    # Save uploaded images and unpack the returned values
    temp1_path, error1 = save_uploaded_file(image1)
    temp2_path, error2 = save_uploaded_file(image2)

    # If any file failed to save, return an error message
    if error1 or error2:
        return {"match": "NO ❌", "similarity_score": 0.0, "message": error1 or error2}

    try:
        # Extract facial embeddings
        embedding1, error1 = extract_face_embedding(temp1_path)
        embedding2, error2 = extract_face_embedding(temp2_path)

        # If either embedding extraction fails, return failure
        if embedding1 is None or embedding2 is None:
            return {
                "match": "NO ❌",
                "similarity_score": 0.0,
                "message": error1 or error2
            }

        # Compute cosine similarity
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]

        # Set a better threshold for match
        match = similarity > 0.30

        return {
            "match": "YES ✅" if match else "NO ❌",
            "similarity_score": round(float(similarity), 3),
        }
    
    finally:
        # Cleanup temporary images safely
        if temp1_path:
            os.remove(temp1_path)
        if temp2_path:
            os.remove(temp2_path)
            
            
            
            
            gksgfpgjsoghohdfghghshghishifgyhidgfidgi