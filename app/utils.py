import os
import tempfile
import cv2
import numpy as np
import fitz  # PyMuPDF
import imghdr
import insightface
from insightface.app import FaceAnalysis  

# Initialize RetinaFace + ArcFace (on CPU)
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0)

def save_uploaded_file(uploaded_file):
    """Save an uploaded file (image or PDF) and return the file path."""
    try:
        file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"}

        if file_extension not in allowed_extensions:
            return None, f"Unsupported file type: {file_extension}"

        # Save file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.write(uploaded_file.file.read())
        temp_file.flush()
        temp_file.close()

        # Handle PDF separately
        if file_extension == ".pdf":
            temp_image_path, err = pdf_to_image(temp_file.name)
            if err:
                os.remove(temp_file.name)
                return None, err
            os.remove(temp_file.name)  # Cleanup original PDF
            return temp_image_path, None

        # Validate if the uploaded file is an actual image
        if imghdr.what(temp_file.name) not in ["jpeg", "png", "bmp", "tiff"]:
            os.remove(temp_file.name)  # Cleanup
            return None, "File is not a valid image format."

        return temp_file.name, None  # Return valid image path
    except Exception as e:
        return None, f"[ERROR] Failed to save uploaded file: {str(e)}"

def pdf_to_image(pdf_path):
    """Convert the first page of a PDF to an image using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return None, "No pages found in PDF."

        page = doc[0]  # Get the first page
        pix = page.get_pixmap()
        
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        pix.save(temp_image_path)
        
        return temp_image_path, None
    except Exception as e:
        return None, f"[ERROR] Failed to convert PDF to image: {str(e)}"

def extract_face_embedding(image_path):
    """Detect and extract facial embeddings using ArcFace. Supports PDFs and images."""
    try:
        # Convert PDF to image if needed
        if image_path.lower().endswith(".pdf"):
            image_path, err = pdf_to_image(image_path)
            if err:
                return None, err

        # Ensure the file is an image before processing
        if imghdr.what(image_path) not in ["jpeg", "png", "bmp", "tiff"]:
            return None, "File is not a valid image format."

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Invalid or corrupted image file: {image_path}")

        faces = face_analyzer.get(img)
        num_faces = len(faces)

        if num_faces == 0:
            print(f"[DEBUG] No face detected in {image_path}")
            return None, "No face detected."

        if num_faces > 1:
            print(f"[DEBUG] Multiple faces detected ({num_faces}) in {image_path}")
            return None, f"Multiple faces detected ({num_faces}). Only one face is allowed."

        face = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])  # Pick the largest face
        embedding = face.normed_embedding

        if embedding is None or len(embedding) == 0:
            return None, "Failed to extract face embedding."

        return embedding / np.linalg.norm(embedding), None  # Normalize the embedding

    except Exception as e:
        print(f"[ERROR] Face extraction error: {str(e)}")
        return None, str(e)
