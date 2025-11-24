"""
Main FastAPI application for waste classification.
Handles all API endpoints for prediction, image upload, and retraining.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import uvicorn

# Add src directory to path to allow imports
# This allows running from either the src/ directory or the parent directory
src_dir = Path(__file__).parent.absolute()
parent_dir = src_dir.parent.absolute()

# Add both src and parent to path for flexibility
for path in [str(src_dir), str(parent_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from prediction import load_model, get_class_names, get_img_size, predict_image, predict_batch
from preprocessing import upload_image_to_db, upload_images_batch, VALID_CLASSES
from model import retrain_model
from database import create_tables, get_training_runs, get_example_image_for_class
from fastapi.responses import Response


# Initialize FastAPI app
app = FastAPI(
    title="Waste Classification API",
    description="API for classifying waste images into 9 categories, uploading training data, and retraining models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and configuration
model = None
class_names = None
img_size = None


def initialize_model():
    """Initialize model and configuration on startup."""
    global model, class_names, img_size
    
    try:
        print("Attempting to load model...")
        model = load_model("models/efficientnetb0_waste_classifier.h5")
        class_names = get_class_names()
        img_size = (224, 224)
        print("Model initialized successfully!")
        print(f"  - Classes: {len(class_names)} classes")
        print(f"  - Image size: {img_size}")
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}")
        print("\nTo fix this:")
        print("1. Train the model using the notebook: notebook/waste-classification.ipynb")
        print("2. Ensure the model is saved to: models/waste_classifier_final.keras")
        print("3. Or set the model path in models/model_config.json")
        model = None
        class_names = None
        img_size = None
    except Exception as e:
        print(f"ERROR: Could not initialize model: {e}")
        print("Model will not be available until this is fixed.")
        model = None
        class_names = None
        img_size = None


@app.on_event("startup")
async def startup_event():
    """Initialize database tables and model on startup."""
    try:
        create_tables()
        print("Database tables initialized")
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")
    
    initialize_model()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Waste Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for classification",
            "/predict/batch": "POST - Upload multiple images for classification",
            "/upload": "POST - Upload image to database for retraining",
            "/upload/batch": "POST - Upload multiple images to database",
            "/retrain": "POST - Trigger model retraining",
            "/retrain/runs": "GET - List retraining runs",
            "/health": "GET - Check API health",
            "/classes": "GET - Get list of class names",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        # Try to load model if not loaded
        try:
            initialize_model()
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Model not loaded",
                    "error": str(e),
                    "help": "Train the model using the notebook first, or check model path in models/model_config.json"
                }
            )
    
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Model not loaded",
                "help": "Train the model using the notebook: notebook/waste-classification.ipynb"
            }
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "num_classes": len(class_names) if class_names else 0,
        "image_size": img_size
    }


@app.get("/classes")
async def get_classes():
    """Get list of class names."""
    if class_names is None:
        try:
            class_names_list = get_class_names()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Could not load classes: {str(e)}")
        return {"classes": class_names_list, "num_classes": len(class_names_list)}
    return {"classes": class_names, "num_classes": len(class_names)}


@app.get("/classes/{class_name}/example")
async def get_class_example(class_name: str):
    """
    Get an example image for a specific class.
    
    Args:
        class_name: Name of the class
    
    Returns:
        Image file as response
    """
    if class_name not in VALID_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class name. Valid classes: {', '.join(VALID_CLASSES)}"
        )
    
    try:
        image_bytes = get_example_image_for_class(class_name, limit=1)
        if image_bytes is None:
            raise HTTPException(
                status_code=404,
                detail=f"No example image found for class: {class_name}"
            )
        
        return Response(content=image_bytes, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving example: {str(e)}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict waste category from uploaded image.
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON response with predicted label, confidence, and all class probabilities
    """
    global model, class_names, img_size
    
    # Ensure model is loaded
    if model is None:
        try:
            initialize_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Make prediction
        result = predict_image(model, image_bytes, class_names, img_size)
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# @app.post("/predict/batch")
# async def predict_batch_endpoint(files: List[UploadFile] = File(...)):
#     """
#     Predict waste categories for multiple images.
    
#     Args:
#         files: List of uploaded image files
    
#     Returns:
#         List of predictions for each image
#     """
#     global model, class_names, img_size
    
#     if model is None:
#         try:
#             initialize_model()
#         except Exception as e:
#             raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
#     try:
#         # Read all files
#         images_data = []
#         filenames = []
#         for file in files:
#             image_bytes = await file.read()
#             images_data.append(image_bytes)
#             filenames.append(file.filename)
        
#         # Make predictions
#         results = predict_batch(model, images_data, class_names, img_size)
        
#         # Add filenames to results
#         for i, result in enumerate(results):
#             if "error" not in result:
#                 result["filename"] = filenames[i]
#             else:
#                 result["filename"] = filenames[i]
        
#         return {"results": results, "total": len(results)}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    class_name: str = None
):
    """
    Upload an image to the database for retraining.
    
    Args:
        file: Uploaded image file
        class_name: Waste category class name (required)
    
    Returns:
        Upload confirmation with image ID
    """
    if not class_name:
        raise HTTPException(
            status_code=400,
            detail="class_name parameter is required. "
                   f"Valid classes: {', '.join(VALID_CLASSES)}"
        )
    
    try:
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        image_id = upload_image_to_db(image_bytes, class_name, file.filename)
        
        return {
            "message": "Image uploaded successfully",
            "image_id": image_id,
            "class_name": class_name,
            "filename": file.filename
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.post("/upload/batch")
async def upload_images_batch_endpoint(
    files: List[UploadFile] = File(...),
    class_names: str = None
):
    """
    Upload multiple images to the database for retraining.
    
    Args:
        files: List of uploaded image files
        class_names: Comma-separated list of class names (must match number of files)
    
    Returns:
        Upload confirmation with image IDs
    """
    if not class_names:
        raise HTTPException(
            status_code=400,
            detail="class_names parameter is required. "
                   "Provide comma-separated class names matching the order of files."
        )
    
    class_name_list = [name.strip() for name in class_names.split(',')]
    
    if len(class_name_list) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Number of class names ({len(class_name_list)}) "
                   f"must match number of files ({len(files)})"
        )
    
    try:
        # Read all files
        images_data = []
        for file, class_name in zip(files, class_name_list):
            image_bytes = await file.read()
            images_data.append((image_bytes, class_name, file.filename))
        
        # Upload to database
        image_ids = upload_images_batch(images_data)
        
        return {
            "message": f"Successfully uploaded {len(image_ids)} images",
            "image_ids": image_ids,
            "total": len(image_ids)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch upload error: {str(e)}")


@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """
    Trigger model retraining using images from the database.
    This runs in the background.
    
    Returns:
        Confirmation message
    """
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    def retrain_task():
        try:
            retrain_model(model_dir)
        except Exception as e:
            print(f"Retraining failed: {e}")
    
    background_tasks.add_task(retrain_task)
    
    return {
        "message": "Model retraining started in background it will take approximately 5 to 10 minutes to complete",
        "status": "started"
    }


@app.get("/retrain/runs")
async def list_training_runs(limit: int = 10):
    """
    Get list of retraining runs to check status.
    
    Args:
        limit: Maximum number of runs to return (default: 10)
    
    Returns:
        List of training runs with status, timestamps, and metrics
    """
    try:
        runs = get_training_runs(limit=limit)
        return {
            "runs": runs,
            "total": len(runs),
            "message": "Use this endpoint to check if retraining is complete. Status can be 'started', 'completed', or 'failed'."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving training runs: {str(e)}")


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port)

