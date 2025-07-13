import cv2
import numpy as np
import face_recognition
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FacePayload(BaseModel):
    image1: str  
    image2: str  

@app.post("/match-face")
def match_face(payload: FacePayload):
    try:
        def decode_image(image_b64: str):
            image_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image data.")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img1_rgb = decode_image(payload.image1)
        img2_rgb = decode_image(payload.image2)

        encodings1 = face_recognition.face_encodings(img1_rgb)
        encodings2 = face_recognition.face_encodings(img2_rgb)

        if not encodings1:
            return {"result": "NO_ENCODING_IN_IMAGE1"}
        if not encodings2:
            return {"result": "NO_ENCODING_IN_IMAGE2"}

        known_encoding = encodings1[0]
        for test_encoding in encodings2:
            distance = face_recognition.face_distance([known_encoding], test_encoding)[0]
            if distance < 0.5:
                return {"result": "MATCH_FOUND"}

        return {"result": "NO_MATCH_FOUND"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")