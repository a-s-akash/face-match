import cv2
import numpy as np
import face_recognition
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Enable CORS for Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FacePayload(BaseModel):
    image1: str  # Base64 image of known face
    image2: str  # Base64 image to compare

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

        # Decode both images
        img1_rgb = decode_image(payload.image1)
        img2_rgb = decode_image(payload.image2)

        # Get face encodings
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# import cv2
# import numpy as np
# import face_recognition
# import base64
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn

# app = FastAPI()

# # Enable CORS for Flutter Web
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class FacePayload(BaseModel):
#     image: str  # Single base64 string

# @app.post("/match-face")
# def match_face(payload: FacePayload):
#     try:
#         # Decode incoming base64 image
#         image_data = base64.b64decode(payload.image)
#         np_arr = np.frombuffer(image_data, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         if img is None:
#             raise HTTPException(status_code=400, detail="Invalid image.")

#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodings = face_recognition.face_encodings(img_rgb)
#         if not encodings:
#             return {"result": "NO_ENCODING_IN_INPUT"}

#         known_encoding = encodings[0]

#         # Open camera
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             raise HTTPException(status_code=500, detail="Camera error.")
#         for _ in range(5):
#             cap.read()
#         ret, frame = cap.read()
#         cap.release()
#         if not ret:
#             raise HTTPException(status_code=500, detail="Camera capture failed.")

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)
#         if not face_locations:
#             return {"result": "NO_FACE_IN_LIVE"}

#         live_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#         for live_encoding in live_encodings:
#             distance = face_recognition.face_distance([known_encoding], live_encoding)[0]
#             if distance < 0.5:
#                 return {"result": "MATCH_FOUND"}

#         return {"result": "NO_MATCH_FOUND"}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
