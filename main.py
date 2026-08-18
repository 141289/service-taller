import re
import cv2
import numpy as np
import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mecánica Don Oscar - OCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = easyocr.Reader(['es'], gpu=False)

@app.get("/")
def inicio():
    return {"mensaje": "Servidor OCR de Mecánica Don Oscar activo"}

@app.post("/api/v1/ocr-patente")
async def procesar_patente(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    resultados = reader.readtext(blurred)

    for (_, texto, probabilidad) in resultados:
        clean = re.sub(r'[^A-Z0-9]', '', texto.upper())
        if 6 <= len(clean) <= 7 and probabilidad > 0.35:
            return {"status": "success", "patente": clean, "confianza": round(probabilidad * 100, 2)}

    return {"status": "warning", "message": "No se detectó patente con claridad", "patente": None}