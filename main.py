import re
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Mecánica Don Oscar - OCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def inicio():
    return {"status": "ok", "mensaje": "Servidor OCR de Mecánica Don Oscar activo"}

@app.post("/api/v1/ocr-patente")
async def procesar_patente(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen.")

    # Procesamiento liviano de imagen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Lectura OCR ligera
    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    texto = pytesseract.image_to_string(gray, config=config)

    # Limpieza de caracteres para formato de patente argentina
    clean = re.sub(r'[^A-Z0-9]', '', texto.upper())

    # Buscar coincidencia de patente formato nuevo (AA123BB) o formato viejo (AAA123)
    match = re.search(r'([A-Z]{2}\d{3}[A-Z]{2}|[A-Z]{3}\d{3})', clean)

    if match:
        return {"status": "success", "patente": match.group(1)}
    
    return {"status": "warning", "message": "No se detectó la patente claramente", "patente": clean[:7] if len(clean)>=6 else None}
