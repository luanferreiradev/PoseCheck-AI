from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from keras.models import load_model
from typing import List, Optional
import tempfile
import os
from pydantic import BaseModel

# run it with: uvicorn api:app --reload
# use o local host com final /docs para acessar o swagger UI e testar o crud visual exemplo: http://localhost:8000/docs

app = FastAPI()

# Carregue seu modelo Keras treinado (substitua pelo caminho real)
MODEL_PATH = "model.h5"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Erro ao carregar o modelo: {e}")

def extract_frames(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    """Extrai frames uniformemente do vídeo."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    return frames

def preprocess_frames(frames: List[np.ndarray]) -> np.ndarray:
    """Normaliza e empilha frames para o modelo."""
    frames = np.array(frames) / 255.0
    return np.expand_dims(frames, axis=0)  # (1, num_frames, 224, 224, 3)

def get_embedding(frames: np.ndarray) -> np.ndarray:
    """Extrai embedding do modelo."""
    emb = model.predict(frames)
    # Se o modelo retorna batch, pega o primeiro
    if len(emb.shape) > 1:
        emb = emb[0]
    return emb

def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))

@app.post("/compare/")
async def compare_videos(
    reference: UploadFile = File(...),
    user: UploadFile = File(...)
):
    if model is None:
        return JSONResponse({"error": "Modelo não carregado."}, status_code=500)

    ref_path = user_path = None
    try:
        # Salva arquivos temporários
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ref_tmp, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as user_tmp:
            ref_tmp.write(await reference.read())
            user_tmp.write(await user.read())
            ref_path = ref_tmp.name
            user_path = user_tmp.name

        # Extrai e processa frames
        ref_frames = extract_frames(ref_path)
        user_frames = extract_frames(user_path)
        if not ref_frames or not user_frames:
            return JSONResponse({"error": "Não foi possível extrair frames dos vídeos."}, status_code=400)
        ref_input = preprocess_frames(ref_frames)
        user_input = preprocess_frames(user_frames)

        # Embeddings
        ref_emb = get_embedding(ref_input)
        user_emb = get_embedding(user_input)

        # Similaridade
        dist = euclidean_distance(ref_emb, user_emb)

        return {"similarity_score": dist}
    except Exception as e:
        return JSONResponse({"error": f"Erro interno: {str(e)}"}, status_code=500)
    finally:
        # Limpeza
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        if user_path and os.path.exists(user_path):
            os.remove(user_path)

# Modelo Pydantic para simular um recurso
class VideoComparison(BaseModel):
    id: int
    reference_name: str
    user_name: str
    similarity_score: Optional[float] = None

# "Banco de dados" em memória
db: List[VideoComparison] = []

@app.get("/comparisons/", response_model=List[VideoComparison])
def list_comparisons():
    return db

@app.post("/comparisons/", response_model=VideoComparison)
def create_comparison(comp: VideoComparison):
    db.append(comp)
    return comp

@app.get("/comparisons/{comp_id}", response_model=VideoComparison)
def get_comparison(comp_id: int):
    for comp in db:
        if comp.id == comp_id:
            return comp
    raise HTTPException(status_code=404, detail="Comparison not found")

@app.put("/comparisons/{comp_id}", response_model=VideoComparison)
def update_comparison(comp_id: int, comp_update: VideoComparison):
    for idx, comp in enumerate(db):
        if comp.id == comp_id:
            db[idx] = comp_update
            return comp_update
    raise HTTPException(status_code=404, detail="Comparison not found")

@app.delete("/comparisons/{comp_id}")
def delete_comparison(comp_id: int):
    for idx, comp in enumerate(db):
        if comp.id == comp_id:
            del db[idx]
            return {"detail": "Deleted"}
    raise HTTPException(status_code=404, detail="Comparison not found")