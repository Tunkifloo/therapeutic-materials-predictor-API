from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import PrediccionRequest, PrediccionResponse, HealthResponse
from predictor import ModeloPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Prediccion Demanda Materiales Terapeuticos",
    description="API para predecir demanda mensual de materiales terapeuticos usando modelo Lasso",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELO_PATH = "model/modelo_alto_volumen_Lasso_alpha5_0_20251115_005312.pkl"
METADATA_PATH = "model/metadata_modelo_alto_volumen_20251115_005312.json"

predictor = None


@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = ModeloPredictor(MODELO_PATH, METADATA_PATH)
        logger.info("Modelo cargado exitosamente")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        raise


@app.get("/", tags=["General"])
async def root():
    return {
        "mensaje": "API de Prediccion de Demanda de Materiales Terapeuticos",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "info": "/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    info = predictor.get_info()

    return HealthResponse(
        status="ok",
        modelo_cargado=True,
        modelo_tipo=info['modelo_tipo'],
        fecha_entrenamiento=info['fecha_entrenamiento'],
        metricas_test=info['metricas_test']
    )


@app.post("/predict", response_model=PrediccionResponse, tags=["Prediccion"])
async def predecir_demanda(request: PrediccionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        data = request.dict()
        resultado = predictor.predecir(data)
        return PrediccionResponse(**resultado)
    except Exception as e:
        logger.error(f"Error en prediccion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {str(e)}")


@app.get("/info", tags=["General"])
async def modelo_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    return predictor.get_info()