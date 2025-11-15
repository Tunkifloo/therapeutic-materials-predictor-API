# 🏥 API de Predicción de Demanda de Materiales Terapéuticos

API REST desarrollada con FastAPI para predecir la demanda mensual de materiales terapéuticos en centros de rehabilitación pediátrica utilizando un modelo de Machine Learning basado en Lasso Regression.

## 📋 Tabla de Contenidos

- [Características](#características)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Endpoints](#endpoints)
- [Modelo de Datos](#modelo-de-datos)
- [Ejemplos](#ejemplos)
- [Monitoreo](#monitoreo)
- [Desarrollo](#desarrollo)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Licencia](#licencia)

## ✨ Características

- **Alto Rendimiento**: Predicciones en tiempo real con latencia < 100ms
- **Validación Robusta**: Validación automática de datos de entrada con Pydantic
- **Documentación Interactiva**: Swagger UI y ReDoc integrados
- **CORS Habilitado**: Listo para integraciones frontend
- **Health Checks**: Endpoints de monitoreo incluidos
- **Logging Estructurado**: Sistema de logs para debugging y auditoría
- **Error Handling**: Manejo robusto de errores con mensajes descriptivos

## 🏗️ Arquitectura
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Cliente   │ ───> │   FastAPI    │ ───> │   Modelo    │
│  (REST API) │ <─── │  (Endpoint)  │ <─── │   Lasso     │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Validación  │
                     │   Pydantic   │
                     └──────────────┘
```

### Componentes

- **main.py**: Punto de entrada de la aplicación y definición de endpoints
- **models.py**: Esquemas de validación con Pydantic
- **predictor.py**: Lógica de predicción y carga del modelo
- **requirements.txt**: Dependencias del proyecto

## 📦 Requisitos

### Software

- Python 3.9+
- pip 21.0+
- virtualenv (recomendado)

### Hardware Mínimo

- CPU: 2 cores
- RAM: 2GB
- Disco: 500MB

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd api-prediccion-demanda
```

### 2. Crear entorno virtual
```bash
python3.9 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Estructura de directorios
```bash
mkdir -p modelo
# Copiar archivos del modelo a la carpeta modelo/
```

Estructura final:
```
api/
├── main.py
├── models.py
├── predictor.py
├── requirements.txt
├── README.md
└── modelo/
    ├── modelo_alto_volumen_Lasso_alpha5_0_20251115_005312.pkl
    ├── metadata_modelo_alto_volumen_20251115_005312.json
    └── features_modelo_alto_volumen_20251115_005312.csv
```

## ⚙️ Configuración

### Variables de Entorno (Opcional)

Crear archivo `.env`:
```bash
# Configuración del servidor
HOST=0.0.0.0
PORT=8000
RELOAD=True

# Rutas del modelo
MODELO_PATH=modelo/modelo_alto_volumen_Lasso_alpha5_0_20251115_005312.pkl
METADATA_PATH=modelo/metadata_modelo_alto_volumen_20251115_005312.json

# Logging
LOG_LEVEL=INFO
```

### Configuración de CORS

Por defecto, CORS está habilitado para todos los orígenes. Para producción, modificar en `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-dominio.com"],  # Especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🎯 Uso

### Iniciar el servidor

#### Modo Desarrollo
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Modo Producción
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Acceder a la documentación

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔌 Endpoints

### GET /

Información general de la API.

**Response:**
```json
{
  "mensaje": "API de Prediccion de Demanda de Materiales Terapeuticos",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "info": "/info"
  }
}
```

### GET /health

Verificación del estado del servicio.

**Response:**
```json
{
  "status": "ok",
  "modelo_cargado": true,
  "modelo_tipo": "Lasso",
  "fecha_entrenamiento": "20251115_005312",
  "metricas_test": {
    "MAE": 168.80,
    "R2": 0.8297
  }
}
```

### POST /predict

Realizar predicción de demanda.

**Request Body:**
```json
{
  "tipo_terapia": "Lenguaje",
  "categoria_material": "MaterialSensorial",
  "anio": 2025,
  "mes": 3,
  "demanda_lag_1m": 2000.0,
  "num_sesiones": 150,
  "demanda_total": 2100.0,
  "es_inicio_ciclo": 0
}
```

**Response:**
```json
{
  "demanda_predicha": 2185.13,
  "tipo_terapia": "Lenguaje",
  "categoria_material": "MaterialSensorial",
  "anio": 2025,
  "mes": 3,
  "modelo_version": "20251115_005312",
  "features_utilizadas": {
    "anio_centrado": 3,
    "es_temporada_alta": 1,
    "es_temporada_baja": 0,
    "demanda_lag_1m": 2000.0,
    "materiales_por_sesion": 14.0,
    "es_inicio_ciclo": 0,
    "tipo_Psicologica": 0,
    "material_MaterialSensorial": 1
  }
}
```

### GET /info

Información detallada del modelo.

**Response:**
```json
{
  "modelo_tipo": "Lasso",
  "fecha_entrenamiento": "20251115_005312",
  "metricas_test": {
    "MAE": 168.80,
    "R2": 0.8297
  },
  "num_features": 8,
  "features_activas": 6,
  "segmentos": [
    ["Lenguaje", "MaterialSensorial"],
    ["Psicologica", "MaterialLectura"],
    ["Psicologica", "MaterialSensorial"]
  ]
}
```

## 📊 Modelo de Datos

### Segmentos Soportados

El modelo solo soporta las siguientes combinaciones:

| Tipo Terapia | Categoría Material |
|--------------|-------------------|
| Lenguaje | MaterialSensorial |
| Psicologica | MaterialLectura |
| Psicologica | MaterialSensorial |

### Campos de Entrada

| Campo | Tipo | Descripción | Rango | Ejemplo |
|-------|------|-------------|-------|---------|
| tipo_terapia | string | Tipo de terapia | Lenguaje, Psicologica | "Lenguaje" |
| categoria_material | string | Categoría del material | MaterialSensorial, MaterialLectura | "MaterialSensorial" |
| anio | int | Año de predicción | 2020-2030 | 2025 |
| mes | int | Mes de predicción | 1-12 | 3 |
| demanda_lag_1m | float | Demanda del mes anterior | >= 0 | 2000.0 |
| num_sesiones | int | Número de sesiones | >= 0 | 150 |
| demanda_total | float | Demanda total actual | >= 0 | 2100.0 |
| es_inicio_ciclo | int | Indicador de inicio de ciclo | 0, 1 | 0 |

### Campos de Salida

| Campo | Tipo | Descripción |
|-------|------|-------------|
| demanda_predicha | float | Demanda predicha en unidades |
| tipo_terapia | string | Tipo de terapia ingresado |
| categoria_material | string | Categoría de material ingresado |
| anio | int | Año de predicción |
| mes | int | Mes de predicción |
| modelo_version | string | Versión del modelo utilizado |
| features_utilizadas | object | Features construidas para la predicción |

## 💡 Ejemplos

### cURL
```bash
# Predicción básica
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tipo_terapia": "Lenguaje",
    "categoria_material": "MaterialSensorial",
    "anio": 2025,
    "mes": 3,
    "demanda_lag_1m": 2000.0,
    "num_sesiones": 150,
    "demanda_total": 2100.0,
    "es_inicio_ciclo": 0
  }'

# Health check
curl http://localhost:8000/health

# Info del modelo
curl http://localhost:8000/info
```

### Python
```python
import requests

# Configuración
API_URL = "http://localhost:8000"

# Realizar predicción
def predecir_demanda(datos):
    response = requests.post(f"{API_URL}/predict", json=datos)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Ejemplo de uso
datos = {
    "tipo_terapia": "Psicologica",
    "categoria_material": "MaterialLectura",
    "anio": 2025,
    "mes": 6,
    "demanda_lag_1m": 1600.0,
    "num_sesiones": 120,
    "demanda_total": 1650.0,
    "es_inicio_ciclo": 0
}

resultado = predecir_demanda(datos)
print(f"Demanda predicha: {resultado['demanda_predicha']} materiales")
```

### JavaScript (Node.js)
```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000';

async function predecirDemanda(datos) {
  try {
    const response = await axios.post(`${API_URL}/predict`, datos);
    return response.data;
  } catch (error) {
    console.error('Error:', error.response.data);
    throw error;
  }
}

const datos = {
  tipo_terapia: 'Lenguaje',
  categoria_material: 'MaterialSensorial',
  anio: 2025,
  mes: 3,
  demanda_lag_1m: 2000.0,
  num_sesiones: 150,
  demanda_total: 2100.0,
  es_inicio_ciclo: 0
};

predecirDemanda(datos)
  .then(resultado => {
    console.log(`Demanda predicha: ${resultado.demanda_predicha} materiales`);
  })
  .catch(error => {
    console.error('Error en la predicción:', error);
  });
```

### Postman Collection

Importar la siguiente colección en Postman:
```json
{
  "info": {
    "name": "API Prediccion Demanda",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/health"
      }
    },
    {
      "name": "Prediccion",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"tipo_terapia\": \"Lenguaje\",\n  \"categoria_material\": \"MaterialSensorial\",\n  \"anio\": 2025,\n  \"mes\": 3,\n  \"demanda_lag_1m\": 2000.0,\n  \"num_sesiones\": 150,\n  \"demanda_total\": 2100.0,\n  \"es_inicio_ciclo\": 0\n}"
        },
        "url": "{{base_url}}/predict"
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000"
    }
  ]
}
```

## 📈 Monitoreo

### Logs

Los logs se generan automáticamente en stdout:
```bash
# Ver logs en tiempo real
tail -f uvicorn.log

# Filtrar errores
grep ERROR uvicorn.log
```

### Métricas del Modelo

Características del modelo actual:

- **Tipo**: Lasso Regression (alpha=5.0)
- **MAE Test**: 168.80 materiales
- **R² Test**: 0.8297
- **Features Activas**: 6 de 8
- **Registros Entrenamiento**: 55
- **Registros Test**: 14

### Performance

Benchmarks en hardware estándar:

- Tiempo de carga del modelo: ~500ms
- Latencia por predicción: ~20ms
- Throughput: ~50 req/s (single worker)

## 🛠️ Desarrollo

### Ejecutar tests
```bash
# Instalar dependencias de desarrollo
pip install pytest pytest-cov httpx

# Ejecutar tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=. --cov-report=html
```

### Pre-commit hooks
```bash
# Instalar pre-commit
pip install pre-commit

# Configurar hooks
pre-commit install

# Ejecutar manualmente
pre-commit run --all-files
```

### Code Style

El proyecto sigue PEP 8. Para validar:
```bash
# Instalar herramientas
pip install black flake8 isort

# Formatear código
black .
isort .

# Validar estilo
flake8 .
```

## 🚢 Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Construir y ejecutar:
```bash
# Build
docker build -t api-prediccion-demanda .

# Run
docker run -d -p 8000:8000 --name api-demanda api-prediccion-demanda

# Logs
docker logs -f api-demanda
```

### Docker Compose
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./modelo:/app/modelo
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-prediccion-demanda
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-demanda
  template:
    metadata:
      labels:
        app: api-demanda
    spec:
      containers:
      - name: api
        image: api-prediccion-demanda:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Cloud Platforms

#### AWS (Elastic Beanstalk)
```bash
eb init -p python-3.9 api-prediccion-demanda
eb create api-demanda-env
eb deploy
```

#### Google Cloud (Cloud Run)
```bash
gcloud run deploy api-prediccion-demanda \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure (App Service)
```bash
az webapp up --name api-prediccion-demanda \
  --runtime "PYTHON:3.9" \
  --sku B1
```

## 🔧 Troubleshooting

### Error: Modelo no cargado

**Problema**: `503 Service Unavailable - Modelo no cargado`

**Solución**:
```bash
# Verificar que los archivos del modelo existen
ls -la modelo/

# Verificar permisos
chmod 644 modelo/*

# Revisar logs
tail -f uvicorn.log | grep ERROR
```

### Error: Segmento no soportado

**Problema**: `ValueError: Segmento X + Y no soportado por el modelo`

**Solución**: Verificar que la combinación de tipo_terapia y categoria_material sea una de las válidas:
- Lenguaje + MaterialSensorial
- Psicologica + MaterialLectura
- Psicologica + MaterialSensorial

### Error: Dependencias

**Problema**: `ModuleNotFoundError`

**Solución**:
```bash
# Reinstalar dependencias
pip install --force-reinstall -r requirements.txt

# Verificar versiones
pip list | grep -E "fastapi|uvicorn|pandas|scikit-learn"
```

### Performance lento

**Problema**: Alta latencia en predicciones

**Solución**:
```bash
# Aumentar workers
uvicorn main:app --workers 4

# Usar Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crear una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abrir un Pull Request

## 📞 Soporte

Para reportar bugs o solicitar features:
- Abrir un issue en GitHub
- Email: nicolocisneros@gmail.com

