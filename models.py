from pydantic import BaseModel, Field, validator
from typing import Literal


class PrediccionRequest(BaseModel):
    tipo_terapia: Literal["Lenguaje", "Psicologica"] = Field(
        ...,
        description="Tipo de terapia"
    )
    categoria_material: Literal["MaterialSensorial", "MaterialLectura"] = Field(
        ...,
        description="Categoria del material"
    )
    anio: int = Field(..., ge=2020, le=2030, description="Año de prediccion")
    mes: int = Field(..., ge=1, le=12, description="Mes de prediccion")
    demanda_lag_1m: float = Field(..., ge=0, description="Demanda del mes anterior")
    num_sesiones: int = Field(..., ge=0, description="Numero de sesiones")
    demanda_total: float = Field(..., ge=0, description="Demanda total actual")
    es_inicio_ciclo: int = Field(..., ge=0, le=1, description="1 si es inicio de ciclo, 0 sino")

    @validator('tipo_terapia', 'categoria_material')
    def validar_segmento(cls, v, values, field):
        if field.name == 'categoria_material' and 'tipo_terapia' in values:
            tipo = values['tipo_terapia']
            material = v

            segmentos_validos = [
                ("Lenguaje", "MaterialSensorial"),
                ("Psicologica", "MaterialLectura"),
                ("Psicologica", "MaterialSensorial")
            ]

            if (tipo, material) not in segmentos_validos:
                raise ValueError(f"Segmento {tipo} + {material} no soportado por el modelo")

        return v


class PrediccionResponse(BaseModel):
    demanda_predicha: float
    tipo_terapia: str
    categoria_material: str
    anio: int
    mes: int
    modelo_version: str
    features_utilizadas: dict


class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool
    modelo_tipo: str
    fecha_entrenamiento: str
    metricas_test: dict