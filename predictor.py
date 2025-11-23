import pickle
import json
import pandas as pd
from pathlib import Path


class ModeloPredictor:
    def __init__(self, modelo_path: str, metadata_path: str):
        self.modelo_path = Path(modelo_path)
        self.metadata_path = Path(metadata_path)
        self.modelo = None
        self.metadata = None
        self.cargar_modelo()

    def cargar_modelo(self):
        with open(self.modelo_path, 'rb') as f:
            self.modelo = pickle.load(f)

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def construir_features(self, data: dict) -> pd.DataFrame:
        anio_base = 2022
        anio_centrado = data['anio'] - anio_base

        es_temporada_alta = 1 if data['mes'] in [3, 4, 5] else 0
        es_temporada_baja = 1 if data['mes'] in [1, 2, 12] else 0

        materiales_por_sesion = data['demanda_total'] / data['num_sesiones'] if data['num_sesiones'] > 0 else 0

        tipo_Psicologica = 1 if data['tipo_terapia'] == 'Psicologica' else 0
        material_MaterialSensorial = 1 if data['categoria_material'] == 'MaterialSensorial' else 0

        features_dict = {
            'anio_centrado': anio_centrado,
            'es_temporada_alta': es_temporada_alta,
            'es_temporada_baja': es_temporada_baja,
            'demanda_lag_1m': data['demanda_lag_1m'],
            'materiales_por_sesion': materiales_por_sesion,
            'es_inicio_ciclo': data['es_inicio_ciclo'],
            'tipo_Psicologica': tipo_Psicologica,
            'material_MaterialSensorial': material_MaterialSensorial
        }

        return pd.DataFrame([features_dict])

    def predecir(self, data: dict) -> dict:
        features_df = self.construir_features(data)

        if isinstance(self.modelo, dict):

            intercepto = self.metadata.get('intercepto', self.modelo.get('intercepto', 0.0))
            prediccion_valor = intercepto
            coeficientes_metadata = self.metadata.get('coeficientes', [])

            for item in coeficientes_metadata:
                nombre_feature = item['Feature']
                coeficiente = item['Coeficiente']

                if nombre_feature in features_df.columns:
                    valor_input = features_df[nombre_feature].values[0]
                    prediccion_valor += valor_input * coeficiente

            prediccion = prediccion_valor

        else:
            prediccion = self.modelo.predict(features_df)[0]

        prediccion = max(0, prediccion)

        resultado = {
            'demanda_predicha': round(prediccion, 2),
            'tipo_terapia': data['tipo_terapia'],
            'categoria_material': data['categoria_material'],
            'anio': data['anio'],
            'mes': data['mes'],
            'modelo_version': self.metadata['fecha_entrenamiento'],
            'features_utilizadas': features_df.to_dict('records')[0]
        }

        return resultado

    def get_info(self) -> dict:
        return {
            'modelo_tipo': self.metadata['modelo_tipo'],
            'fecha_entrenamiento': self.metadata['fecha_entrenamiento'],
            'metricas_test': self.metadata['metricas_test'],
            'num_features': self.metadata['num_features'],
            'features_activas': self.metadata['features_activas'],
            'segmentos': self.metadata['segmentos']
        }