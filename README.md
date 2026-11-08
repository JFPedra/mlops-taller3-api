# API Pipeline Taller 3 — MLOps

## Descripción
Este proyecto entrena y expone mediante **FastAPI** un `Pipeline` productivo de **scikit-learn** para clasificación binaria a partir de lecturas de sensores (`sensor_temperatura` y `sensor_presion`).

El pipeline encapsula:
1. **Imputación estadística** (`SimpleImputer(strategy="median")`) para manejar valores nulos en `sensor_temperatura`.
2. **Normalización Z-score** (`StandardScaler()`).
3. **Clasificador predictivo** (`LogisticRegression`).

El artefacto entrenado se serializa en `pipeline.joblib` mediante `train_pipeline.py` y es consumido por la API en `main.py`.

---

## Instalación

Instala las dependencias del proyecto ejecutando:

```bash
pip install -r requirements.txt
```

*(Opcional)* Si deseas reentrenar y regenerar el archivo `pipeline.joblib`:

```bash
python train_pipeline.py
```

---

## Ejecución

Para levantar el servidor de la API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Una vez iniciado, puedes consultar la documentación interactiva (Swagger UI) en `http://127.0.0.1:8000/docs`.

---

## Ejemplo de uso con `curl`

### Petición con ambos sensores
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sensor_temperatura": 25.4, "sensor_presion": 120.0}'
```

### Petición con valor nulo (imputado automáticamente por el pipeline)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"sensor_temperatura": null, "sensor_presion": 120.0}'
```

**Respuesta esperada:**
```json
{
  "prediccion": 1,
  "probabilidad_clase_1": 0.9609
}
```

---

## Notas técnicas
- **versión de scikit-learn: 1.9.1** (Es importante mantener esta misma versión al cargar `pipeline.joblib` para garantizar compatibilidad en la deserialización del modelo).

