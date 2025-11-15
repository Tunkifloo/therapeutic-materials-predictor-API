# test_ruta.py
from pathlib import Path
import os

print("Directorio actual:", os.getcwd())
print("\nContenido de la carpeta model:")

model_dir = Path("model")
if model_dir.exists():
    for item in model_dir.iterdir():
        print(f"  {item.name}")
else:
    print("  La carpeta model NO existe")

print("\n¿El archivo existe?")
ruta = Path("model/modelo_alto_volumen_Lasso_alpha5_0_20251115_005312.pkl")
print(f"  {ruta}: {ruta.exists()}")