# -*- coding: utf-8 -*-
"""Copia a ../demo/ las cinco fotografías de la sección DEMO del cuaderno.

Todas salen de la partición de **prueba** (ningún modelo las ve al entrenar) y se eligen de forma
determinista: la primera por orden alfabético de cada clase indicada. Cuatro son de campo (tres
clases frecuentes y la más rara, nematodos) y una es de PlantVillage: una hoja aislada sobre fondo
uniforme, un caso atípico para un modelo entrenado con fotografías de campo.

Uso:
    python preparar_demo.py            # lee ../../datos y escribe ../demo
"""
import csv
import hashlib
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CODIGO = AQUI.parent
DATOS = CODIGO.parent / "datos"
DEMO = CODIGO / "demo"

# (conjunto, carpeta de la clase, nombre corto, descripción del caso)
CASOS = [
    ("campo", "Phytopthora", "phytophthora", "Campo · tizón tardío (Phytophthora), clase comparable con PlantVillage"),
    ("campo", "Healthy", "sana", "Campo · hoja sana, clase comparable con PlantVillage"),
    ("campo", "Fungi", "hongos", "Campo · hongos, la clase más frecuente"),
    ("campo", "Nematode", "nematodos", "Caso límite · nematodos, la clase más rara (10 fotos de prueba)"),
    ("laboratorio", "Potato___Late_blight", "plantvillage_tizon_tardio",
     "Caso atípico · foto de laboratorio (PlantVillage) que el modelo de campo nunca vio"),
]
CARPETA = {"campo": "campo_shabrina", "laboratorio": "plantvillage_papa"}


def sha256(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main():
    DEMO.mkdir(exist_ok=True)
    filas = []
    for i, (conjunto, clase, corto, descripcion) in enumerate(CASOS, 1):
        with open(CODIGO / "particiones" / f"{conjunto}_particion.csv", encoding="utf-8") as f:
            candidatos = sorted(r["archivo"] for r in csv.DictReader(f)
                                if r["clase"] == clase and r["estado"] == "conservada" and r["particion"] == "prueba")
        original = candidatos[0]
        origen = DATOS / CARPETA[conjunto] / clase / original
        destino = DEMO / f"caso{i}_{corto}{origen.suffix.lower()}"
        shutil.copy2(origen, destino)
        filas.append({"caso": i, "archivo": destino.name, "conjunto": conjunto, "clase_esperada": clase,
                      "archivo_original": original, "sha256": sha256(destino), "descripcion": descripcion})
        print(f"caso {i}: {destino.name}  ←  {clase}/{original}")
    with open(DEMO / "casos_demo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
