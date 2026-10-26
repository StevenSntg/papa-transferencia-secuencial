# -*- coding: utf-8 -*-
"""Descarga los dos conjuntos de datos del trabajo y verifica su integridad.

1. Campo — «Potato Leaf Disease Dataset in Uncontrolled Environment» (Shabrina et al., 2023),
   Mendeley Data, DOI 10.17632/ptz377bwb8.1, licencia CC BY 4.0. Se usa la API pública de
   Mendeley Data (no requiere credenciales) y se comprueba el SHA-256 de cada imagen.
2. Laboratorio — subconjunto de papa de PlantVillage (Hughes y Salathé, 2015), tomado del
   repositorio de sus autores: https://github.com/spMohanty/PlantVillage-Dataset (carpeta
   raw/color). Se clona solo lo necesario con un «sparse checkout» de git.

Uso:
    python descargar_datasets.py            # descarga en ../../datos
    python descargar_datasets.py --destino D:/datos

El script se puede interrumpir y volver a ejecutar: omite lo que ya está descargado y verificado.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MENDELEY_ID = "ptz377bwb8"
MENDELEY_VERSION = 1
API = f"https://data.mendeley.com/public-api/datasets/{MENDELEY_ID}"
CABECERAS = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.mendeley-public-dataset.1+json"}

PLANTVILLAGE_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PLANTVILLAGE_CLASES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


def pedir_json(url):
    req = urllib.request.Request(url, headers=CABECERAS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def bajar_archivo(item, carpeta, intentos=4):
    """Descarga una imagen y valida tamaño y SHA-256. Devuelve (ruta, estado)."""
    destino = carpeta / item["filename"]
    esperado = item["content_details"]["sha256_hash"]
    if destino.exists() and destino.stat().st_size == item["size"] and sha256(destino) == esperado:
        return destino, "ya estaba"
    ultimo_error = None
    for _ in range(intentos):
        try:
            req = urllib.request.Request(item["content_details"]["download_url"], headers={"User-Agent": "Mozilla/5.0"})
            tmp = destino.with_suffix(destino.suffix + ".part")
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if sha256(tmp) != esperado:
                raise IOError("SHA-256 no coincide")
            tmp.replace(destino)
            return destino, "descargado"
        except Exception as e:  # reintento simple ante cortes de red
            ultimo_error = e
    return destino, f"ERROR: {ultimo_error}"


def descargar_campo(destino, hilos=8):
    raiz = destino / "campo_shabrina"
    carpetas = pedir_json(f"{API}/folders/{MENDELEY_VERSION}")
    tareas = []
    for c in carpetas:
        archivos = pedir_json(f"{API}/files?folder_id={c['id']}&version={MENDELEY_VERSION}")
        (raiz / c["name"]).mkdir(parents=True, exist_ok=True)
        print(f"  {c['name']:<12} {len(archivos):>4} imágenes", flush=True)
        tareas += [(a, raiz / c["name"], c["name"]) for a in archivos]
    print(f"  total        {len(tareas):>4} imágenes · {sum(a['size'] for a, _, _ in tareas) / 1e6:.0f} MB", flush=True)

    filas, errores, hechos = [], 0, 0
    with ThreadPoolExecutor(max_workers=hilos) as ex:
        futuros = {ex.submit(bajar_archivo, a, carpeta): (a, clase) for a, carpeta, clase in tareas}
        for fut in as_completed(futuros):
            a, clase = futuros[fut]
            ruta, estado = fut.result()
            hechos += 1
            errores += estado.startswith("ERROR")
            filas.append([clase, a["filename"], a["size"], a["content_details"]["sha256_hash"], estado])
            if hechos % 200 == 0 or hechos == len(tareas):
                print(f"  {hechos}/{len(tareas)} (errores: {errores})", flush=True)

    with open(raiz / "manifiesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "archivo", "bytes", "sha256", "estado"])
        w.writerows(sorted(filas))
    return errores


def borrar_arbol(ruta):
    """rmtree que también elimina archivos de solo lectura (los objetos de .git en Windows)."""
    def forzar(func, r, _exc):
        os.chmod(r, stat.S_IWRITE)
        func(r)

    if ruta.exists():
        shutil.rmtree(ruta, onerror=forzar)


def descargar_plantvillage(destino):
    raiz = destino / "plantvillage_papa"
    if all((raiz / c).is_dir() and any((raiz / c).iterdir()) for c in PLANTVILLAGE_CLASES):
        print("  ya estaba descargado")
        return 0
    tmp = destino / "_plantvillage_git"
    borrar_arbol(tmp)
    rutas = [f"raw/color/{c}" for c in PLANTVILLAGE_CLASES]
    ordenes = [
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", PLANTVILLAGE_REPO, str(tmp)],
        ["git", "-C", str(tmp), "sparse-checkout", "set", "--no-cone", *rutas, "/LICENSE*", "/README*"],
    ]
    for o in ordenes:
        subprocess.run(o, check=True)
    commit = subprocess.run(["git", "-C", str(tmp), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    raiz.mkdir(parents=True, exist_ok=True)
    for c in PLANTVILLAGE_CLASES:
        shutil.copytree(tmp / "raw" / "color" / c, raiz / c, dirs_exist_ok=True)
    for extra in list(tmp.glob("LICENSE*")) + list(tmp.glob("README*")):
        shutil.copy2(extra, raiz / f"ORIGEN_{extra.name}")
    (raiz / "ORIGEN.txt").write_text(
        f"Fuente: {PLANTVILLAGE_REPO}\nCommit: {commit}\nCarpetas: {', '.join(rutas)}\n", encoding="utf-8"
    )
    borrar_arbol(tmp)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--destino", default=str(Path(__file__).resolve().parents[2] / "datos"))
    ap.add_argument("--solo", choices=["campo", "plantvillage"])
    args = ap.parse_args()
    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)

    errores = 0
    if args.solo in (None, "plantvillage"):
        print("PlantVillage (papa):", flush=True)
        errores += descargar_plantvillage(destino)
    if args.solo in (None, "campo"):
        print("Conjunto de campo (Mendeley Data):", flush=True)
        errores += descargar_campo(destino)
    print("Terminado." if not errores else f"Terminado con {errores} errores: vuelva a ejecutar el script.")
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
