# -*- coding: utf-8 -*-
"""Audita los dos conjuntos de datos y genera las particiones de referencia.

Pasos (los mismos para el conjunto de campo y para el de laboratorio):
1. Duplicados exactos: imágenes con el mismo SHA-256. Si las copias están en la misma clase se
   conserva una; si están en clases distintas (etiqueta contradictoria) se descartan todas.
2. Casi duplicados: huella perceptual dHash de 256 bits; dos imágenes se agrupan si su distancia
   de Hamming es menor o igual que UMBRAL. No se eliminan: el grupo completo va a una sola
   partición, para que ninguna imagen de prueba tenga una casi copia en el entrenamiento.
3. Partición estratificada por clase y por grupo: 70 % entrenamiento, 15 % validación, 15 % prueba,
   con semilla fija.

Uso:
    python auditar_y_particionar.py            # lee ../../datos y escribe ../particiones

Salidas (en 02_CODIGO/particiones/): campo_particion.csv, laboratorio_particion.csv y
resumen_particiones.json.
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parents[1] / "datos"
SALIDA = AQUI.parent / "particiones"

SEMILLA = 42
PROPORCIONES = {"prueba": 0.15, "validacion": 0.15}  # el resto es entrenamiento
LADO_HASH = 16          # dHash de 16 × 16 = 256 bits
UMBRAL = 12             # distancia de Hamming máxima para considerar casi duplicado (< 5 % de los bits)
EXTENSIONES = {".jpg", ".jpeg", ".png"}

CONJUNTOS = {
    "campo": DATOS / "campo_shabrina",
    "laboratorio": DATOS / "plantvillage_papa",
}


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def dhash(ruta):
    """Huella dHash: compara cada píxel con su vecino derecho en una miniatura en grises."""
    with Image.open(ruta) as im:
        im.draft("L", (LADO_HASH * 8, LADO_HASH * 8))  # decodificación JPEG reducida: más rápida
        im = im.convert("L").resize((LADO_HASH + 1, LADO_HASH), Image.LANCZOS)
        a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def procesar(ruta):
    return sha256(ruta), dhash(ruta)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def raiz(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def unir(self, i, j):
        ri, rj = self.raiz(i), self.raiz(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)


def auditar(nombre, carpeta):
    rutas = sorted(p for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES)
    clases = [p.parent.name for p in rutas]
    with ThreadPoolExecutor(8) as ex:
        resultados = list(ex.map(procesar, rutas))
    shas = [r[0] for r in resultados]

    # 1. Duplicados exactos.
    por_sha = defaultdict(list)
    for i, s in enumerate(shas):
        por_sha[s].append(i)
    estado = ["conservada"] * len(rutas)
    grupos_exactos = entre_clases = 0
    for idx in por_sha.values():
        if len(idx) == 1:
            continue
        grupos_exactos += 1
        if len({clases[i] for i in idx}) > 1:
            entre_clases += 1
            for i in idx:
                estado[i] = "descartada: misma imagen en dos clases"
        else:
            for i in idx[1:]:
                estado[i] = "descartada: copia exacta"
    vivos = [i for i in range(len(rutas)) if estado[i] == "conservada"]

    # 2. Casi duplicados entre las imágenes conservadas.
    bits = np.stack([resultados[i][1] for i in vivos]).astype(np.uint8)
    uf = UnionFind(len(vivos))
    pares = pares_entre_clases = 0
    for a in range(len(vivos)):
        dist = np.count_nonzero(bits[a + 1:] != bits[a], axis=1)
        for b in np.nonzero(dist <= UMBRAL)[0]:
            b = a + 1 + int(b)
            uf.unir(a, b)
            pares += 1
            if clases[vivos[a]] != clases[vivos[b]]:
                pares_entre_clases += 1
    miembros = defaultdict(list)
    for a in range(len(vivos)):
        miembros[uf.raiz(a)].append(vivos[a])
    grupos = list(miembros.values())
    grupos_casi = [g for g in grupos if len(g) > 1]

    # 3. Partición estratificada por clase y por grupo.
    rng = random.Random(SEMILLA)
    por_clase = defaultdict(list)
    for g in grupos:
        mayoritaria = Counter(clases[i] for i in g).most_common(1)[0][0]
        por_clase[mayoritaria].append(g)
    particion = {}
    for clase in sorted(por_clase):
        gs = por_clase[clase]
        rng.shuffle(gs)
        total = sum(len(g) for g in gs)
        metas = {k: round(total * v) for k, v in PROPORCIONES.items()}
        cuenta = Counter()
        for g in gs:
            destino = "entrenamiento"
            for k in ("prueba", "validacion"):
                if cuenta[k] + len(g) <= metas[k]:
                    destino = k
                    break
            cuenta[destino] += len(g)
            for i in g:
                particion[i] = destino

    grupo_de = {}
    for n, g in enumerate(grupos, start=1):
        for i in g:
            grupo_de[i] = n

    SALIDA.mkdir(parents=True, exist_ok=True)
    with open(SALIDA / f"{nombre}_particion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "archivo", "sha256", "estado", "grupo", "particion"])
        for i, p in enumerate(rutas):
            w.writerow([clases[i], p.name, shas[i], estado[i], grupo_de.get(i, ""), particion.get(i, "")])

    tabla = {}
    for c in sorted(set(clases)):
        fila = {"original": clases.count(c),
                "depurada": sum(1 for i in vivos if clases[i] == c)}
        for k in ("entrenamiento", "validacion", "prueba"):
            fila[k] = sum(1 for i in vivos if clases[i] == c and particion[i] == k)
        tabla[c] = fila
    return {
        "imagenes_originales": len(rutas),
        "grupos_duplicados_exactos": grupos_exactos,
        "grupos_exactos_entre_clases": entre_clases,
        "descartadas_copia_exacta": sum(1 for e in estado if e == "descartada: copia exacta"),
        "descartadas_etiqueta_contradictoria": sum(1 for e in estado if "dos clases" in e),
        "imagenes_depuradas": len(vivos),
        "pares_casi_duplicados": pares,
        "pares_casi_duplicados_entre_clases": pares_entre_clases,
        "grupos_casi_duplicados": len(grupos_casi),
        "imagenes_en_grupos_casi_duplicados": sum(len(g) for g in grupos_casi),
        "por_clase": tabla,
        "totales": {k: sum(f[k] for f in tabla.values())
                    for k in ("original", "depurada", "entrenamiento", "validacion", "prueba")},
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    resumen = {"semilla": SEMILLA, "dhash_bits": LADO_HASH * LADO_HASH, "umbral_hamming": UMBRAL,
               "proporciones": {"entrenamiento": 0.70, **PROPORCIONES}}
    for nombre, carpeta in CONJUNTOS.items():
        print("→", nombre, carpeta)
        resumen[nombre] = auditar(nombre, carpeta)
        print(json.dumps(resumen[nombre], ensure_ascii=False, indent=1))
    with open(SALIDA / "resumen_particiones.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=1)
    print("Listo:", SALIDA)


if __name__ == "__main__":
    main()
