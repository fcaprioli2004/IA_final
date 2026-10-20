"""Normaliza en lote TODAS las huellas autorizadas de data_sheet.

Este archivo NO implementa filtros ni operaciones de normalizacion:
TODAS las funciones de imagen se importan desde normalizacion.py.

Colocar ambos scripts en IA_final/:
    IA_final/
        normalizacion.py                  <-- biblioteca reutilizable
        normalizar_autorizadas.py         <-- procesamiento del dataset
        data_sheet/
            Huellas_1/{Fingerprints,Impression_1,Impression_py,...}
            Huellas_2/...
            Huellas_3/...

Salida plana, sin subcarpetas:
    data_sheet/Huellas_1/Normalizadas/*.png
    data_sheet/Huellas_2/Normalizadas/*.png
    data_sheet/Huellas_3/Normalizadas/*.png
    data_sheet/registro_normalizacion_autorizadas.csv

Uso:
    py normalizar_autorizadas.py
    py normalizar_autorizadas.py --perfil basico
    py normalizar_autorizadas.py --datos "C:/ruta/data_sheet"

No modifica originales. Excluye Normalizadas para evitar recursividad.
No procesa Huellas_intrusas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

# IMPORTANTISIMO: pipeline COMPARTIDO con el futuro reconocimiento.
from normalizacion import (
    PERFIL_PREDETERMINADO,
    ANCHO_PREDETERMINADO,
    ALTO_PREDETERMINADO,
    guardar_png,
    normalizar_huella_con_metricas,
    leer_gris,
)

EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IDENTIDADES = ("Huellas_1", "Huellas_2", "Huellas_3")


def imagenes_de_identidad(carpeta: Path) -> list[Path]:
    """Localiza imágenes, excluyendo las ya procesadas."""
    imagenes = []
    for p in carpeta.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONES:
            continue
        relativa = p.relative_to(carpeta)
        if any(parte.casefold() == "normalizadas" for parte in relativa.parts[:-1]):
            continue
        imagenes.append(p)
    return sorted(imagenes, key=lambda p: str(p).casefold())


def nombre_unico(carpeta_id: Path, origen: Path) -> str:
    """Nombra cada imagen usando su ruta relativa para evitar colisiones."""
    relativa = origen.relative_to(carpeta_id).with_suffix("")
    partes = [re.sub(r"[^a-zA-Z0-9_-]+", "_", p) for p in relativa.parts]
    descripcion = "__".join(partes)
    if len(descripcion) > 115:
        descripcion = descripcion[:115].rstrip("_")
    identificador = hashlib.sha256(relativa.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{descripcion}__{identificador}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normaliza autorizadas usando normalizacion.py")
    parser.add_argument("--perfil", choices=("basico", "suave"), default=PERFIL_PREDETERMINADO)
    parser.add_argument("--ancho", type=int, default=ANCHO_PREDETERMINADO)
    parser.add_argument("--alto", type=int, default=ALTO_PREDETERMINADO)
    parser.add_argument("--datos", type=Path, default=Path(__file__).resolve().parent / "data_sheet")
    args = parser.parse_args()

    if args.ancho < 20 or args.alto < 20:
        parser.error("El ancho y el alto deben ser al menos 20 pixeles")
    raiz = args.datos.resolve()
    if not raiz.is_dir():
        parser.error(f"No existe la carpeta: {raiz}")

    registro = []
    procesadas = 0
    errores = 0
    print(f"Dataset: {raiz}")
    print(f"Perfil: {args.perfil} | Dimensiones: {args.ancho}x{args.alto}")
    print("Se incluyen todas las imagenes de Huellas_1/2/3 excepto Normalizadas.\n")

    for identidad in IDENTIDADES:
        carpeta_id = raiz / identidad
        if not carpeta_id.is_dir():
            print(f"[AVISO] No existe {carpeta_id}")
            continue

        archivos = imagenes_de_identidad(carpeta_id)
        destino_id = carpeta_id / "Normalizadas"
        destino_id.mkdir(parents=True, exist_ok=True)
        correctas_id = 0

        for origen in archivos:
            destino = destino_id / nombre_unico(carpeta_id, origen)
            fila = {
                "identidad": identidad,
                "origen": str(origen.relative_to(raiz)).replace("\\", "/"),
                "destino": str(destino.relative_to(raiz)).replace("\\", "/"),
                "perfil": args.perfil,
            }
            try:
                # LA MISMA funcion publica que utilizara reconocimiento.py
                # para una huella desconocida, sin pipeline duplicado.
                resultado, metricas = normalizar_huella_con_metricas(
                    origen, perfil=args.perfil, ancho=args.ancho, alto=args.alto
                )
                guardar_png(destino, resultado)
                # Solo para registrar las dimensiones originales.
                alto_orig, ancho_orig = leer_gris(origen).shape
                fila.update({
                    "alto_original": alto_orig,
                    "ancho_original": ancho_orig,
                    "alto_normalizado": resultado.shape[0],
                    "ancho_normalizado": resultado.shape[1],
                    **metricas,
                    "estado": "ok",
                    "error": "",
                })
                correctas_id += 1
                procesadas += 1
            except Exception as exc:
                errores += 1
                fila.update({"estado": "error", "error": str(exc)})
                print(f"  [ERROR] {origen}: {exc}")
            registro.append(fila)

        print(f"{identidad}: {correctas_id}/{len(archivos)} imagenes en {destino_id}")

    if registro:
        csv_path = raiz / "registro_normalizacion_autorizadas.csv"
        columnas = [
            "identidad", "origen", "destino", "perfil",
            "alto_original", "ancho_original", "alto_normalizado", "ancho_normalizado",
            "polaridad_invertida", "media_intensidad", "desviacion_intensidad",
            "fraccion_pixeles_oscuros", "nitidez_laplaciano", "estado", "error",
        ]
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columnas)
            writer.writeheader()
            writer.writerows(registro)
        print(f"\nRegistro: {csv_path}")

    print(f"Total normalizadas: {procesadas} | Errores: {errores}")
    if not procesadas:
        print("No se encontraron imagenes para procesar. Verifica la estructura de carpetas.")
    print("Nota: los PNG contienen valores 0..255, no estan estandarizados para K-means.")


if __name__ == "__main__":
    main()
