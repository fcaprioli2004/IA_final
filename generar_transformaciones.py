"""Aumento reproducible de impresiones sinteticas generadas por Anguli.

Ubicacion recomendada: IA_final/generar_transformaciones.py
Datos esperados:       IA_final/data_sheet/Huellas_1, Huellas_2, Huellas_3

Acepta DOS organizaciones de entrada:
  NUEVA: Huellas_N/Impression_1/1.png ... 10.png
  ANTIGUA: Huellas_N/Impression_1/fp_1/1.png ... Impression_10/fp_1/1.png

No modifica las imagenes originales. Genera nuevas en
Huellas_N/Impression_py/generacion_01 y un CSV en data_sheet/.
"""

from pathlib import Path
import csv
import json

import cv2
import numpy as np

# ========== PARAMETROS EDITABLES ==========
DATA_DIR = Path(__file__).resolve().parent / "data_sheet"
IDENTIDADES = (1, 2, 3)
NOMBRE_GENERACION = "generacion_01"   # cambiar a generacion_02 si repetis una prueba
SEMILLA = 2026
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

rng = np.random.default_rng(SEMILLA)


def leer_gris(ruta: Path) -> np.ndarray:
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise RuntimeError(f"No se pudo abrir: {ruta}")
    return imagen


def guardar_png(ruta: Path, imagen: np.ndarray) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(ruta), imagen):
        raise RuntimeError(f"No se pudo guardar: {ruta}")


def transformar_afin(imagen: np.ndarray, angulo=0.0, escala=1.0, tx=0, ty=0):
    """Rotacion, escalado y traslacion: misma geometria global."""
    alto, ancho = imagen.shape
    matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, escala)
    matriz[:, 2] += [tx, ty]
    return cv2.warpAffine(imagen, matriz, (ancho, alto),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def deformacion_elastica(imagen: np.ndarray, rng, amplitud=10):
    """Desplazamiento espacial suave y NO uniforme (en pixeles)."""
    alto, ancho = imagen.shape
    # Campo aleatorio a baja resolucion: produce variaciones espaciales suaves.
    ny, nx = max(3, alto // 50), max(3, ancho // 50)

    def campo():
        base = rng.normal(0, 1, (ny, nx)).astype(np.float32)
        suave = cv2.resize(base, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        suave = cv2.GaussianBlur(suave, (0, 0), sigmaX=12)
        maximo = max(float(np.max(np.abs(suave))), 1e-6)
        return amplitud * suave / maximo

    dx, dy = campo(), campo()
    x, y = np.meshgrid(np.arange(ancho, dtype=np.float32),
                       np.arange(alto, dtype=np.float32))
    return cv2.remap(imagen, x + dx, y + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def presion_simulada(imagen: np.ndarray):
    """En imagen con crestas OSCURAS, erosion engrosa lineas oscuras."""
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(imagen, nucleo, iterations=1)


def ruido_gaussiano(imagen: np.ndarray, rng, sigma=20):
    ruido = rng.normal(0, sigma, imagen.shape)
    return np.clip(imagen.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def ruido_sal_pimienta(imagen: np.ndarray, rng, proporcion=0.4):
    salida = imagen.copy()
    cantidad = max(2, round(imagen.size * proporcion))
    indices = rng.choice(imagen.size, size=cantidad, replace=False)
    plano = salida.reshape(-1)
    mitad = cantidad // 2
    plano[indices[:mitad]] = 0
    plano[indices[mitad:]] = 255
    return salida


def recorte_parcial(imagen: np.ndarray, lado="derecha", fraccion=0.2):
    """Oculta un porcentaje de la zona que contiene crestas, no del margen blanco."""
    salida = imagen.copy()
    yy, xx = np.where(imagen < 220)
    if xx.size == 0:
        return salida
    izquierda, derecha = int(xx.min()), int(xx.max()) + 1
    arriba, abajo = int(yy.min()), int(yy.max()) + 1
    if lado == "derecha":
        desde = derecha - max(1, round((derecha - izquierda) * fraccion))
        salida[:, desde:derecha] = 255
    elif lado == "izquierda":
        hasta = izquierda + max(1, round((derecha - izquierda) * fraccion))
        salida[:, izquierda:hasta] = 255
    elif lado == "arriba":
        hasta = arriba + max(1, round((abajo - arriba) * fraccion))
        salida[arriba:hasta, :] = 255
    else:
        desde = abajo - max(1, round((abajo - arriba) * fraccion))
        salida[desde:abajo, :] = 255
    return salida


def contraste(imagen: np.ndarray, factor=0.3):
    """factor < 1 reduce contraste alrededor de gris medio."""
    datos = factor * (imagen.astype(np.float32) - 127.5) + 127.5
    return np.clip(datos, 0, 255).astype(np.uint8)


def imagenes_fuente(carpeta: Path):
    """Detecta automaticamente la estructura nueva o la estructura Anguli anterior."""
    impresion_unica = carpeta / "Impression_1"
    if impresion_unica.is_dir():
        directas = [p for p in impresion_unica.iterdir()
                    if p.is_file() and p.suffix.lower() in EXTENSIONES]
        if directas:
            directas.sort(key=lambda p: (0, int(p.stem)) if p.stem.isdigit()
                          else (1, p.stem))
            return [(int(p.stem) if p.stem.isdigit() else i, p)
                    for i, p in enumerate(directas, 1)], "nueva"

    antiguas = []
    for numero in range(1, 11):
        carpeta_imp = carpeta / f"Impression_{numero}"
        if carpeta_imp.is_dir():
            archivos = sorted(p for p in carpeta_imp.rglob("*")
                              if p.is_file() and p.suffix.lower() in EXTENSIONES)
            if len(archivos) != 1:
                raise RuntimeError(f"Se esperaba 1 imagen en {carpeta_imp}; hay {len(archivos)}")
            antiguas.append((numero, archivos[0]))
    return antiguas, "antigua"


def ejecutar():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"No encuentro la carpeta: {DATA_DIR}. "
                                "Guarda el script dentro de IA_final, al lado de data_sheet.")

    registro = DATA_DIR / f"registro_{NOMBRE_GENERACION}.csv"
    salidas = [DATA_DIR / f"Huellas_{n}" / "Impression_py" / NOMBRE_GENERACION
               for n in IDENTIDADES]
    # Evitar borrar/sobreescribir ensayos anteriores.
    if registro.exists() or any(ruta.exists() for ruta in salidas):
        raise FileExistsError(
            f"La {NOMBRE_GENERACION} ya existe. Cambia NOMBRE_GENERACION "
            "por 'generacion_02' para hacer otra prueba, sin borrar resultados anteriores.")

    todos = []
    total_fuentes = 0
    for numero_identidad, destino in zip(IDENTIDADES, salidas):
        origen = DATA_DIR / f"Huellas_{numero_identidad}"
        fuentes, organizacion = imagenes_fuente(origen)
        if not fuentes:
            raise FileNotFoundError(f"No hay imagenes de impresiones en {origen}")
        print(f"Huellas_{numero_identidad}: {len(fuentes)} imagenes, estructura {organizacion}")
        total_fuentes += len(fuentes)

        for numero_impresion, entrada in fuentes:
            imagen = leer_gris(entrada)
            stem = f"imp_{numero_impresion:02d}"
            angulo = int(rng.choice([-20, -10, 10, 20]))
            tx, ty = int(rng.integers(-15, 15)), int(rng.integers(-15, 15))
            escala = float(rng.choice([0.85, 1.15]))
            lado = str(rng.choice(["izquierda", "derecha", "arriba", "abajo"]))

            variantes = [
                ("original", imagen.copy(), {}),
                ("rotacion", transformar_afin(imagen, angulo=angulo), {"angulo_grados": angulo}),
                ("traslacion", transformar_afin(imagen, tx=tx, ty=ty), {"tx": tx, "ty": ty}),
                ("escalado", transformar_afin(imagen, escala=escala), {"factor": escala}),
                ("elastica", deformacion_elastica(imagen, rng, 5), {"amplitud_px": 5}),
                ("presion", presion_simulada(imagen), {"metodo": "erosion_3x3"}),
                ("gaussiano", ruido_gaussiano(imagen, rng, sigma=20), {"sigma": 20}),
                ("sal_pimienta", ruido_sal_pimienta(imagen, rng, 0.05), {"proporcion": 0.05}),
                ("recorte", recorte_parcial(imagen, lado=lado, fraccion=0.2),
                 {"lado": lado, "fraccion_roi": 0.2}),
                ("contraste", contraste(imagen, factor=0.5), {"factor": 0.5}),
            ]

            for tipo, resultado, parametros in variantes:
                salida = destino / tipo / f"{stem}_{tipo}.png"
                guardar_png(salida, resultado)
                todos.append({
                    "identidad": f"Huellas_{numero_identidad}",
                    "impresion": numero_impresion,
                    "grupo_origen": f"H{numero_identidad}_I{numero_impresion:02d}",
                    "transformacion": tipo,
                    "parametros": json.dumps(parametros, ensure_ascii=False),
                    "imagen_origen": str(entrada.relative_to(DATA_DIR)),
                    "imagen_generada": str(salida.relative_to(DATA_DIR)),
                })

    with registro.open("w", newline="", encoding="utf-8-sig") as archivo:
        campos = ["identidad", "impresion", "grupo_origen", "transformacion",
                  "parametros", "imagen_origen", "imagen_generada"]
        escritor = csv.DictWriter(archivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()
        escritor.writerows(todos)

    print(f"\nTERMINADO: {total_fuentes} impresiones de entrada")
    print(f"Generadas: {len(todos)} imagenes (incluyen una copia original por impresion)")
    print(f"Registro: {registro}")


if __name__ == "__main__":
    ejecutar()
