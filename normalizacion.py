"""Normalizacion de huellas reutilizable para el proyecto IA_final.

Este archivo es la UNICA fuente de las funciones de preprocesamiento.
El procesamiento por lotes y el futuro reconocedor deben importarlas desde aqui.

Pipeline por defecto (identico al usado antes en normalizar_autorizadas.py):
    1) convertir a escala de grises uint8
    2) corregir polaridad cuando las esquinas son oscuras
    3) filtro de mediana 3x3 + CLAHE si perfil='suave'
    4) encajar la imagen sin deformar en lienzo blanco de 275x400

No se recorta ROI ni se modifica la orientacion intencionalmente.
No identifica personas ni contiene el modelo K-means.

Uso desde otro archivo:
    from normalizacion import normalizar_huella
    imagen = normalizar_huella('imagen.png')
    imagen_float = normalizar_huella('imagen.png', valores_01=True)

Dependencias: py -m pip install opencv-python numpy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# CONFIGURACION COMPARTIDA: cambiar aqui si se decide otra configuracion.
PERFIL_PREDETERMINADO = "suave"  # "basico" o "suave"
ANCHO_PREDETERMINADO = 275
ALTO_PREDETERMINADO = 400


def convertir_a_gris_uint8(matriz: np.ndarray) -> np.ndarray:
    """Convierte imagen gris/BGR/BGRA a uint8. No normaliza la huella aun.

    BGR/BGRA es el orden de canales utilizado por OpenCV. Un canal alfa
    transparente se compone con un fondo blanco.
    """
    if not isinstance(matriz, np.ndarray) or matriz.size == 0:
        raise ValueError("La imagen debe ser una matriz NumPy no vacia")

    img = matriz
    if img.ndim == 3:
        if img.shape[2] == 4:
            if img.dtype not in (np.uint8, np.uint16):
                raise ValueError("Para imagenes BGRA se admite uint8 o uint16")
            gris = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            maximo = float(np.iinfo(img.dtype).max)
            alfa = img[:, :, 3].astype(np.float32) / maximo
            img = gris.astype(np.float32) * alfa + maximo * (1.0 - alfa)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("La imagen debe tener uno, tres o cuatro canales")

    if img.dtype == np.uint16:
        img = (img / 257.0).round().astype(np.uint8)
    elif img.dtype == np.uint8:
        pass
    elif np.issubdtype(img.dtype, np.floating):
        valores = img.astype(np.float32)
        finitos = valores[np.isfinite(valores)]
        if not finitos.size:
            raise ValueError("La imagen no contiene valores finitos")
        if finitos.min() >= 0 and finitos.max() <= 1:
            valores *= 255.0
        img = np.clip(np.nan_to_num(valores, nan=255.0), 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Tipo de datos de imagen no soportado: {img.dtype}")

    if img.ndim != 2 or min(img.shape) < 10:
        raise ValueError(f"Dimensiones de imagen no validas: {img.shape}")
    return img


def leer_gris(ruta: str | Path) -> np.ndarray:
    """Lee ruta Windows (incluso con espacios/acentos) y devuelve gris uint8."""
    ruta = Path(ruta).expanduser()
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_UNCHANGED)
    if imagen is None:
        raise ValueError(f"OpenCV no pudo decodificar la imagen: {ruta}")
    return convertir_a_gris_uint8(imagen)


def corregir_polaridad(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Deja las crestas oscuras sobre fondo claro, segun mediana de esquinas."""
    h, w = img.shape
    dy, dx = max(1, h // 12), max(1, w // 12)
    esquinas = np.concatenate((
        img[:dy, :dx].ravel(), img[:dy, -dx:].ravel(),
        img[-dy:, :dx].ravel(), img[-dy:, -dx:].ravel(),
    ))
    invertir = bool(np.median(esquinas) < 110)
    return (255 - img if invertir else img.copy()), invertir


def encajar_sin_deformar(img: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Redimensiona preservando relacion de aspecto; rellena con blanco."""
    h, w = img.shape
    factor = min(ancho / w, alto / h)
    nuevo_w = max(1, round(w * factor))
    nuevo_h = max(1, round(h * factor))
    interpolacion = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    escalada = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=interpolacion)
    salida = np.full((alto, ancho), 255, dtype=np.uint8)
    ox = (ancho - nuevo_w) // 2
    oy = (alto - nuevo_h) // 2
    salida[oy:oy+nuevo_h, ox:ox+nuevo_w] = escalada
    return salida


def normalizar(
    imagen_gris: np.ndarray,
    perfil: str = PERFIL_PREDETERMINADO,
    ancho: int = ANCHO_PREDETERMINADO,
    alto: int = ALTO_PREDETERMINADO,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aplica el MISMO pipeline que el normalizador de autorizadas anterior.

    Recibe escala de grises uint8; devuelve imagen uint8 y metricas.
    """
    if perfil not in ("basico", "suave"):
        raise ValueError("El perfil debe ser 'basico' o 'suave'")
    if ancho < 20 or alto < 20:
        raise ValueError("Ancho y alto deben ser al menos 20 pixeles")
    if imagen_gris.ndim != 2 or imagen_gris.dtype != np.uint8:
        raise ValueError("normalizar() necesita una imagen gris uint8")

    img, invertida = corregir_polaridad(imagen_gris)

    if perfil == "suave":
        img = cv2.medianBlur(img, 3)
        clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        img = clahe.apply(img)

    # Se conserva la variabilidad en giro, traslado y recortes.
    img = encajar_sin_deformar(img, ancho, alto)
    metricas = {
        "polaridad_invertida": "si" if invertida else "no",
        "media_intensidad": round(float(img.mean()), 3),
        "desviacion_intensidad": round(float(img.std()), 3),
        "fraccion_pixeles_oscuros": round(float((img < 200).mean()), 5),
        "nitidez_laplaciano": round(float(cv2.Laplacian(img, cv2.CV_64F).var()), 3),
    }
    return img, metricas


def normalizar_huella_con_metricas(
    entrada: str | Path | np.ndarray,
    *,
    perfil: str = PERFIL_PREDETERMINADO,
    ancho: int = ANCHO_PREDETERMINADO,
    alto: int = ALTO_PREDETERMINADO,
    valores_01: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Acepta una ruta o imagen OpenCV y devuelve la imagen + metricas.

    La misma funcion se usa para la base autorizada y para nuevas muestras.
    """
    if isinstance(entrada, (str, Path)):
        gris = leer_gris(entrada)
    else:
        gris = convertir_a_gris_uint8(entrada)
    resultado, metricas = normalizar(gris, perfil, ancho, alto)
    if valores_01:
        return resultado.astype(np.float32) / 255.0, metricas
    return resultado, metricas


def normalizar_huella(
    entrada: str | Path | np.ndarray,
    *,
    perfil: str = PERFIL_PREDETERMINADO,
    ancho: int = ANCHO_PREDETERMINADO,
    alto: int = ALTO_PREDETERMINADO,
    valores_01: bool = False,
) -> np.ndarray:
    """API principal: devuelve solo la huella ya normalizada, en memoria.

    No clasifica ni comprueba identidades: eso sera responsabilidad de otro modulo.
    """
    resultado, _ = normalizar_huella_con_metricas(
        entrada, perfil=perfil, ancho=ancho, alto=alto, valores_01=valores_01
    )
    return resultado


def guardar_png(destino: str | Path, imagen: np.ndarray) -> None:
    """Guarda imagen gris uint8 en PNG, sin modificar la entrada."""
    destino = Path(destino)
    if imagen.ndim != 2 or imagen.dtype != np.uint8:
        raise ValueError("Para guardar PNG se requiere imagen gris uint8")
    destino.parent.mkdir(parents=True, exist_ok=True)
    correcto, buf = cv2.imencode(".png", imagen)
    if not correcto:
        raise IOError("Fallo la codificacion PNG")
    buf.tofile(str(destino))
