"""Módulo de almacenamiento, validación y consulta en disco para Tribuna Global.

Gestiona la lectura del catálogo de jugadores, la persistencia de opiniones
mediante escritura atómica con archivo temporal y el filtrado en memoria
sin realizar llamadas al modelo de lenguaje.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

BASE_DIR: Path = Path(__file__).resolve().parent
RUTA_JUGADORES: Path = BASE_DIR / "datos" / "jugadores.json"
RUTA_OPINIONES: Path = BASE_DIR / "datos" / "opiniones.json"

SENTIMIENTOS: tuple[str, ...] = ("positivo", "negativo", "neutro", "mixto")
CategoriaSentimiento = Literal["positivo", "negativo", "neutro", "mixto"]
VERSION_ESQUEMA: int = 1
VERSION_PROMPT_DEFECTO: str = "1.0"


class ErrorAlmacenamiento(Exception):
    """Excepción base para errores de lectura o escritura de datos."""


class ErrorArchivoDanado(ErrorAlmacenamiento):
    """El archivo JSON está corrupto o no cumple el esquema esperado."""


class ErrorEscritura(ErrorAlmacenamiento):
    """No fue posible escribir o reemplazar el archivo de destino."""


class Jugador(BaseModel):
    """Contrato de cada futbolista dentro del catálogo."""

    id: str = Field(min_length=1)
    nombre: str = Field(min_length=1)
    pais: str = Field(min_length=1)


class RegistroOpinion(BaseModel):
    """Contrato de un registro persistido en opiniones.json."""

    id: str = Field(min_length=1)
    fecha_utc: str = Field(min_length=1)
    jugador_id: str = Field(min_length=1)
    jugador_nombre: str = Field(min_length=1)
    opinion_original: str = Field(min_length=5, max_length=1000)
    sentimiento: CategoriaSentimiento
    razon: str = Field(min_length=1, max_length=300)
    modelo: str = Field(min_length=1)
    version_prompt: str = Field(min_length=1)


def cargar_jugadores(ruta: Path | str = RUTA_JUGADORES) -> dict[str, dict[str, str]]:
    """Lee y valida el catálogo de futbolistas.

    Comprueba que el archivo exista, tenga JSON válido, versión entera,
    lista no vacía de jugadores y ausencia de identificadores repetidos.
    Devuelve un diccionario indexado por ``id`` del jugador.
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        raise ErrorArchivoDanado(f"No se encontró el catálogo de jugadores en: {ruta_path}")

    try:
        contenido = ruta_path.read_text(encoding="utf-8")
        datos = json.loads(contenido)
    except json.JSONDecodeError as exc:
        raise ErrorArchivoDanado(
            f"El archivo de jugadores contiene JSON inválido ({ruta_path.name}): {exc}"
        ) from exc
    except OSError as exc:
        raise ErrorAlmacenamiento(f"No se pudo leer el catálogo de jugadores: {exc}") from exc

    if not isinstance(datos, dict):
        raise ErrorArchivoDanado("El catálogo de jugadores debe ser un objeto JSON.")

    if not isinstance(datos.get("version"), int):
        raise ErrorArchivoDanado("El catálogo de jugadores carece de un campo 'version' entero.")

    lista = datos.get("jugadores")
    if not isinstance(lista, list) or len(lista) == 0:
        raise ErrorArchivoDanado("El catálogo debe incluir una lista no vacía en 'jugadores'.")

    catalogo: dict[str, dict[str, str]] = {}
    for indice, item in enumerate(lista):
        if not isinstance(item, dict):
            raise ErrorArchivoDanado(f"El elemento #{indice} del catálogo no es un objeto.")
        try:
            jugador = Jugador(**item)
        except ValidationError as exc:
            raise ErrorArchivoDanado(
                f"Jugador inválido en la posición {indice} del catálogo: {exc}"
            ) from exc

        if jugador.id in catalogo:
            raise ErrorArchivoDanado(
                f"Identificador de jugador duplicado en el catálogo: '{jugador.id}'."
            )
        catalogo[jugador.id] = jugador.model_dump()

    return catalogo


def cargar_opiniones(ruta: Path | str = RUTA_OPINIONES) -> dict[str, Any]:
    """Lee el diccionario persistente de opiniones.

    Si el archivo no existe, devuelve la estructura inicial vacía sin sobrescribir.
    Si el archivo existe pero contiene JSON inválido o un esquema inesperado,
    lanza ``ErrorArchivoDanado`` conservando el archivo intacto en disco.
    """
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        return {"version": VERSION_ESQUEMA, "opiniones": []}

    try:
        contenido = ruta_path.read_text(encoding="utf-8")
        datos = json.loads(contenido)
    except json.JSONDecodeError as exc:
        raise ErrorArchivoDanado(
            f"El archivo '{ruta_path.name}' contiene JSON inválido y fue conservado sin cambios."
        ) from exc
    except OSError as exc:
        raise ErrorAlmacenamiento(f"No se pudo leer '{ruta_path.name}': {exc}") from exc

    if not isinstance(datos, dict):
        raise ErrorArchivoDanado(
            f"Estructura raíz inválida en '{ruta_path.name}': se esperaba un objeto JSON."
        )

    if not isinstance(datos.get("version"), int) or not isinstance(datos.get("opiniones"), list):
        raise ErrorArchivoDanado(
            f"Esquema inesperado en '{ruta_path.name}': se requieren 'version' (int) y 'opiniones' (list)."
        )

    opiniones_validadas: list[dict[str, Any]] = []
    for indice, item in enumerate(datos["opiniones"]):
        if not isinstance(item, dict):
            raise ErrorArchivoDanado(
                f"El registro #{indice} en '{ruta_path.name}' no es un objeto JSON válido."
            )
        try:
            reg = RegistroOpinion(**item)
        except ValidationError as exc:
            raise ErrorArchivoDanado(
                f"El registro #{indice} en '{ruta_path.name}' no cumple el esquema: {exc}"
            ) from exc
        opiniones_validadas.append(reg.model_dump())

    return {"version": datos["version"], "opiniones": opiniones_validadas}


def crear_registro(
    jugador: dict[str, str],
    texto: str,
    analisis: Any,
    modelo: str,
    *,
    registro_id: str | None = None,
    version_prompt: str = VERSION_PROMPT_DEFECTO,
) -> dict[str, Any]:
    """Construye un registro listo para persistir a partir del análisis validado.

    El programa genera el UUID, la fecha UTC, la identidad del jugador y los
    metadatos del modelo; el modelo de IA únicamente aporta ``sentimiento`` y ``razon``.
    """
    if not isinstance(jugador, dict) or not jugador.get("id") or not jugador.get("nombre"):
        raise ValueError("El diccionario del jugador debe incluir 'id' y 'nombre'.")

    texto_limpio = texto.strip() if isinstance(texto, str) else ""

    if isinstance(analisis, dict):
        es_pertinente = analisis.get("es_pertinente", True)
        sentimiento = analisis.get("sentimiento")
        razon = analisis.get("razon")
    else:
        es_pertinente = getattr(analisis, "es_pertinente", True)
        sentimiento = getattr(analisis, "sentimiento", None)
        razon = getattr(analisis, "razon", None)

    if not es_pertinente or sentimiento is None:
        raise ValueError("No se puede crear un registro para una opinión no pertinente o sin sentimiento.")

    identificador = registro_id or str(uuid.uuid4())
    fecha_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    registro = RegistroOpinion(
        id=identificador,
        fecha_utc=fecha_utc,
        jugador_id=jugador["id"],
        jugador_nombre=jugador["nombre"],
        opinion_original=texto_limpio,
        sentimiento=sentimiento,
        razon=razon.strip() if isinstance(razon, str) else "",
        modelo=modelo.strip() if isinstance(modelo, str) and modelo.strip() else "desconocido",
        version_prompt=version_prompt,
    )
    return registro.model_dump()


def _escribir_json_atomico(ruta: Path, datos: dict[str, Any]) -> None:
    """Escribe ``datos`` en un archivo temporal del mismo directorio y reemplaza el destino."""
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ErrorEscritura(f"No se pudo preparar el directorio '{ruta.parent}': {exc}") from exc

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=ruta.parent,
            prefix=f".{ruta.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(datos, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(tmp_path, ruta)
    except Exception as exc:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise ErrorEscritura(f"Fallo al guardar el archivo '{ruta.name}': {exc}") from exc


def guardar_opinion(ruta: Path | str, registro: dict[str, Any]) -> bool:
    """Añade un registro al archivo de opiniones sin perder los anteriores.

    - Valida el registro contra ``RegistroOpinion`` antes de tocar el disco.
    - Lee el estado actual (si está corrupto propaga ``ErrorArchivoDanado`` sin sobrescribirlo).
    - Evita duplicados por ``id`` en caso de reintento o reejecución (devuelve ``False`` si ya existía).
    - Escribe de forma atómica y devuelve ``True`` cuando el registro queda persistido.
    """
    ruta_path = Path(ruta)
    try:
        registro_validado = RegistroOpinion(**registro).model_dump()
    except ValidationError as exc:
        raise ValueError(f"El registro a guardar no es válido: {exc}") from exc

    estado = cargar_opiniones(ruta_path)
    ids_existentes = {op["id"] for op in estado["opiniones"]}
    if registro_validado["id"] in ids_existentes:
        return False

    estado["opiniones"].append(registro_validado)
    _escribir_json_atomico(ruta_path, estado)
    return True


def consultar_opiniones(
    registros: list[dict[str, Any]] | dict[str, Any],
    jugador_id: str | None = None,
    sentimiento: str | None = None,
) -> list[dict[str, Any]]:
    """Filtra registros por jugador y/o sentimiento sin consultar al modelo.

    Acepta tanto la lista de registros como el diccionario completo ``{"version": ..., "opiniones": [...]}``.
    Los valores ``None``, ``""`` o ``"Todos"`` desactivan el filtro correspondiente.
    Devuelve una nueva lista ordenada de más reciente a más antigua por ``fecha_utc``.
    """
    if isinstance(registros, dict):
        lista = list(registros.get("opiniones", []))
    elif isinstance(registros, list):
        lista = list(registros)
    else:
        raise TypeError("Se esperaba una lista de registros o el diccionario de opiniones.")

    filtrar_jugador = jugador_id not in (None, "", "Todos", "todos")
    filtrar_sentimiento = sentimiento not in (None, "", "Todos", "todos")

    resultado: list[dict[str, Any]] = []
    for reg in lista:
        if filtrar_jugador and reg.get("jugador_id") != jugador_id:
            continue
        if filtrar_sentimiento and reg.get("sentimiento") != sentimiento:
            continue
        resultado.append(reg)

    resultado.sort(key=lambda r: r.get("fecha_utc", ""), reverse=True)
    return resultado


def contar_por_sentimiento(registros: list[dict[str, Any]]) -> dict[str, int]:
    """Cuenta cuántos registros hay en cada una de las 4 categorías oficiales."""
    conteo: dict[str, int] = {categoria: 0 for categoria in SENTIMIENTOS}
    for reg in registros:
        cat = reg.get("sentimiento")
        if cat in conteo:
            conteo[cat] += 1
    return conteo
