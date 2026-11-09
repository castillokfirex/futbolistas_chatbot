"""Pruebas unitarias para el módulo almacenamiento.py (Fase 1)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROYECTO_DIR = Path(__file__).resolve().parent.parent
if str(PROYECTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROYECTO_DIR))

import almacenamiento as alm


def test_catalogo_oficial_tiene_20_jugadores_sin_duplicados():
    """Actividad 1: el catálogo oficial tiene los 20 futbolistas exactos y sin IDs repetidos."""
    catalogo = alm.cargar_jugadores(alm.RUTA_JUGADORES)
    assert len(catalogo) == 20
    assert catalogo["j001"]["nombre"] == "Lionel Messi"
    assert catalogo["j002"]["nombre"] == "Cristiano Ronaldo"
    assert catalogo["j012"]["nombre"] == "James Rodríguez"
    assert catalogo["j013"]["nombre"] == "Luis Díaz"
    assert catalogo["j020"]["nombre"] == "Ronaldo Nazário"


def test_cargar_jugadores_detecta_id_duplicado(tmp_path: Path):
    """Actividad 1: si hay identificadores repetidos en jugadores.json, lanza ErrorArchivoDanado."""
    ruta_dup = tmp_path / "jugadores_dup.json"
    datos = {
        "version": 1,
        "jugadores": [
            {"id": "j001", "nombre": "Lionel Messi", "pais": "Argentina"},
            {"id": "j001", "nombre": "Clon", "pais": "Argentina"},
        ],
    }
    ruta_dup.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(alm.ErrorArchivoDanado, match="duplicado"):
        alm.cargar_jugadores(ruta_dup)


def test_semilla_opiniones_cumple_distribucion_4x4():
    """Entregable: opiniones.json inicial tiene 12 registros en 4 jugadores y las 4 categorías."""
    datos = alm.cargar_opiniones(alm.RUTA_OPINIONES)
    opiniones = datos["opiniones"]
    assert len(opiniones) == 12

    jugadores_presentes = {op["jugador_id"] for op in opiniones}
    assert jugadores_presentes == {"j001", "j002", "j012", "j013"}

    conteo = alm.contar_por_sentimiento(opiniones)
    assert conteo == {"positivo": 3, "negativo": 3, "neutro": 3, "mixto": 3}


def test_cargar_opiniones_archivo_inexistente_inicializa_vacio(tmp_path: Path):
    """Actividad 5: si el archivo no existe, devuelve la estructura vacía."""
    ruta_nueva = tmp_path / "no_existe.json"
    estado = alm.cargar_opiniones(ruta_nueva)
    assert estado == {"version": 1, "opiniones": []}
    assert not ruta_nueva.exists()


def test_archivo_danado_no_se_sobrescribe(tmp_path: Path):
    """Caso de aceptación #13: JSON existente inválido lanza error y conserva el archivo intacto."""
    ruta_corrupta = tmp_path / "opiniones.json"
    contenido_corrupto = "{ json malformado sin cierre "
    ruta_corrupta.write_text(contenido_corrupto, encoding="utf-8")

    with pytest.raises(alm.ErrorArchivoDanado):
        alm.cargar_opiniones(ruta_corrupta)

    # Tampoco guardar_opinion debe pisar el archivo corrupto
    analisis_fake = SimpleNamespace(es_pertinente=True, sentimiento="positivo", razon="Gran partido.")
    reg = alm.crear_registro(
        {"id": "j001", "nombre": "Lionel Messi", "pais": "Argentina"},
        "Su visión de juego es extraordinaria",
        analisis_fake,
        "gemini-2.5-flash",
    )
    with pytest.raises(alm.ErrorArchivoDanado):
        alm.guardar_opinion(ruta_corrupta, reg)

    assert ruta_corrupta.read_text(encoding="utf-8") == contenido_corrupto


def test_conservacion_dos_opiniones_mismo_jugador_y_prevencion_duplicados(tmp_path: Path):
    """Casos #9 y #15: dos opiniones independientes se guardan; reintentar el mismo UUID no duplica."""
    ruta = tmp_path / "opiniones.json"
    jugador = {"id": "j001", "nombre": "Lionel Messi", "pais": "Argentina"}
    analisis = SimpleNamespace(es_pertinente=True, sentimiento="positivo", razon="Elogio claro.")

    reg1 = alm.crear_registro(jugador, "Primera opinión excelente sobre su juego", analisis, "gemini-2.5-flash")
    reg2 = alm.crear_registro(jugador, "Segunda opinión excelente sobre su juego", analisis, "gemini-2.5-flash")

    assert alm.guardar_opinion(ruta, reg1) is True
    assert alm.guardar_opinion(ruta, reg2) is True
    # Reintento con el mismo registro_id (idempotencia en reejecución)
    assert alm.guardar_opinion(ruta, reg1) is False

    cargado = alm.cargar_opiniones(ruta)
    assert len(cargado["opiniones"]) == 2


def test_fallo_de_escritura_limpia_temporal_y_lanza_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Caso #14: ante fallo en os.replace se lanza ErrorEscritura y no quedan archivos .tmp."""
    ruta = tmp_path / "opiniones.json"
    jugador = {"id": "j002", "nombre": "Cristiano Ronaldo", "pais": "Portugal"}
    analisis = SimpleNamespace(es_pertinente=True, sentimiento="positivo", razon="Gran goleador.")
    reg = alm.crear_registro(jugador, "Un goleador histórico impresionante", analisis, "gemini-2.5-flash")

    def replace_fallido(src, dst):
        raise PermissionError("Permiso denegado simulado")

    monkeypatch.setattr(os, "replace", replace_fallido)

    with pytest.raises(alm.ErrorEscritura):
        alm.guardar_opinion(ruta, reg)

    temporales = list(tmp_path.glob("*.tmp"))
    assert temporales == []
    assert not ruta.exists()


def test_consultar_opiniones_filtra_sin_llamar_modelo():
    """Actividad 6: filtrado por jugador, por sentimiento y combinado."""
    datos = alm.cargar_opiniones(alm.RUTA_OPINIONES)

    todos = alm.consultar_opiniones(datos, jugador_id="Todos", sentimiento="Todos")
    assert len(todos) == 12

    solo_messi = alm.consultar_opiniones(datos, jugador_id="j001", sentimiento="Todos")
    assert len(solo_messi) == 3
    assert all(r["jugador_id"] == "j001" for r in solo_messi)

    solo_positivos = alm.consultar_opiniones(datos, jugador_id=None, sentimiento="positivo")
    assert len(solo_positivos) == 3
    assert all(r["sentimiento"] == "positivo" for r in solo_positivos)

    diaz_neutro = alm.consultar_opiniones(datos, jugador_id="j013", sentimiento="neutro")
    assert len(diaz_neutro) == 1
    assert diaz_neutro[0]["jugador_nombre"] == "Luis Díaz"
