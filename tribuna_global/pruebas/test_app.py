"""Pruebas de integración de la interfaz Streamlit (app.py) con AppTest (Fases 3 y 4)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROYECTO_DIR = Path(__file__).resolve().parent.parent
if str(PROYECTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROYECTO_DIR))

import almacenamiento as alm
import analisis as an

RUTA_APP = str(PROYECTO_DIR / "app.py")


@pytest.fixture()
def opiniones_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Crea una copia aislada de opiniones.json para no mutar el archivo semilla en las pruebas."""
    destino = tmp_path / "opiniones.json"
    shutil.copyfile(alm.RUTA_OPINIONES, destino)
    monkeypatch.setattr(alm, "RUTA_OPINIONES", destino)
    return destino


def test_app_emitir_opinion_cambio_jugador_y_persistencia_tras_reinicio(
    opiniones_temp: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Casos #8, #10 y #11: emitir opinión de Messi y luego de Luis Díaz, limpiar chat y verificar tras reinicio."""
    respuestas = [
        an.AnalisisOpinion(
            es_pertinente=True,
            sentimiento="positivo",
            razon="El comentario destaca positivamente la visión de juego de Messi.",
        ),
        an.AnalisisOpinion(
            es_pertinente=True,
            sentimiento="mixto",
            razon="Valora el regate de Luis Díaz pero señala falta de definición.",
        ),
    ]

    def analizar_falso(jugador, texto, **_kwargs):
        return respuestas.pop(0)

    monkeypatch.setattr(an, "analizar_opinion", analizar_falso)

    at = AppTest.from_file(RUTA_APP)
    at.run(timeout=20)

    # 1. Opinión para Lionel Messi (j001 por defecto)
    at.chat_input(key="chat_input_opinion").set_value(
        "Su visión de juego es extraordinaria"
    ).run(timeout=20)
    assert len(at.success) == 1

    # 2. Cambiar jugador a Luis Díaz (j013) y enviar segunda opinión
    at.sidebar.selectbox(key="selectbox_jugador_id").select("j013").run(timeout=20)
    at.chat_input(key="chat_input_opinion").set_value(
        "Su regate es tremendo, pero le faltó puntería en el área"
    ).run(timeout=20)
    assert len(at.success) == 1

    # Verificar que en disco hay 12 + 2 = 14 registros con sus jugadores independientes
    en_disco = alm.cargar_opiniones(opiniones_temp)["opiniones"]
    assert len(en_disco) == 14
    nuevos = en_disco[-2:]
    assert nuevos[0]["jugador_id"] == "j001"
    assert nuevos[1]["jugador_id"] == "j013"

    # 3. Caso #11: Botón «Nueva conversación» vacía el chat de la sesión y conserva el archivo
    at.button(key="btn_nueva_conversacion").click().run(timeout=20)
    assert len(at.session_state["historial_opiniones"]) == 0
    assert len(alm.cargar_opiniones(opiniones_temp)["opiniones"]) == 14

    # 4. Caso #8: Simular reinicio de la aplicación con una nueva instancia de AppTest
    at_reiniciado = AppTest.from_file(RUTA_APP)
    at_reiniciado.run(timeout=20)
    at_reiniciado.sidebar.radio(key="radio_seccion_activa").set_value(
        "Consultar opiniones"
    ).run(timeout=20)

    assert at_reiniciado.metric[0].value == "14"


def test_app_consultar_opiniones_filtros_sin_llamar_al_modelo(
    opiniones_temp: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Actividad 6 y Caso #15: cambiar filtros actualiza métricas sin invocar al clasificador ni duplicar."""
    llamadas_llm = []

    def prohibido_llamar_llm(*_args, **_kwargs):
        llamadas_llm.append(1)
        raise AssertionError("Consultar opiniones NO debe llamar al modelo")

    monkeypatch.setattr(an, "analizar_opinion", prohibido_llamar_llm)

    at = AppTest.from_file(RUTA_APP)
    at.run(timeout=20)

    at.sidebar.radio(key="radio_seccion_activa").set_value("Consultar opiniones").run(
        timeout=20
    )
    assert at.metric[0].value == "12"

    # Filtrar por Cristiano Ronaldo (j002 -> 3 registros en la semilla)
    at.selectbox(key="filtro_consulta_jugador").select("j002").run(timeout=20)
    assert at.metric[0].value == "3"

    # Filtrar además por sentimiento positivo (1 registro)
    at.selectbox(key="filtro_consulta_sentimiento").select("positivo").run(timeout=20)
    assert at.metric[0].value == "1"

    assert llamadas_llm == []
    assert len(alm.cargar_opiniones(opiniones_temp)["opiniones"]) == 12


def test_app_manejo_errores_no_confirma_guardado_y_conserva_opinion(
    opiniones_temp: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Casos #6, #12 y #14: mensaje no pertinente, fallo de API y fallo de escritura."""
    # Caso #6: No pertinente -> no se guarda
    def simular_no_pertinente(_jugador, _texto, **_kwargs):
        raise an.OpinionNoPertinente("La capital de Francia no es una opinión deportiva.")

    monkeypatch.setattr(an, "analizar_opinion", simular_no_pertinente)
    at = AppTest.from_file(RUTA_APP)
    at.run(timeout=20)
    at.chat_input(key="chat_input_opinion").set_value(
        "La capital de Francia es París"
    ).run(timeout=20)

    assert len(at.success) == 0
    assert len(at.warning) >= 1
    assert len(alm.cargar_opiniones(opiniones_temp)["opiniones"]) == 12

    # Caso #12: Fallo de API -> muestra error, no guarda, conserva en envio_pendiente
    def simular_fallo_api(_jugador, _texto, **_kwargs):
        raise an.ErrorClasificacion("Servicio de IA temporalmente no disponible.")

    monkeypatch.setattr(an, "analizar_opinion", simular_fallo_api)
    at.chat_input(key="chat_input_opinion").set_value(
        "Su visión de juego es extraordinaria"
    ).run(timeout=20)

    assert len(at.success) == 0
    assert len(at.error) == 1
    assert at.session_state["envio_pendiente"] is not None
    assert len(alm.cargar_opiniones(opiniones_temp)["opiniones"]) == 12

    # Caso #14: Clasificación exitosa pero fallo de escritura en disco -> no aparece st.success
    monkeypatch.setattr(
        an,
        "analizar_opinion",
        lambda _j, _t, **_k: an.AnalisisOpinion(
            es_pertinente=True,
            sentimiento="positivo",
            razon="Elogio claro.",
        ),
    )

    def guardar_fallido(_ruta, _reg):
        raise alm.ErrorEscritura("Destino sin permisos de escritura.")

    monkeypatch.setattr(alm, "guardar_opinion", guardar_fallido)
    at.chat_input(key="chat_input_opinion").set_value(
        "Su visión de juego es extraordinaria"
    ).run(timeout=20)

    assert len(at.success) == 0
    assert len(at.error) == 1
    assert at.session_state["envio_pendiente"]["etapa"] == "almacenamiento"


def test_app_seccion_debate_streaming_sin_tocar_opiniones(
    opiniones_temp: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verifica la pestaña 'Analista y Debate IA' usando streaming sin escribir en opiniones.json."""
    def stream_falso(pregunta: str, historial=None, **_kwargs):
        yield "**Cristiano Ronaldo** es el mejor jugador del mundo.\n\n"
        yield "**1. Títulos y Palmarés**\n5 Champions League.\n"
        yield "**2. Importancia y Relevancia Histórica**\nLeyenda absoluta."

    monkeypatch.setattr(an, "stream_debate", stream_falso)

    at = AppTest.from_file(RUTA_APP)
    at.run(timeout=20)
    at.sidebar.radio(key="radio_seccion_activa").set_value(
        "Analista y Debate IA"
    ).run(timeout=20)

    at.button(key="btn_sug_goat").click().run(timeout=20)

    historial_debate = at.session_state["historial_debate"]
    assert len(historial_debate) == 2
    assert "Cristiano Ronaldo" in historial_debate[-1]["texto"]
    assert len(alm.cargar_opiniones(opiniones_temp)["opiniones"]) == 12
