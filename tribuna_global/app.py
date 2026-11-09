"""Interfaz principal de Streamlit para Tribuna Global.

Organizada en tres secciones independientes seleccionables desde ``st.sidebar.radio``:
1. «Emitir opinión»: selección de jugador del catálogo, clasificación con temperatura 0
   mediante ``invoke`` y persistencia atómica en ``datos/opiniones.json``.
2. «Consultar opiniones»: filtros por jugador y sentimiento, métricas por categoría,
   tabla de registros y descarga JSON sin invocar al modelo de lenguaje.
3. «Analista y Debate IA»: conversación en streaming (``.stream()``) sobre historia,
   rankings por posición y el debate del GOAT (sosteniendo a Cristiano Ronaldo con 6 secciones).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

import almacenamiento as alm
import analisis as an

load_dotenv()

SECCION_EMITIR = "Emitir opinión"
SECCION_CONSULTAR = "Consultar opiniones"
SECCION_DEBATE = "Analista y Debate IA"
SECCIONES = (SECCION_EMITIR, SECCION_CONSULTAR, SECCION_DEBATE)

ETIQUETAS_SENTIMIENTO: dict[str, str] = {
    "positivo": "🟢 Positivo",
    "negativo": "🔴 Negativo",
    "neutro": "⚪ Neutro",
    "mixto": "🟡 Mixto",
}


def inicializar_estado() -> None:
    """Inicializa las claves de ``st.session_state`` separadas por responsabilidad."""
    if "historial_opiniones" not in st.session_state:
        st.session_state.historial_opiniones = []
    if "envio_pendiente" not in st.session_state:
        st.session_state.envio_pendiente = None
    if "ids_guardados_sesion" not in st.session_state:
        st.session_state.ids_guardados_sesion = set()
    if "historial_debate" not in st.session_state:
        st.session_state.historial_debate = []


def formatear_jugador(jugador: dict[str, str]) -> str:
    """Formatea un jugador para selectores mostrando nombre y país."""
    return f"{jugador['nombre']} ({jugador['pais']})"


def procesar_envio_opinion(
    jugador_capturado: dict[str, str],
    texto_opinion: str,
    *,
    registro_id: str | None = None,
) -> None:
    """Ejecuta el flujo de validación, clasificación, persistencia y retroalimentación."""
    id_envio = registro_id or str(uuid.uuid4())
    jugador_fijo = dict(jugador_capturado)

    # 1. Validación de entrada antes de llamar al modelo
    try:
        texto_limpio = an.validar_entrada(texto_opinion)
    except an.ErrorEntrada as exc:
        st.warning(f"⚠️ Entrada no válida: {exc}")
        st.session_state.envio_pendiente = {
            "registro_id": id_envio,
            "jugador": jugador_fijo,
            "texto": texto_opinion,
            "etapa": "entrada",
            "mensaje_error": str(exc),
        }
        return

    # 2. Clasificación con el modelo (solo la opinión actual, sin historial visual)
    nombre_modelo = an.obtener_nombre_modelo()
    try:
        analisis_resultado = an.analizar_opinion(jugador_fijo, texto_limpio)
    except an.OpinionNoPertinente as exc:
        st.session_state.envio_pendiente = None
        st.session_state.historial_opiniones.append(
            {
                "rol": "user",
                "jugador_id": jugador_fijo["id"],
                "jugador_nombre": jugador_fijo["nombre"],
                "jugador_pais": jugador_fijo["pais"],
                "texto": texto_limpio,
            }
        )
        st.session_state.historial_opiniones.append(
            {
                "rol": "assistant",
                "jugador_id": jugador_fijo["id"],
                "jugador_nombre": jugador_fijo["nombre"],
                "sentimiento": None,
                "razon": exc.razon,
                "guardado": False,
                "es_pertinente": False,
            }
        )
        with st.chat_message("user"):
            st.markdown(f"**[Jugador: {jugador_fijo['nombre']}]**\n\n{texto_limpio}")
        with st.chat_message("assistant"):
            st.markdown(
                f"**Jugador seleccionado:** {jugador_fijo['nombre']}  \n"
                f"**Resultado:** No pertinente (no guardado)  \n"
                f"**Justificación:** {exc.razon}"
            )
        st.warning(
            f"🚫 Opinión no pertinente para **{jugador_fijo['nombre']}**: {exc.razon} "
            "(No se guardó en el archivo)."
        )
        return
    except an.ErrorClasificacion as exc:
        st.error(f"❌ Error de clasificación: {exc}")
        st.session_state.envio_pendiente = {
            "registro_id": id_envio,
            "jugador": jugador_fijo,
            "texto": texto_limpio,
            "etapa": "clasificacion",
            "mensaje_error": str(exc),
        }
        return

    # 3. Construcción del registro y escritura atómica en disco
    try:
        registro = alm.crear_registro(
            jugador_fijo,
            texto_limpio,
            analisis_resultado,
            nombre_modelo,
            registro_id=id_envio,
            version_prompt=an.VERSION_PROMPT,
        )
        guardado_nuevo = alm.guardar_opinion(alm.RUTA_OPINIONES, registro)
    except (alm.ErrorAlmacenamiento, ValueError) as exc:
        st.error(
            f"❌ Error al guardar en disco: {exc}. "
            "No se registró la opinión; tu texto se conserva abajo para reintentar."
        )
        st.session_state.envio_pendiente = {
            "registro_id": id_envio,
            "jugador": jugador_fijo,
            "texto": texto_limpio,
            "etapa": "almacenamiento",
            "mensaje_error": str(exc),
        }
        return

    # 4. Confirmación únicamente tras una escritura exitosa
    st.session_state.envio_pendiente = None
    st.session_state.ids_guardados_sesion.add(id_envio)

    st.session_state.historial_opiniones.append(
        {
            "rol": "user",
            "jugador_id": jugador_fijo["id"],
            "jugador_nombre": jugador_fijo["nombre"],
            "jugador_pais": jugador_fijo["pais"],
            "texto": texto_limpio,
        }
    )
    st.session_state.historial_opiniones.append(
        {
            "rol": "assistant",
            "jugador_id": jugador_fijo["id"],
            "jugador_nombre": jugador_fijo["nombre"],
            "sentimiento": analisis_resultado.sentimiento,
            "razon": analisis_resultado.razon,
            "guardado": True,
            "es_pertinente": True,
            "registro_id": id_envio,
        }
    )

    etiqueta = ETIQUETAS_SENTIMIENTO.get(
        analisis_resultado.sentimiento or "", analisis_resultado.sentimiento or ""
    )
    with st.chat_message("user"):
        st.markdown(f"**[Jugador: {jugador_fijo['nombre']}]**\n\n{texto_limpio}")
    with st.chat_message("assistant"):
        st.markdown(
            f"**Jugador analizado:** {jugador_fijo['nombre']}  \n"
            f"**Sentimiento estimado:** {etiqueta}  \n"
            f"**Justificación:** {analisis_resultado.razon}"
        )
        st.caption("💾 Guardado en `datos/opiniones.json`.")
    if guardado_nuevo:
        st.success(
            f"✅ Opinión sobre **{jugador_fijo['nombre']}** clasificada como "
            f"**{etiqueta}** y guardada correctamente."
        )
    else:
        st.info(
            f"ℹ️ Este envío sobre **{jugador_fijo['nombre']}** ya se encontraba guardado."
        )


def renderizar_seccion_emitir(
    catalogo: dict[str, dict[str, str]],
    jugador_actual: dict[str, str],
) -> None:
    """Renderiza la sección «Emitir opinión»."""
    st.subheader("🗣️ Emitir opinión sobre un futbolista")
    st.caption(
        f"Futbolista activo en barra lateral: **{jugador_actual['nombre']} ({jugador_actual['pais']})** · "
        f"Longitud permitida: {an.LONGITUD_MINIMA} a {an.LONGITUD_MAXIMA} caracteres."
    )

    col_info, col_limpiar = st.columns([3, 1])
    with col_info:
        st.markdown(
            "Cada opinión se analiza de manera independiente con temperatura 0 "
            "y se asocia al jugador seleccionado en el momento del envío."
        )
    with col_limpiar:
        if st.button("🗑️ Nueva conversación", key="btn_nueva_conversacion"):
            st.session_state.historial_opiniones = []
            st.session_state.envio_pendiente = None
            st.rerun()

    # Historial visual de la sesión
    for item in st.session_state.historial_opiniones:
        if item["rol"] == "user":
            with st.chat_message("user"):
                st.markdown(
                    f"**[Jugador: {item['jugador_nombre']}]**\n\n{item['texto']}"
                )
        else:
            with st.chat_message("assistant"):
                if item.get("es_pertinente") and item.get("sentimiento"):
                    etiqueta = ETIQUETAS_SENTIMIENTO.get(
                        item["sentimiento"], item["sentimiento"]
                    )
                    st.markdown(
                        f"**Jugador analizado:** {item['jugador_nombre']}  \n"
                        f"**Sentimiento estimado:** {etiqueta}  \n"
                        f"**Justificación:** {item['razon']}"
                    )
                    if item.get("guardado"):
                        st.caption("💾 Guardado en `datos/opiniones.json`.")
                else:
                    st.markdown(
                        f"**Jugador seleccionado:** {item['jugador_nombre']}  \n"
                        f"**Resultado:** No pertinente (no guardado)  \n"
                        f"**Justificación:** {item['razon']}"
                    )

    # Panel de recuperación si el último envío falló
    pendiente = st.session_state.envio_pendiente
    if isinstance(pendiente, dict):
        st.info(
            f"📌 Se conservó tu opinión para **{pendiente['jugador']['nombre']}** tras un fallo de "
            f"**{pendiente['etapa']}**: «{pendiente['texto']}»"
        )
        if st.button("🔄 Reintentar envío conservado", key="btn_reintentar_envio"):
            procesar_envio_opinion(
                pendiente["jugador"],
                pendiente["texto"],
                registro_id=pendiente["registro_id"],
            )

    opinion_usuario = st.chat_input(
        f"Escribe tu opinión sobre {jugador_actual['nombre']}...",
        key="chat_input_opinion",
    )
    if opinion_usuario is not None:
        procesar_envio_opinion(jugador_actual, opinion_usuario)


def renderizar_seccion_consultar(catalogo: dict[str, dict[str, str]]) -> None:
    """Renderiza la sección «Consultar opiniones» sin realizar llamadas al modelo."""
    st.subheader("📊 Consultar opiniones registradas")
    st.caption(
        "Esta sección lee directamente `datos/opiniones.json` y filtra en memoria sin invocar al modelo."
    )

    try:
        datos_opiniones = alm.cargar_opiniones(alm.RUTA_OPINIONES)
    except alm.ErrorAlmacenamiento as exc:
        st.error(f"❌ Error al leer el archivo de opiniones: {exc}")
        return

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        opciones_jugador = ["Todos"] + list(catalogo.keys())
        filtro_jugador = st.selectbox(
            "Filtrar por jugador",
            options=opciones_jugador,
            format_func=lambda jid: "Todos los jugadores"
            if jid == "Todos"
            else formatear_jugador(catalogo[jid]),
            key="filtro_consulta_jugador",
        )
    with col_f2:
        opciones_sentimiento = ["Todos"] + list(alm.SENTIMIENTOS)
        filtro_sentimiento = st.selectbox(
            "Filtrar por sentimiento",
            options=opciones_sentimiento,
            format_func=lambda s: "Todos los sentimientos"
            if s == "Todos"
            else ETIQUETAS_SENTIMIENTO.get(s, s.capitalize()),
            key="filtro_consulta_sentimiento",
        )

    filtradas = alm.consultar_opiniones(
        datos_opiniones,
        jugador_id=filtro_jugador,
        sentimiento=filtro_sentimiento,
    )
    conteos = alm.contar_por_sentimiento(filtradas)

    # Recuentos con los mismos filtros aplicados
    m_total, m_pos, m_neg, m_neu, m_mix = st.columns(5)
    m_total.metric("Total filtrados", len(filtradas))
    m_pos.metric("Positivo", conteos["positivo"])
    m_neg.metric("Negativo", conteos["negativo"])
    m_neu.metric("Neutro", conteos["neutro"])
    m_mix.metric("Mixto", conteos["mixto"])

    if not filtradas:
        st.info("ℹ️ No se encontraron opiniones registradas para los filtros seleccionados.")
    else:
        filas_tabla: list[dict[str, Any]] = [
            {
                "Fecha (UTC)": reg["fecha_utc"],
                "Jugador": reg["jugador_nombre"],
                "Opinión original": reg["opinion_original"],
                "Sentimiento": reg["sentimiento"],
                "Razón": reg["razon"],
            }
            for reg in filtradas
        ]
        st.dataframe(filas_tabla, hide_index=True)

    payload_descarga = {
        "version": datos_opiniones.get("version", 1),
        "filtros_aplicados": {
            "jugador_id": None if filtro_jugador == "Todos" else filtro_jugador,
            "sentimiento": None if filtro_sentimiento == "Todos" else filtro_sentimiento,
        },
        "total": len(filtradas),
        "conteo_por_sentimiento": conteos,
        "opiniones": filtradas,
    }
    json_descarga = json.dumps(payload_descarga, ensure_ascii=False, indent=2)
    st.download_button(
        label="⬇️ Descargar resultados filtrados (JSON)",
        data=json_descarga.encode("utf-8"),
        file_name="opiniones_filtradas.json",
        mime="application/json",
        key="btn_descargar_json",
    )


def renderizar_seccion_debate() -> None:
    """Renderiza la pestaña adicional «Analista y Debate IA» con salida en streaming."""
    st.subheader("🎙️ Analista y Debate Futbolero (IA)")
    st.caption(
        "Debate sobre el mejor jugador de la historia (GOAT) o solicita listas categorizadas "
        "(mejores defensas, delanteros, mediocampistas, etc.). Las respuestas usan `.stream()` y no alteran `opiniones.json`."
    )

    col_sug1, col_sug2, col_limpiar = st.columns([2, 2, 1])
    pregunta_rapida: str | None = None
    with col_sug1:
        if st.button(
            "🐐 ¿Quién es el mejor jugador de la historia?",
            key="btn_sug_goat",
        ):
            pregunta_rapida = "¿Quién es el mejor jugador del mundo y de la historia?"
    with col_sug2:
        if st.button(
            "🛡️ Mencióname los mejores defensas de la historia",
            key="btn_sug_defensas",
        ):
            pregunta_rapida = "Mencióname los mejores defensas de la historia con su país, época y justificación."
    with col_limpiar:
        if st.button("🗑️ Limpiar debate", key="btn_limpiar_debate"):
            st.session_state.historial_debate = []
            st.rerun()

    for msg in st.session_state.historial_debate:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["texto"])

    pregunta_chat = st.chat_input(
        "Pregunta al analista (ej. ¿Quién es el mejor del mundo? o Mejores delanteros)...",
        key="chat_input_debate",
    )
    consulta = pregunta_chat if pregunta_chat is not None else pregunta_rapida

    if consulta is not None:
        try:
            consulta_limpia = an.validar_entrada(consulta)
        except an.ErrorEntrada as exc:
            st.warning(f"⚠️ Consulta no válida: {exc}")
            return

        with st.chat_message("user"):
            st.markdown(consulta_limpia)

        try:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                respuesta_acumulada = ""
                for fragmento in an.stream_debate(
                    consulta_limpia,
                    historial=st.session_state.historial_debate,
                ):
                    respuesta_acumulada += fragmento
                    placeholder.markdown(respuesta_acumulada + "▌")
                placeholder.markdown(respuesta_acumulada)

            st.session_state.historial_debate.append(
                {"rol": "user", "texto": consulta_limpia}
            )
            st.session_state.historial_debate.append(
                {"rol": "assistant", "texto": respuesta_acumulada}
            )
        except an.ErrorClasificacion as exc:
            st.error(f"❌ {exc}")


def main() -> None:
    """Punto de entrada principal de la aplicación Streamlit."""
    st.set_page_config(
        page_title="Tribuna Global - Percepción de Futbolistas",
        page_icon="⚽",
        layout="wide",
    )
    inicializar_estado()

    st.title("⚽ Tribuna Global")
    st.markdown(
        "Sistema de registro y clasificación de opiniones de aficionados sobre futbolistas "
        "internacionales con **LangChain** y **Streamlit**. *La clasificación refleja lo expresado "
        "por cada participante y no constituye una medida objetiva de calidad deportiva.*"
    )

    try:
        catalogo = alm.cargar_jugadores(alm.RUTA_JUGADORES)
    except alm.ErrorAlmacenamiento as exc:
        st.error(f"❌ Error crítico en el catálogo de jugadores: {exc}")
        st.stop()
        return

    with st.sidebar:
        st.header("🧭 Navegación")
        seccion_activa = st.radio(
            "Sección",
            options=list(SECCIONES),
            key="radio_seccion_activa",
        )

        st.divider()
        st.header("👟 Selección de futbolista")
        ids_jugadores = list(catalogo.keys())
        jugador_id_seleccionado = st.selectbox(
            "Jugador del catálogo",
            options=ids_jugadores,
            format_func=lambda jid: formatear_jugador(catalogo[jid]),
            key="selectbox_jugador_id",
        )
        jugador_actual = catalogo[jugador_id_seleccionado]
        st.caption(
            f"ID interno: `{jugador_actual['id']}` · País: **{jugador_actual['pais']}**"
        )

        st.divider()
        st.caption(f"Modelo configurado: `{an.obtener_nombre_modelo()}` · Temperatura clasificador: `0`")

    if seccion_activa == SECCION_EMITIR:
        renderizar_seccion_emitir(catalogo, jugador_actual)
    elif seccion_activa == SECCION_CONSULTAR:
        renderizar_seccion_consultar(catalogo)
    else:
        renderizar_seccion_debate()


if __name__ == "__main__":
    main()
