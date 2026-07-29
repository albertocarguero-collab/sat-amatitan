# -*- coding: utf-8 -*-
"""
Geoportal Streamlit - SAT de sequia agricola, Microcuenca Amatitan
Autor: Carlos Carbajal
Curso: MCHV-513 Sistemas de Alerta e IA: Creacion de Geoportales para la Gestion de Cuencas

Descripcion:
Aplicacion Streamlit conectada a Google Earth Engine para monitorear sequia agricola
mediante SPI-3, VCI, IISS y una alerta integrada.
"""

import datetime
import json

import ee
import geemap.foliumap as geemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st_stats
import streamlit as st
from streamlit_folium import st_folium


# ==============================================================================
# CONFIGURACION GENERAL
# ==============================================================================

APP_TITLE = "SAT de Sequia Agricola - Microcuenca Amatitan"
PROJECT_ID_DEFAULT = "micuencaamatitan"

RUTA_CUENCA = "projects/micuencaamatitan/assets/MicrocuencaAmatitan"
RUTA_DRENAJE = "projects/micuencaamatitan/assets/RiosMicrocuencaAmatitan"
RUTA_DEM = "projects/micuencaamatitan/assets/FABDEM_TITIHUAPA"

CRS_METRICO = "EPSG:32616"
ESCALA_DEM = 30
ESCALA_CHIRPS = 5566
ESCALA_MODIS = 250

# Para calibrar SPI se recomienda una linea base historica cerrada.
# Los anos recientes se usan para monitoreo o comparacion, no necesariamente para calibracion.
ANIO_BASE_SPI_INICIO = 1981
ANIO_BASE_SPI_FIN_DEFAULT = 2023

# Linea base vegetacion. Puede ampliarse si se justifica metodologicamente.
ANIO_BASE_MODIS_INICIO = 2000
ANIO_BASE_MODIS_FIN = 2023

NOMBRES_ALERTA = {
    0: "Normal",
    1: "Vigilancia",
    2: "Prealerta",
    3: "Alerta",
    4: "Emergencia",
}

COLORES_ALERTA = {
    0: "#1a9850",
    1: "#91cf60",
    2: "#fee08b",
    3: "#fc8d59",
    4: "#b2182b",
}

UMBRALES_DEFAULT = {
    "SPI3": {
        "vigilancia": -0.5,
        "prealerta": -1.0,
        "alerta": -1.5,
        "emergencia": -2.0,
    },
    "VCI": {
        "vigilancia": 50,
        "prealerta": 40,
        "alerta": 30,
        "emergencia": 20,
    },
    "IISS": {
        "baja_max": 0.40,
        "moderada_max": 0.60,
        "alta_max": 0.80,
        "muy_alta_min": 0.80,
    },
}


# ==============================================================================
# INICIALIZACION DE GOOGLE EARTH ENGINE
# ==============================================================================

@st.cache_resource(show_spinner=False)
def inicializar_gee(project_id: str):
    """
    Inicializa Google Earth Engine.

    Uso local:
    1. Ejecutar en terminal: earthengine authenticate
    2. Luego: streamlit run app.py

    Uso en Streamlit Cloud:
    Configurar .streamlit/secrets.toml con una cuenta de servicio.
    """
    try:
        if "gee" in st.secrets:
            service_account = st.secrets["gee"].get("service_account")
            private_key = st.secrets["gee"].get("private_key")
            project = st.secrets["gee"].get("project", project_id)
            credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
            ee.Initialize(credentials, project=project)
        elif "EARTHENGINE_CREDENTIALS" in st.secrets:
            creds = json.loads(st.secrets["EARTHENGINE_CREDENTIALS"])
            credentials = ee.ServiceAccountCredentials(
                creds["client_email"], key_data=json.dumps(creds)
            )
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize(project=project_id)

        return True, "Google Earth Engine conectado correctamente."
    except Exception as exc:
        return False, f"No se pudo inicializar Google Earth Engine: {exc}"


# ==============================================================================
# CARGA DE ASSETS
# ==============================================================================

@st.cache_resource(show_spinner=False)
def cargar_assets():
    """Carga microcuenca, drenaje y DEM desde Earth Engine."""
    microcuenca = ee.FeatureCollection(RUTA_CUENCA)
    geom = microcuenca.geometry()

    drenaje = ee.FeatureCollection(RUTA_DRENAJE).filterBounds(geom)

    dem_raw = ee.Image(RUTA_DEM)
    dem_band = dem_raw.bandNames().get(0)
    dem = dem_raw.select([dem_band]).rename("elevation").clip(geom)

    return microcuenca, geom, drenaje, dem


# ==============================================================================
# FUNCIONES GEOESPACIALES E INDICADORES
# ==============================================================================

def calcular_pendiente(dem, geom):
    """Calcula pendiente en grados usando UTM Zona 16N."""
    dem_metric = dem.reproject(crs=CRS_METRICO, scale=ESCALA_DEM)
    return ee.Terrain.slope(dem_metric).rename("Slope").clip(geom)


def lluvia_mensual_feature(year: int, month: int, geom):
    """Devuelve un ee.Feature con la lluvia mensual media CHIRPS."""
    inicio = ee.Date.fromYMD(year, month, 1)
    fin = inicio.advance(1, "month")

    col = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(inicio, fin)
        .select("precipitation")
    )

    lluvia_mes = col.sum().clip(geom)
    lluvia_media = lluvia_mes.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=ESCALA_CHIRPS,
        maxPixels=1e9,
    ).get("precipitation")

    return ee.Feature(
        None,
        {
            "date": inicio.format("YYYY-MM-dd"),
            "year": year,
            "month": month,
            "rainfall": lluvia_media,
        },
    )


@st.cache_data(show_spinner=False)
def construir_serie_chirps(project_id: str, anio_inicio: int, anio_fin: int):
    """
    Construye serie mensual CHIRPS en cache.
    project_id se incluye para invalidar cache si cambia el proyecto.
    """
    ok, mensaje = inicializar_gee(project_id)
    if not ok:
        raise RuntimeError(mensaje)

    _, geom, _, _ = cargar_assets()

    features = []
    for year in range(anio_inicio, anio_fin + 1):
        for month in range(1, 13):
            features.append(lluvia_mensual_feature(year, month, geom))

    fc = ee.FeatureCollection(features)
    datos = fc.getInfo().get("features", [])

    df = pd.DataFrame([f["properties"] for f in datos])
    if df.empty:
        return pd.DataFrame(columns=["date", "year", "month", "rainfall"])

    df["date"] = pd.to_datetime(df["date"])
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    return df.dropna(subset=["rainfall"]).sort_values("date").reset_index(drop=True)


def calcular_spi3_gamma(df_lluvia: pd.DataFrame):
    """Calcula SPI-3 con ajuste Gamma por mes calendario."""
    df = df_lluvia.copy().sort_values("date").reset_index(drop=True)
    df["rain_3m"] = df["rainfall"].rolling(window=3, min_periods=3).sum()
    df["month_end"] = df["date"].dt.month
    df["SPI3"] = np.nan

    parametros_gamma = {}

    for mes in range(1, 13):
        idx = df["month_end"] == mes
        valores = df.loc[idx, "rain_3m"].dropna()

        if len(valores) < 10:
            continue

        n_total = len(valores)
        n_ceros = (valores <= 0).sum()
        prob_cero = n_ceros / n_total
        valores_pos = valores[valores > 0]

        if len(valores_pos) < 10:
            continue

        shape, loc, scale = st_stats.gamma.fit(valores_pos, floc=0)
        parametros_gamma[mes] = {
            "shape": shape,
            "loc": loc,
            "scale": scale,
            "prob_cero": prob_cero,
        }

        cdf_gamma = st_stats.gamma.cdf(df.loc[idx, "rain_3m"], shape, loc=loc, scale=scale)
        cdf_ajustada = prob_cero + (1 - prob_cero) * cdf_gamma
        cdf_ajustada = np.clip(cdf_ajustada, 0.0001, 0.9999)
        df.loc[idx, "SPI3"] = st_stats.norm.ppf(cdf_ajustada)

    return df, parametros_gamma


def calcular_spi3_actual(geom, parametros_gamma):
    """Calcula SPI-3 de los ultimos tres meses completos."""
    hoy = datetime.datetime.now()
    fecha_fin = ee.Date.fromYMD(hoy.year, hoy.month, 1)
    fecha_inicio = fecha_fin.advance(-3, "month")

    lluvia_img = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(fecha_inicio, fecha_fin)
        .select("precipitation")
        .sum()
    )

    lluvia_3m = lluvia_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=ESCALA_CHIRPS,
        maxPixels=1e9,
    ).get("precipitation").getInfo()

    mes_ref_fecha = datetime.datetime(hoy.year, hoy.month, 1) - pd.DateOffset(months=1)
    mes_ref = int(mes_ref_fecha.month)

    params = parametros_gamma.get(mes_ref)
    fecha_inicio_txt = fecha_inicio.format("YYYY-MM-dd").getInfo()
    fecha_fin_txt = fecha_fin.format("YYYY-MM-dd").getInfo()

    if params is None or lluvia_3m is None:
        return None, lluvia_3m, mes_ref, fecha_inicio_txt, fecha_fin_txt

    cdf_gamma = st_stats.gamma.cdf(
        lluvia_3m,
        params["shape"],
        loc=params["loc"],
        scale=params["scale"],
    )
    cdf_ajustada = params["prob_cero"] + (1 - params["prob_cero"]) * cdf_gamma
    cdf_ajustada = np.clip(cdf_ajustada, 0.0001, 0.9999)
    spi3 = st_stats.norm.ppf(cdf_ajustada)

    return spi3, lluvia_3m, mes_ref, fecha_inicio_txt, fecha_fin_txt


def clasificar_spi3(spi3: float, umbrales: dict):
    """Clasifica un valor SPI-3 en nivel de alerta."""
    if spi3 is None:
        return 0, "Sin datos SPI-3"
    if spi3 <= umbrales["SPI3"]["emergencia"]:
        return 4, "Emergencia climatica"
    if spi3 <= umbrales["SPI3"]["alerta"]:
        return 3, "Alerta climatica"
    if spi3 <= umbrales["SPI3"]["prealerta"]:
        return 2, "Prealerta climatica"
    if spi3 <= umbrales["SPI3"]["vigilancia"]:
        return 1, "Vigilancia climatica"
    return 0, "Condicion climatica normal"


def obtener_mes_modis_disponible(geom, max_retroceso: int = 6):
    """Busca el ultimo mes cerrado con MODIS disponible."""
    hoy = datetime.datetime.now()

    for i in range(1, max_retroceso + 1):
        fecha_ref = datetime.datetime(hoy.year, hoy.month, 1) - pd.DateOffset(months=i)
        year = int(fecha_ref.year)
        month = int(fecha_ref.month)

        inicio = ee.Date.fromYMD(year, month, 1)
        fin = inicio.advance(1, "month")

        col = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(inicio, fin)
            .filterBounds(geom)
            .select("NDVI")
        )

        if col.size().getInfo() > 0:
            return year, month

    return None, None


def calcular_vci_mes(geom, year: int, month: int):
    """Calcula VCI para un ano y mes especifico."""
    if year is None or month is None:
        return None

    inicio = ee.Date.fromYMD(year, month, 1)
    fin = inicio.advance(1, "month")

    col_actual = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(inicio, fin)
        .filterBounds(geom)
        .select("NDVI")
    )

    if col_actual.size().getInfo() == 0:
        return None

    ndvi_actual = col_actual.median().multiply(0.0001).rename("NDVI_actual").clip(geom)

    ndvi_hist_mes = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterBounds(geom)
        .filter(ee.Filter.calendarRange(ANIO_BASE_MODIS_INICIO, ANIO_BASE_MODIS_FIN, "year"))
        .filter(ee.Filter.calendarRange(month, month, "month"))
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    )

    if ndvi_hist_mes.size().getInfo() == 0:
        return None

    ndvi_min = ndvi_hist_mes.min().rename("NDVI_min").clip(geom)
    ndvi_max = ndvi_hist_mes.max().rename("NDVI_max").clip(geom)
    denominador = ndvi_max.subtract(ndvi_min).where(ndvi_max.subtract(ndvi_min).eq(0), 0.0001)

    return (
        ndvi_actual.subtract(ndvi_min)
        .divide(denominador)
        .multiply(100)
        .clamp(0, 100)
        .rename("VCI")
        .clip(geom)
    )


def promedio_imagen(img, geom, banda: str, escala: int):
    """Calcula promedio espacial de una imagen de Earth Engine."""
    if img is None:
        return None
    try:
        return img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=escala,
            maxPixels=1e9,
        ).get(banda).getInfo()
    except Exception:
        return None


def clasificar_vci(vci: float, umbrales: dict):
    """Clasifica VCI promedio en nivel de alerta."""
    if vci is None:
        return 0, "Sin datos VCI"
    if vci < umbrales["VCI"]["emergencia"]:
        return 4, "Emergencia vegetativa"
    if vci < umbrales["VCI"]["alerta"]:
        return 3, "Alerta vegetativa"
    if vci < umbrales["VCI"]["prealerta"]:
        return 2, "Prealerta vegetativa"
    if vci < umbrales["VCI"]["vigilancia"]:
        return 1, "Vigilancia vegetativa"
    return 0, "Condicion vegetativa normal"


def clasificar_vci_imagen(vci, geom):
    """Clasifica VCI como imagen categorical."""
    if vci is None:
        return ee.Image(0).rename("VCI_clase").clip(geom)
    return (
        ee.Image(0)
        .where(vci.lte(50).And(vci.gt(40)), 1)
        .where(vci.lte(40).And(vci.gt(30)), 2)
        .where(vci.lte(30).And(vci.gt(20)), 3)
        .where(vci.lte(20), 4)
        .rename("VCI_clase")
        .clip(geom)
    )


def normalizar_imagen(img, geom, nombre_banda: str, escala: int):
    """Normaliza una imagen entre 0 y 1 dentro de la microcuenca."""
    stats = img.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=geom,
        scale=escala,
        maxPixels=1e9,
    )

    min_val = ee.Number(stats.get(f"{nombre_banda}_min"))
    max_val = ee.Number(stats.get(f"{nombre_banda}_max"))
    denominador = max_val.subtract(min_val).max(0.0001)
    return img.subtract(min_val).divide(denominador).clamp(0, 1)


def calcular_iiss(geom, pendiente):
    """Calcula IISS con NDVI P10 historico y pendiente."""
    modis_hist = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(f"{ANIO_BASE_MODIS_INICIO}-01-01", f"{ANIO_BASE_MODIS_FIN}-12-31")
        .filterBounds(geom)
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
    )

    ndvi_p10 = modis_hist.reduce(ee.Reducer.percentile([10])).rename("NDVI_P10").clip(geom)
    ndvi_norm = normalizar_imagen(ndvi_p10, geom, "NDVI_P10", ESCALA_MODIS)
    vuln_vegetacion = ee.Image(1).subtract(ndvi_norm).rename("Vuln_Vegetacion")
    pendiente_norm = normalizar_imagen(pendiente, geom, "Slope", ESCALA_DEM).rename("Vuln_Pendiente")

    iiss = (
        vuln_vegetacion.multiply(0.70)
        .add(pendiente_norm.multiply(0.30))
        .rename("IISS")
        .clip(geom)
    )
    return iiss, ndvi_p10


def clasificar_iiss(iiss, geom):
    """Clasifica IISS en cuatro niveles."""
    return (
        ee.Image(0)
        .where(iiss.gt(0).And(iiss.lte(0.40)), 1)
        .where(iiss.gt(0.40).And(iiss.lte(0.60)), 2)
        .where(iiss.gt(0.60).And(iiss.lte(0.80)), 3)
        .where(iiss.gt(0.80), 4)
        .rename("IISS_clase")
        .clip(geom)
    )


def crear_alerta_integrada(nivel_spi: int, vci_clase, iiss_clase, geom):
    """Integra SPI-3, VCI e IISS en una capa de alerta."""
    spi_img = ee.Image.constant(nivel_spi).rename("SPI_clase")

    condicion_vigilancia = spi_img.gte(1).Or(vci_clase.gte(1))
    condicion_prealerta = spi_img.gte(2).And(vci_clase.gte(2)).And(iiss_clase.gte(3))
    condicion_alerta = spi_img.gte(3).And(vci_clase.gte(3)).And(iiss_clase.gte(3))
    condicion_emergencia = spi_img.gte(4).And(vci_clase.gte(4)).And(iiss_clase.eq(4))

    return (
        ee.Image(0)
        .where(condicion_vigilancia, 1)
        .where(condicion_prealerta, 2)
        .where(condicion_alerta, 3)
        .where(condicion_emergencia, 4)
        .rename("Alerta_Sequia")
        .clip(geom)
    )


def area_por_clase(imagen_clase, geom, escala: int = ESCALA_MODIS):
    """Calcula area por clase en hectareas."""
    area_img = ee.Image.pixelArea().addBands(imagen_clase)
    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="clase"),
        geometry=geom,
        scale=escala,
        maxPixels=1e9,
    )

    grupos = stats.getInfo().get("groups", [])
    registros = []
    for g in grupos:
        registros.append({"clase": int(g["clase"]), "area_ha": round(g["sum"] / 10000, 2)})
    return pd.DataFrame(registros)


def promedio_por_clase_para_descarga(df_area):
    """Agrega nombres de alerta a la tabla de area."""
    if df_area.empty:
        return df_area
    df = df_area.copy().sort_values("clase")
    df["nivel"] = df["clase"].map(NOMBRES_ALERTA)
    return df[["clase", "nivel", "area_ha"]]


# ==============================================================================
# FUNCIONES DE VISUALIZACION
# ==============================================================================

def agregar_leyenda_alerta(mapa):
    labels = [NOMBRES_ALERTA[i] for i in range(5)]
    colors = [COLORES_ALERTA[i] for i in range(5)]
    try:
        mapa.add_legend(title="Nivel de alerta", keys=labels, colors=colors)
    except TypeError:
        mapa.add_legend(title="Nivel de alerta", labels=labels, colors=colors)
    return mapa


def construir_grafico_spi(df_spi: pd.DataFrame):
    df_plot = df_spi.dropna(subset=["SPI3"]).copy()
    df_plot["color"] = df_plot["SPI3"].apply(lambda x: "#d73027" if x < 0 else "#4575b4")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(df_plot["date"], df_plot["SPI3"], width=25, color=df_plot["color"], alpha=0.75)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(-1.0, color="orange", linestyle="--", label="SPI <= -1.0")
    ax.axhline(-1.5, color="red", linestyle="--", label="SPI <= -1.5")
    ax.axhline(-2.0, color="darkred", linestyle="--", label="SPI <= -2.0")
    ax.set_title("SPI-3 historico - Microcuenca Amatitan")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("SPI-3")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# ==============================================================================
# INTERFAZ STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="SAT Sequia Agricola Amatitan",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌱 SAT de Sequía Agrícola - Microcuenca Amatitán")
st.caption("Geoportal operativo con Google Earth Engine, CHIRPS, MODIS, FABDEM, SPI-3, VCI e IISS")

with st.sidebar:
    st.header("Configuración")
    project_id = st.text_input("Proyecto Google Earth Engine", value=PROJECT_ID_DEFAULT)

    st.subheader("Línea base SPI-3")
    anio_inicio = st.number_input("Año inicial CHIRPS", min_value=1981, max_value=2023, value=ANIO_BASE_SPI_INICIO, step=1)
    anio_fin = st.number_input("Año final línea base", min_value=1990, max_value=2025, value=ANIO_BASE_SPI_FIN_DEFAULT, step=1)

    st.subheader("MODIS")
    max_retroceso_modis = st.slider("Meses hacia atrás para buscar MODIS", 1, 12, 6)

    st.subheader("Capas")
    mostrar_alerta = st.checkbox("Alerta integrada", value=True)
    mostrar_iiss = st.checkbox("IISS", value=True)
    mostrar_vci = st.checkbox("VCI actual", value=False)
    mostrar_drenaje = st.checkbox("Red de drenaje", value=True)

    st.subheader("Umbrales")
    usar_umbrales_sensibles = st.checkbox("Usar umbrales sensibles de fase piloto", value=False)

    if usar_umbrales_sensibles:
        UMBRALES = {
            "SPI3": {"vigilancia": -0.2, "prealerta": -0.7, "alerta": -1.2, "emergencia": -1.8},
            "VCI": {"vigilancia": 60, "prealerta": 45, "alerta": 35, "emergencia": 25},
            "IISS": UMBRALES_DEFAULT["IISS"],
        }
    else:
        UMBRALES = UMBRALES_DEFAULT

    if st.button("Actualizar / limpiar caché"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


ok, mensaje = inicializar_gee(project_id)
if not ok:
    st.error(mensaje)
    st.info("En local ejecuta: earthengine authenticate. En Streamlit Cloud configura .streamlit/secrets.toml.")
    st.stop()
else:
    st.sidebar.success("GEE conectado")

try:
    microcuenca, geom, drenaje, dem = cargar_assets()
except Exception as exc:
    st.error("No se pudieron cargar los assets de Earth Engine. Revisa rutas y permisos.")
    st.exception(exc)
    st.stop()


with st.spinner("Calculando indicadores del SAT..."):
    try:
        area_km2 = geom.area().divide(1e6).getInfo()
        pendiente = calcular_pendiente(dem, geom)

        df_lluvia = construir_serie_chirps(project_id, int(anio_inicio), int(anio_fin))
        if df_lluvia.empty:
            st.error("No se pudo construir la serie CHIRPS.")
            st.stop()

        df_spi, parametros_gamma = calcular_spi3_gamma(df_lluvia)
        spi3_actual, lluvia_3m_actual, mes_ref_spi, fecha_inicio_spi, fecha_fin_spi = calcular_spi3_actual(geom, parametros_gamma)
        nivel_spi, texto_spi = clasificar_spi3(spi3_actual, UMBRALES)

        anio_ref_vci, mes_ref_vci = obtener_mes_modis_disponible(geom, max_retroceso_modis)
        vci_actual = calcular_vci_mes(geom, anio_ref_vci, mes_ref_vci)
        vci_promedio = promedio_imagen(vci_actual, geom, "VCI", ESCALA_MODIS)
        nivel_vci, texto_vci = clasificar_vci(vci_promedio, UMBRALES)
        vci_clase = clasificar_vci_imagen(vci_actual, geom)

        iiss, ndvi_p10 = calcular_iiss(geom, pendiente)
        iiss_clase = clasificar_iiss(iiss, geom)
        alerta_integrada = crear_alerta_integrada(nivel_spi, vci_clase, iiss_clase, geom)
        df_area_alerta = area_por_clase(alerta_integrada, geom, ESCALA_MODIS)
        df_area_alerta_view = promedio_por_clase_para_descarga(df_area_alerta)

    except Exception as exc:
        st.error("Ocurrió un error durante el cálculo del SAT.")
        st.exception(exc)
        st.stop()


tab_monitoreo, tab_mapa, tab_spi, tab_areas, tab_descargas, tab_metodo = st.tabs(
    ["Monitoreo actual", "Mapa", "SPI histórico", "Áreas", "Descargas", "Metodología"]
)

with tab_monitoreo:
    st.subheader("Indicadores actuales")
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Área microcuenca", f"{area_km2:.2f} km²")
    c2.metric("SPI-3 actual", "Sin datos" if spi3_actual is None else f"{spi3_actual:.2f}", texto_spi)
    c3.metric("Lluvia 3 meses", "Sin datos" if lluvia_3m_actual is None else f"{lluvia_3m_actual:.1f} mm")
    c4.metric("VCI promedio", "Sin datos" if vci_promedio is None else f"{vci_promedio:.1f}", texto_vci)
    c5.metric("Estado general", NOMBRES_ALERTA.get(max(nivel_spi, nivel_vci), "Sin datos"))

    st.markdown("---")
    if nivel_spi >= 4 and nivel_vci >= 4:
        st.error("EMERGENCIA: activar respuesta prioritaria en zonas IISS Muy Alta.")
    elif nivel_spi >= 3 and nivel_vci >= 3:
        st.warning("ALERTA: activar medidas de mitigación en zonas IISS Alta y Muy Alta.")
    elif nivel_spi >= 2 and nivel_vci >= 2:
        st.warning("PREALERTA: iniciar monitoreo quincenal y comunicación preventiva con productores.")
    elif nivel_spi >= 1 or nivel_vci >= 1:
        st.info("VIGILANCIA: mantener monitoreo mensual y revisar pronóstico climático.")
    else:
        st.success("NORMAL: continuar monitoreo rutinario.")

    st.write(f"**Periodo SPI-3:** {fecha_inicio_spi} a {fecha_fin_spi}")
    st.write(f"**Mes de referencia VCI:** {'Sin datos' if anio_ref_vci is None else f'{anio_ref_vci}-{mes_ref_vci:02d}'}")

with tab_mapa:
    st.subheader("Mapa integrado de alerta")

    vis_alerta = {"min": 0, "max": 4, "palette": [COLORES_ALERTA[i] for i in range(5)]}
    vis_iiss = {"min": 0, "max": 1, "palette": ["#ffffcc", "#fed976", "#fd8d3c", "#fc4e2a", "#bd0026", "#800026"]}
    vis_vci = {"min": 0, "max": 100, "palette": ["red", "yellow", "green"]}

    mapa = geemap.Map()
    mapa.centerObject(microcuenca, 13)
    mapa.add_basemap("SATELLITE")

    if mostrar_alerta:
        mapa.addLayer(alerta_integrada, vis_alerta, "Alerta integrada")
    if mostrar_iiss:
        mapa.addLayer(iiss, vis_iiss, "IISS")
    if mostrar_vci and vci_actual is not None:
        mapa.addLayer(vci_actual, vis_vci, "VCI actual")
    if mostrar_drenaje:
        mapa.addLayer(drenaje, {"color": "00ffff"}, "Red de drenaje")

    mapa.addLayer(microcuenca, {"color": "white", "fillColor": "00000000"}, "Microcuenca")
    agregar_leyenda_alerta(mapa)
    st_folium(mapa, width=None, height=650)

with tab_spi:
    st.subheader("Serie histórica SPI-3")
    st.pyplot(construir_grafico_spi(df_spi))
    with st.expander("Ver tabla SPI-3"):
        st.dataframe(df_spi[["date", "rainfall", "rain_3m", "SPI3"]], use_container_width=True)

with tab_areas:
    st.subheader("Área por nivel de alerta")
    if df_area_alerta_view.empty:
        st.info("No se pudo calcular el área por nivel de alerta.")
    else:
        st.dataframe(df_area_alerta_view, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            df_area_alerta_view["nivel"],
            df_area_alerta_view["area_ha"],
            color=[COLORES_ALERTA.get(c, "gray") for c in df_area_alerta_view["clase"]],
        )
        ax.set_title("Área por nivel de alerta")
        ax.set_xlabel("Nivel")
        ax.set_ylabel("Área ha")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

with tab_descargas:
    st.subheader("Descarga de resultados")
    st.download_button(
        "Descargar SPI-3 CSV",
        data=df_spi.to_csv(index=False).encode("utf-8"),
        file_name="spi3_amatitan.csv",
        mime="text/csv",
    )

    if not df_area_alerta_view.empty:
        st.download_button(
            "Descargar áreas por alerta CSV",
            data=df_area_alerta_view.to_csv(index=False).encode("utf-8"),
            file_name="areas_alerta_amatitan.csv",
            mime="text/csv",
        )

with tab_metodo:
    st.subheader("Metodología")
    st.markdown(
        """
        Este geoportal implementa un prototipo operativo de SAT de sequía agrícola para la microcuenca Amatitán.

        **Componentes:**
        1. **SPI-3:** déficit acumulado de precipitación de tres meses con CHIRPS.
        2. **VCI:** condición actual de vegetación con MODIS NDVI.
        3. **IISS:** susceptibilidad espacial con NDVI P10 histórico y pendiente FABDEM.
        4. **Alerta integrada:** combinación de condición climática, estrés vegetal y susceptibilidad territorial.

        **Lectura operativa:**
        - SPI-3 indica cuándo hay déficit de lluvia.
        - VCI indica si la vegetación muestra estrés.
        - IISS indica dónde priorizar acciones.

        **Nota:** El sistema debe validarse con registros locales de campo, calendario agrícola, rendimientos y reportes de afectación.
        """
    )
