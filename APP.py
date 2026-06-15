import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="App Analizadora de Datasets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.cache_data
def cargar_csv(archivo):
    """Carga un CSV desde un objeto subido o una ruta local."""
    try:
        df = pd.read_csv(archivo, low_memory=False)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("/", "_")
            .str.replace("-", "_")
        )
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None


def detectar_tipos(df):
    """Clasifica columnas en numéricas, categóricas, binarias y de fecha."""
    numericas = df.select_dtypes(include=["number"]).columns.tolist()
    categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    binarias = [c for c in numericas if df[c].nunique() == 2]
    fechas = []
    for col in df.columns:
        if "date" in col or "fecha" in col:
            try:
                pd.to_datetime(df[col], errors="raise")
                fechas.append(col)
            except Exception:
                pass
    return numericas, categoricas, binarias, fechas


def convertir_fechas(df, cols_fecha):
    """Convierte columnas de fecha con manejo de errores."""
    for col in cols_fecha:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def detectar_outliers_iqr(df, col):
    """Devuelve la cantidad de outliers usando la regla IQR."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] < lower) | (df[col] > upper)].shape[0]



with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/bar-chart.png",
        width=72,
    )
    st.title("📊 Menú Principal")
    st.markdown("---")

    seccion = st.selectbox(
        "Navegar a:",
        [
            "🏠 Home",
            "📂 Carga y Perfil del Dataset",
            "⚙️ Procesamiento de Datos",
            "📈 Análisis Visual",
        ],
    )

    st.markdown("---")
    st.subheader("📁 Cargar Dataset")

    datasets_predefinidos = {
        "— Selecciona uno —": None,
        "AI Impact on Jobs 2030": "data/AI_Impact_on_Jobs_2030.csv",
        "Superstore Sales": "data/sample_-_superstore.csv",
        "E-commerce Risk": "data/synthetic_ecommerce_order_risk_dataset.csv",
        "Teen Mental Health": "data/Teen_Mental_Health_Dataset.csv",
    }

    dataset_elegido = st.selectbox(
        "Dataset predefinido:", list(datasets_predefinidos.keys())
    )
    archivo_subido = st.file_uploader("O sube tu propio CSV:", type=["csv"])

    # ── Cargar y guardar en session_state ──
    if archivo_subido is not None:
        df_cargado = cargar_csv(archivo_subido)
        if df_cargado is not None:
            st.session_state["df"] = df_cargado
            st.session_state["nombre_dataset"] = archivo_subido.name
            st.success(f"✅ Cargado: {archivo_subido.name}")

    elif dataset_elegido != "— Selecciona uno —":
        ruta = datasets_predefinidos[dataset_elegido]
        try:
            df_cargado = cargar_csv(ruta)
            if df_cargado is not None:
                st.session_state["df"] = df_cargado
                st.session_state["nombre_dataset"] = dataset_elegido
                st.success(f"✅ Dataset cargado: {dataset_elegido}")
        except Exception:
            st.warning("⚠️ No se encontró el archivo local. Sube el CSV manualmente.")

    if "df" in st.session_state:
        st.info(
            f"Dataset activo: **{st.session_state.get('nombre_dataset', 'N/A')}**\n\n"
            f"{st.session_state['df'].shape[0]:,} filas · {st.session_state['df'].shape[1]} columnas"
        )

    st.markdown("---")
    st.caption("Diploma Business Analyst · DMC Institute · 2025")


if seccion == "🏠 Home":
    st.title("📊 App Analizadora de Datasets con Streamlit")
    st.markdown("**Especialización Python for Analytics | DMC Institute · 2025**")

    st.markdown(
        """
        ---
        ### 🎯 Objetivo del Proyecto
        Esta aplicación interactiva permite **cargar, validar, procesar y visualizar** 
        cualquiera de los cuatro datasets propuestos, generando un análisis exploratorio 
        de datos (EDA) dinámico y adaptable a diferentes estructuras de datos.

        > ⚠️ **Nota de uso responsable:** Los resultados son de carácter exploratorio 
        > y no reemplazan validación técnica o profesional. El análisis del dataset 
        > de salud mental adolescente es descriptivo y no constituye diagnóstico clínico.
        ---
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗂️ Datasets Disponibles")
        datasets_info = {
            "🤖 AI Impact on Jobs 2030": "3,000 filas · 20 columnas. Mercado laboral e impacto de la IA en empleos, salarios, habilidades y demanda futura.",
            "🏪 Superstore Sales": "10,194 filas · 21 columnas. Ventas de una tienda: pedidos, regiones, categorías, descuentos y utilidad.",
            "🛒 E-commerce Risk": "12,000 filas · 23 columnas. Pedidos con variables de riesgo operativo, fraude, devoluciones y comportamiento de compra.",
            "🧠 Teen Mental Health": "1,200 filas · 13 columnas. Hábitos digitales, sueño, actividad física e indicadores de bienestar en adolescentes.",
        }
        for nombre, desc in datasets_info.items():
            with st.expander(nombre):
                st.write(desc)

    with col2:
        st.subheader("🛠️ Tecnologías Utilizadas")
        tecnologias = {
            "Python 3.x": "Lenguaje base de la aplicación",
            "Streamlit": "Framework para la interfaz interactiva",
            "Pandas": "Manipulación y análisis de datos",
            "NumPy": "Operaciones numéricas",
            "Plotly": "Gráficos interactivos",
            "Matplotlib": "Visualizaciones estáticas",
            "Seaborn": "Gráficos estadísticos",
            "GitHub": "Control de versiones y portafolio",
        }
        for tech, desc in tecnologias.items():
            st.markdown(f"- **{tech}** – {desc}")

    st.markdown("---")
    st.subheader("🔄 Flujo de la Aplicación")
    cols = st.columns(4)
    pasos = [
        ("📂", "1. Cargar", "Sube tu CSV o elige un dataset predefinido"),
        ("⚙️", "2. Procesar", "Limpieza, detección de tipos y validación"),
        ("📈", "3. Visualizar", "Gráficos univariados, bivariados y temporales"),
        ("💡", "4. Insights", "Hallazgos clave para toma de decisiones"),
    ]
    for col, (icon, titulo, desc) in zip(cols, pasos):
        with col:
            st.metric(label=f"{icon} {titulo}", value="", delta=None)
            st.caption(desc)


elif seccion == "📂 Carga y Perfil del Dataset":
    st.title("📂 Carga y Perfil del Dataset")

    if "df" not in st.session_state:
        st.warning("⚠️ Carga un dataset desde el menú lateral para continuar.")
        st.stop()

    df = st.session_state["df"]
    num_cols, cat_cols, bin_cols, date_cols = detectar_tipos(df)

    # ── Métricas rápidas ──
    st.subheader("📌 Métricas Rápidas")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Numéricas", len(num_cols))
    c4.metric("Categóricas", len(cat_cols))
    c5.metric("Nulos", int(df.isnull().sum().sum()))
    c6.metric("Duplicados", int(df.duplicated().sum()))

    st.markdown("---")

    # ── Vista previa ──
    st.subheader("👁️ Vista Previa del Dataset")
    n_filas = st.slider("Número de filas a mostrar:", 5, 50, 10)
    st.dataframe(df.head(n_filas), use_container_width=True)

    # ── Información de columnas ──
    st.subheader("🗃️ Información de Columnas")
    col_info = pd.DataFrame(
        {
            "Columna": df.columns,
            "Tipo": df.dtypes.values,
            "Nulos": df.isnull().sum().values,
            "% Nulos": (df.isnull().sum().values / len(df) * 100).round(2),
            "Únicos": df.nunique().values,
        }
    )
    st.dataframe(col_info, use_container_width=True)

    # ── Clasificación de variables ──
    st.subheader("🏷️ Clasificación de Variables")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Variables Numéricas:**")
        if num_cols:
            st.write(", ".join(num_cols))
        else:
            st.info("No se detectaron variables numéricas.")

        st.markdown("**Variables de Fecha:**")
        if date_cols:
            st.write(", ".join(date_cols))
        else:
            st.info("No se detectaron columnas de fecha.")

    with col_b:
        st.markdown("**Variables Categóricas:**")
        if cat_cols:
            st.write(", ".join(cat_cols))
        else:
            st.info("No se detectaron variables categóricas.")

        st.markdown("**Variables Binarias (dentro de numéricas):**")
        if bin_cols:
            st.write(", ".join(bin_cols))
        else:
            st.info("No se detectaron variables binarias.")

    # ── Resumen estadístico ──
    st.subheader("📊 Estadística Descriptiva")
    if st.checkbox("Mostrar estadística descriptiva completa"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ── Selección de columnas ──
    st.subheader("🔎 Explorar Columnas Específicas")
    cols_selec = st.multiselect(
        "Selecciona columnas para explorar:",
        options=df.columns.tolist(),
        default=df.columns[:5].tolist(),
    )
    if cols_selec:
        st.dataframe(df[cols_selec].head(20), use_container_width=True)


elif seccion == "⚙️ Procesamiento de Datos":
    st.title("⚙️ Procesamiento de Datos")

    if "df" not in st.session_state:
        st.warning("⚠️ Carga un dataset desde el menú lateral para continuar.")
        st.stop()

    df = st.session_state["df"].copy()
    num_cols, cat_cols, bin_cols, date_cols = detectar_tipos(df)

    # ── Conversión de fechas ──
    if date_cols:
        df = convertir_fechas(df, date_cols)
        st.success(f"✅ Columnas de fecha convertidas: {', '.join(date_cols)}")

    # ── Análisis de nulos ──
    st.subheader("🕳️ Valores Faltantes por Columna")
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    nulos_df = pd.DataFrame({"Nulos": nulos, "% Nulos": nulos_pct}).query("Nulos > 0")

    if nulos_df.empty:
        st.success("✅ El dataset no tiene valores nulos.")
    else:
        st.dataframe(nulos_df, use_container_width=True)
        fig_nulos = px.bar(
            nulos_df.reset_index(),
            x="index",
            y="% Nulos",
            title="Porcentaje de Nulos por Columna",
            color="% Nulos",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_nulos, use_container_width=True)

    # ── Duplicados ──
    st.subheader("🔁 Registros Duplicados")
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        st.success("✅ No se encontraron filas duplicadas.")
    else:
        st.warning(f"⚠️ Se encontraron {n_dup} filas duplicadas ({n_dup/len(df)*100:.2f}%).")
        if st.checkbox("Mostrar filas duplicadas"):
            st.dataframe(df[df.duplicated()], use_container_width=True)

    # ── Outliers ──
    st.subheader("🎯 Detección de Outliers (Regla IQR)")
    if num_cols:
        outlier_data = {
            col: detectar_outliers_iqr(df, col)
            for col in num_cols
            if col not in bin_cols
        }
        outlier_df = pd.DataFrame(
            outlier_data.items(), columns=["Variable", "Outliers"]
        ).sort_values("Outliers", ascending=False)
        st.dataframe(outlier_df, use_container_width=True)

        col_box = st.selectbox(
            "Visualizar outliers en boxplot:", [c for c in num_cols if c not in bin_cols]
        )
        fig_box = px.box(
            df, y=col_box, title=f"Distribución y Outliers: {col_box}", color_discrete_sequence=["#636EFA"]
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No hay variables numéricas para analizar outliers.")

    # ── Filtros dinámicos ──
    st.subheader("🔧 Filtros Dinámicos")
    df_filtrado = df.copy()

    if cat_cols:
        cat_filtro = st.selectbox("Filtrar por variable categórica:", ["— Ninguno —"] + cat_cols)
        if cat_filtro != "— Ninguno —":
            opciones = df[cat_filtro].dropna().unique().tolist()
            selec = st.multiselect(f"Valores de {cat_filtro}:", opciones, default=opciones[:3])
            df_filtrado = df_filtrado[df_filtrado[cat_filtro].isin(selec)]

    if num_cols:
        num_filtro = st.selectbox(
            "Filtrar por rango numérico:", ["— Ninguno —"] + [c for c in num_cols if c not in bin_cols]
        )
        if num_filtro != "— Ninguno —":
            min_v = float(df[num_filtro].min())
            max_v = float(df[num_filtro].max())
            rango = st.slider(f"Rango de {num_filtro}:", min_v, max_v, (min_v, max_v))
            df_filtrado = df_filtrado[df_filtrado[num_filtro].between(rango[0], rango[1])]

    st.info(f"Dataset filtrado: **{df_filtrado.shape[0]:,} filas**")
    if st.checkbox("Mostrar datos filtrados"):
        st.dataframe(df_filtrado.head(100), use_container_width=True)

    st.session_state["df_filtrado"] = df_filtrado
    st.session_state["num_cols"] = num_cols
    st.session_state["cat_cols"] = cat_cols
    st.session_state["bin_cols"] = bin_cols
    st.session_state["date_cols"] = date_cols


elif seccion == "📈 Análisis Visual":
    st.title("📈 Análisis Visual")

    if "df" not in st.session_state:
        st.warning("⚠️ Carga un dataset desde el menú lateral para continuar.")
        st.stop()

    df_orig = st.session_state["df"].copy()
    df = st.session_state.get("df_filtrado", df_orig).copy()

    # Re-detectar si no se pasó por procesamiento
    num_cols = st.session_state.get("num_cols", df.select_dtypes(include="number").columns.tolist())
    cat_cols = st.session_state.get("cat_cols", df.select_dtypes(include=["object", "category"]).columns.tolist())
    bin_cols = st.session_state.get("bin_cols", [c for c in num_cols if df[c].nunique() == 2])
    date_cols = st.session_state.get("date_cols", [])

    num_puras = [c for c in num_cols if c not in bin_cols]

    # Convertir fechas si existen
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    tabs = st.tabs([
        "📋 Resumen",
        "📊 Univariado",
        "🔗 Bivariado",
        "🌐 Multivariado",
        "📅 Temporal",
        "💡 Insights",
    ])

    with tabs[0]:
        st.subheader("📋 Resumen del Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{df.shape[0]:,}")
        c2.metric("Columnas", df.shape[1])
        c3.metric("Nulos totales", int(df.isnull().sum().sum()))
        c4.metric("Duplicados", int(df.duplicated().sum()))

        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Tipos de datos:**")
            tipo_counts = df.dtypes.value_counts().reset_index()
            tipo_counts.columns = ["Tipo", "Cantidad"]
            fig_tipos = px.pie(tipo_counts, names="Tipo", values="Cantidad", title="Distribución de Tipos de Datos")
            st.plotly_chart(fig_tipos, use_container_width=True)
        with col_r:
            if df.isnull().sum().sum() > 0:
                st.markdown("**Mapa de nulos:**")
                nulos_pct = (df.isnull().sum() / len(df) * 100).reset_index()
                nulos_pct.columns = ["Columna", "% Nulos"]
                nulos_pct = nulos_pct[nulos_pct["% Nulos"] > 0]
                fig_nulos2 = px.bar(nulos_pct, x="Columna", y="% Nulos", color="% Nulos",
                                    color_continuous_scale="Oranges", title="% de Nulos por Columna")
                st.plotly_chart(fig_nulos2, use_container_width=True)
            else:
                st.success("✅ Sin valores nulos en el dataset.")

        st.subheader("📐 Estadística Descriptiva")
        if num_puras:
            st.dataframe(df[num_puras].describe().T.round(3), use_container_width=True)
        else:
            st.info("No hay columnas numéricas para describir.")

    with tabs[1]:
        st.subheader("📊 Análisis Univariado")
        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown("#### Variables Numéricas")
            if num_puras:
                var_num = st.selectbox("Selecciona variable numérica:", num_puras, key="uni_num")
                tipo_grafico = st.radio("Tipo de gráfico:", ["Histograma", "Boxplot"], horizontal=True, key="uni_tipo")

                if tipo_grafico == "Histograma":
                    bins = st.slider("Número de bins:", 10, 100, 30, key="uni_bins")
                    fig = px.histogram(df, x=var_num, nbins=bins, title=f"Distribución de {var_num}",
                                       color_discrete_sequence=["#636EFA"])
                    fig.update_layout(bargap=0.05)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Media: {df[var_num].mean():.2f} | Mediana: {df[var_num].median():.2f} | Std: {df[var_num].std():.2f}")
                else:
                    fig = px.box(df, y=var_num, title=f"Boxplot de {var_num}", color_discrete_sequence=["#EF553B"])
                    st.plotly_chart(fig, use_container_width=True)
                    outliers_n = detectar_outliers_iqr(df, var_num)
                    st.caption(f"Outliers detectados (IQR): {outliers_n}")
            else:
                st.info("No hay variables numéricas disponibles.")

        with col_der:
            st.markdown("#### Variables Categóricas")
            if cat_cols:
                var_cat = st.selectbox("Selecciona variable categórica:", cat_cols, key="uni_cat")
                top_n = st.slider("Top N categorías:", 5, 30, 10, key="uni_topn")
                conteo = df[var_cat].value_counts().head(top_n).reset_index()
                conteo.columns = [var_cat, "Conteo"]

                tipo_cat = st.radio("Tipo de gráfico:", ["Barras", "Pie"], horizontal=True, key="uni_cat_tipo")
                if tipo_cat == "Barras":
                    fig_cat = px.bar(conteo, x=var_cat, y="Conteo", title=f"Top {top_n}: {var_cat}",
                                     color="Conteo", color_continuous_scale="Blues")
                else:
                    fig_cat = px.pie(conteo, names=var_cat, values="Conteo", title=f"Proporción: {var_cat}")
                st.plotly_chart(fig_cat, use_container_width=True)
                st.caption(f"Categorías únicas: {df[var_cat].nunique()}")
            else:
                st.info("No hay variables categóricas disponibles.")

        # Seaborn – histogramas múltiples
        if num_puras and st.checkbox("📐 Ver distribuciones múltiples (Seaborn)"):
            vars_sel = st.multiselect("Variables numéricas:", num_puras, default=num_puras[:4])
            if vars_sel:
                fig_sns, axes = plt.subplots(1, len(vars_sel), figsize=(5 * len(vars_sel), 4))
                if len(vars_sel) == 1:
                    axes = [axes]
                for ax, col in zip(axes, vars_sel):
                    sns.histplot(df[col].dropna(), ax=ax, kde=True, color="#4C72B0")
                    ax.set_title(col)
                    ax.set_xlabel("")
                plt.tight_layout()
                st.pyplot(fig_sns)

    with tabs[2]:
        st.subheader("🔗 Análisis Bivariado")

        tipo_biv = st.selectbox(
            "Tipo de análisis bivariado:",
            ["Numérico vs Numérico (Scatter)", "Numérico vs Categórico (Boxplot)", "Categórico vs Categórico (Barras agrupadas)"],
        )

        if tipo_biv == "Numérico vs Numérico (Scatter)":
            if len(num_puras) >= 2:
                c1, c2, c3 = st.columns(3)
                eje_x = c1.selectbox("Eje X:", num_puras, key="biv_x")
                eje_y = c2.selectbox("Eje Y:", num_puras, index=min(1, len(num_puras)-1), key="biv_y")
                color_var = c3.selectbox("Color por:", ["— Ninguno —"] + cat_cols, key="biv_color")
                color_arg = None if color_var == "— Ninguno —" else color_var

                sample_size = min(2000, len(df))
                df_sample = df.sample(sample_size, random_state=42)
                fig_sc = px.scatter(df_sample, x=eje_x, y=eje_y, color=color_arg,
                                    title=f"{eje_x} vs {eje_y}", opacity=0.6,
                                    trendline="ols" if color_arg is None else None)
                st.plotly_chart(fig_sc, use_container_width=True)
                corr = df[[eje_x, eje_y]].corr().iloc[0, 1]
                st.caption(f"Correlación de Pearson: **{corr:.3f}**")
            else:
                st.info("Se necesitan al menos 2 variables numéricas.")

        elif tipo_biv == "Numérico vs Categórico (Boxplot)":
            if num_puras and cat_cols:
                c1, c2 = st.columns(2)
                var_num2 = c1.selectbox("Variable numérica:", num_puras, key="biv_num")
                var_cat2 = c2.selectbox("Variable categórica:", cat_cols, key="biv_cat2")
                top_cat = st.slider("Top N categorías:", 3, 20, 8, key="biv_topcat")
                top_vals = df[var_cat2].value_counts().head(top_cat).index
                df_biv = df[df[var_cat2].isin(top_vals)]
                fig_biv = px.box(df_biv, x=var_cat2, y=var_num2, color=var_cat2,
                                 title=f"{var_num2} por {var_cat2}")
                st.plotly_chart(fig_biv, use_container_width=True)

                # Seaborn complementario
                if st.checkbox("Ver versión Seaborn"):
                    fig_s, ax = plt.subplots(figsize=(10, 5))
                    sns.boxplot(data=df_biv, x=var_cat2, y=var_num2, palette="Set2", ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
                    ax.set_title(f"{var_num2} por {var_cat2}")
                    plt.tight_layout()
                    st.pyplot(fig_s)
            else:
                st.info("Se necesita al menos una variable numérica y una categórica.")

        else:  # Categórico vs Categórico
            if len(cat_cols) >= 2:
                c1, c2 = st.columns(2)
                cat1 = c1.selectbox("Variable 1:", cat_cols, key="biv_cc1")
                cat2 = c2.selectbox("Variable 2:", cat_cols, index=min(1, len(cat_cols)-1), key="biv_cc2")
                top_n_cc = st.slider("Top N (var 1):", 3, 15, 6, key="biv_topcc")
                top_vals1 = df[cat1].value_counts().head(top_n_cc).index
                df_cc = df[df[cat1].isin(top_vals1)]
                ct = df_cc.groupby([cat1, cat2]).size().reset_index(name="Conteo")
                fig_cc = px.bar(ct, x=cat1, y="Conteo", color=cat2, barmode="group",
                                title=f"{cat1} vs {cat2}")
                st.plotly_chart(fig_cc, use_container_width=True)
            else:
                st.info("Se necesitan al menos 2 variables categóricas.")


    with tabs[3]:
        st.subheader("🌐 Análisis Multivariado")

        sub_multi = st.radio(
            "Selecciona análisis:",
            ["Heatmap de Correlación", "Barras Apiladas", "Scatter 3D"],
            horizontal=True,
        )

        if sub_multi == "Heatmap de Correlación":
            if len(num_puras) >= 2:
                vars_corr = st.multiselect("Variables para correlación:", num_puras, default=num_puras[:8])
                if len(vars_corr) >= 2:
                    corr_mat = df[vars_corr].corr().round(3)

                    # Seaborn heatmap
                    fig_h, ax = plt.subplots(figsize=(max(8, len(vars_corr)), max(6, len(vars_corr) - 1)))
                    sns.heatmap(corr_mat, annot=True, cmap="coolwarm", center=0,
                                fmt=".2f", linewidths=0.5, ax=ax)
                    ax.set_title("Matriz de Correlación")
                    plt.tight_layout()
                    st.pyplot(fig_h)
                    st.caption("Valores cercanos a 1 o -1 indican alta correlación positiva o negativa.")
                else:
                    st.info("Selecciona al menos 2 variables.")
            else:
                st.info("No hay suficientes variables numéricas.")

        elif sub_multi == "Barras Apiladas":
            if cat_cols and num_puras:
                c1, c2, c3 = st.columns(3)
                cat_ap = c1.selectbox("Eje X (categoría):", cat_cols, key="multi_cat")
                num_ap = c2.selectbox("Valor (numérico):", num_puras, key="multi_num")
                color_ap = c3.selectbox("Color (categoría):", ["— Ninguno —"] + cat_cols, key="multi_col")
                top_ap = st.slider("Top N categorías:", 3, 20, 8)
                top_v = df[cat_ap].value_counts().head(top_ap).index
                df_ap = df[df[cat_ap].isin(top_v)]

                if color_ap == "— Ninguno —":
                    agg = df_ap.groupby(cat_ap)[num_ap].mean().reset_index()
                    fig_ap = px.bar(agg, x=cat_ap, y=num_ap, title=f"Promedio de {num_ap} por {cat_ap}")
                else:
                    agg2 = df_ap.groupby([cat_ap, color_ap])[num_ap].mean().reset_index()
                    fig_ap = px.bar(agg2, x=cat_ap, y=num_ap, color=color_ap, barmode="stack",
                                    title=f"{num_ap} por {cat_ap} y {color_ap}")
                st.plotly_chart(fig_ap, use_container_width=True)

        else:  # Scatter 3D
            if len(num_puras) >= 3:
                c1, c2, c3, c4 = st.columns(4)
                x3 = c1.selectbox("X:", num_puras, key="s3x")
                y3 = c2.selectbox("Y:", num_puras, index=1, key="s3y")
                z3 = c3.selectbox("Z:", num_puras, index=2, key="s3z")
                col3 = c4.selectbox("Color:", ["— Ninguno —"] + cat_cols, key="s3c")
                df_s3 = df.sample(min(2000, len(df)), random_state=42)
                fig3d = px.scatter_3d(df_s3, x=x3, y=y3, z=z3,
                                      color=None if col3 == "— Ninguno —" else col3,
                                      title="Scatter 3D", opacity=0.6, height=600)
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("Se necesitan al menos 3 variables numéricas para el Scatter 3D.")

    with tabs[4]:
        st.subheader("📅 Análisis Temporal")

        if not date_cols:
            # Intentar detectar automáticamente
            posibles = [c for c in df.columns if "date" in c or "fecha" in c or "time" in c]
            if posibles:
                date_cols = posibles
                for col in date_cols:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                st.info("ℹ️ No se detectaron columnas de fecha en este dataset.")
                st.stop()

        col_fecha = st.selectbox("Columna de fecha:", date_cols, key="temp_fecha")
        df_t = df.dropna(subset=[col_fecha]).copy()
        df_t[col_fecha] = pd.to_datetime(df_t[col_fecha], errors="coerce")

        granularidad = st.radio("Granularidad:", ["Mes", "Trimestre", "Año"], horizontal=True)
        freq_map = {"Mes": "ME", "Trimestre": "QE", "Año": "YE"}

        if num_puras:
            var_temp = st.selectbox("Variable a analizar:", num_puras, key="temp_var")
            serie = df_t.set_index(col_fecha)[var_temp].resample(freq_map[granularidad]).mean().reset_index()
            serie.columns = [col_fecha, "Valor"]

            fig_temp = px.line(serie, x=col_fecha, y="Valor",
                               title=f"Evolución de {var_temp} por {granularidad}",
                               markers=True)
            fig_temp.update_traces(line_color="#00CC96", line_width=2)
            st.plotly_chart(fig_temp, use_container_width=True)
            st.caption(f"Mínimo: {serie['Valor'].min():.2f} | Máximo: {serie['Valor'].max():.2f} | Promedio: {serie['Valor'].mean():.2f}")

        if cat_cols:
            st.markdown("---")
            cat_temp = st.selectbox("Conteo por categoría a lo largo del tiempo:", ["— Ninguno —"] + cat_cols)
            if cat_temp != "— Ninguno —":
                df_tc = df_t.copy()
                df_tc["periodo"] = df_tc[col_fecha].dt.to_period(
                    "M" if granularidad == "Mes" else ("Q" if granularidad == "Trimestre" else "Y")
                ).astype(str)
                top_v = df_tc[cat_temp].value_counts().head(6).index
                df_tc = df_tc[df_tc[cat_temp].isin(top_v)]
                conteo_tc = df_tc.groupby(["periodo", cat_temp]).size().reset_index(name="Conteo")
                fig_tc = px.line(conteo_tc, x="periodo", y="Conteo", color=cat_temp,
                                 title=f"Conteo de {cat_temp} por {granularidad}", markers=True)
                st.plotly_chart(fig_tc, use_container_width=True)

    with tabs[5]:
        st.subheader("💡 Insights y Hallazgos Clave")

        nombre_ds = st.session_state.get("nombre_dataset", "Dataset cargado")
        st.markdown(f"**Dataset analizado:** {nombre_ds}")
        st.markdown(f"**Total de registros:** {df.shape[0]:,} filas · {df.shape[1]} columnas")

        st.markdown("---")
        col_ins1, col_ins2 = st.columns(2)

        with col_ins1:
            st.markdown("#### 📌 Estadísticos Clave")
            if num_puras:
                for col in num_puras[:5]:
                    st.metric(
                        label=col,
                        value=f"{df[col].mean():.2f}",
                        delta=f"std: {df[col].std():.2f}",
                    )

        with col_ins2:
            st.markdown("#### 📌 Categorías Más Frecuentes")
            if cat_cols:
                for col in cat_cols[:4]:
                    top1 = df[col].value_counts().idxmax()
                    pct = df[col].value_counts(normalize=True).max() * 100
                    st.metric(label=col, value=top1, delta=f"{pct:.1f}%")

        st.markdown("---")
        st.markdown("#### 🔍 Correlaciones Destacadas")
        if len(num_puras) >= 2:
            corr_flat = (
                df[num_puras].corr()
                .unstack()
                .reset_index()
            )
            corr_flat.columns = ["Var1", "Var2", "Correlación"]
            corr_flat = corr_flat[corr_flat["Var1"] < corr_flat["Var2"]].sort_values(
                "Correlación", key=abs, ascending=False
            )
            st.dataframe(corr_flat.head(10).reset_index(drop=True), use_container_width=True)
        else:
            st.info("No hay suficientes variables numéricas para calcular correlaciones.")

        st.markdown("---")
        st.markdown("#### 📝 Notas Interpretativas")
        st.info(
            "Los gráficos y métricas anteriores son de carácter **exploratorio**. "
            "Los patrones identificados deben ser validados con técnicas estadísticas formales "
            "antes de tomar decisiones de negocio. Este análisis no reemplaza la validación técnica o profesional."
        )

        if st.checkbox("📥 Ver datos actuales (muestra)"):
            st.dataframe(df.head(50), use_container_width=True)
