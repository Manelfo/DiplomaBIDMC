import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="App Analizadora de Datasets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  FUNCIONES UTILITARIAS
# ─────────────────────────────────────────────
def cargar_csv(archivo):
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
    numericas = df.select_dtypes(include=["number"]).columns.tolist()
    categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    binarias = [c for c in numericas if df[c].nunique() == 2]
    fechas = []
    for col in df.columns:
        if "date" in col or "fecha" in col or "time" in col:
            try:
                pd.to_datetime(df[col].dropna().head(20), errors="raise")
                fechas.append(col)
            except Exception:
                pass
    return numericas, categoricas, binarias, fechas


def convertir_fechas(df, cols_fecha):
    for col in cols_fecha:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def detectar_outliers_iqr(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    return df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)].shape[0]


def df_a_str(df):
    """Convierte todas las columnas a tipos compatibles con Arrow/PyArrow."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("string"):
            df[col] = df[col].astype(str)
    return df


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Analizadora de Datasets")
    st.markdown("**Renato Robello**")
    st.markdown("*Diploma Business Analyst · 2025*")
    st.markdown("---")

    seccion = st.selectbox(
        "📌 Navegar a:",
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

    dataset_elegido = st.selectbox("Dataset predefinido:", list(datasets_predefinidos.keys()))
    archivo_subido = st.file_uploader("O sube tu propio CSV:", type=["csv"])

    if archivo_subido is not None:
        df_cargado = cargar_csv(archivo_subido)
        if df_cargado is not None:
            st.session_state["df"] = df_cargado
            st.session_state["nombre_dataset"] = archivo_subido.name
            st.success(f"✅ {archivo_subido.name}")
    elif dataset_elegido != "— Selecciona uno —":
        ruta = datasets_predefinidos[dataset_elegido]
        try:
            df_cargado = cargar_csv(ruta)
            if df_cargado is not None:
                st.session_state["df"] = df_cargado
                st.session_state["nombre_dataset"] = dataset_elegido
                st.success(f"✅ {dataset_elegido}")
        except Exception:
            st.warning("⚠️ Sube el CSV manualmente.")

    if "df" in st.session_state:
        st.info(
            f"**Activo:** {st.session_state.get('nombre_dataset','')}\n\n"
            f"{st.session_state['df'].shape[0]:,} filas · {st.session_state['df'].shape[1]} cols"
        )

    st.markdown("---")
    st.caption("DMC Institute · Python for Analytics")


# ─────────────────────────────────────────────
#  SECCIÓN 1: HOME
# ─────────────────────────────────────────────
if seccion == "🏠 Home":
    st.title("📊 App Analizadora de Datasets con Streamlit")
    st.markdown("### Renato Robello · Diploma Business Analyst · DMC Institute · 2025")
    st.markdown("---")

    st.markdown(
        """
        ### 🎯 Objetivo del Proyecto
        Esta aplicación interactiva permite **cargar, validar, procesar y visualizar** 
        cualquiera de los cuatro datasets propuestos, generando un análisis exploratorio 
        de datos (EDA) dinámico adaptable a diferentes estructuras.

        > ⚠️ **Nota de uso responsable:** Los resultados son exploratorios y no reemplazan 
        > validación técnica o profesional. El análisis de salud mental adolescente es 
        > descriptivo y no constituye diagnóstico clínico.
        ---
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗂️ Datasets Disponibles")
        datasets_info = {
            "🤖 AI Impact on Jobs 2030": "3,000 filas · 20 cols. Mercado laboral e impacto de IA en empleos, salarios y demanda futura.",
            "🏪 Superstore Sales": "10,194 filas · 21 cols. Ventas: pedidos, regiones, categorías, descuentos y utilidad.",
            "🛒 E-commerce Risk": "12,000 filas · 23 cols. Pedidos con riesgo operativo, fraude y devoluciones.",
            "🧠 Teen Mental Health": "1,200 filas · 13 cols. Hábitos digitales, sueño y bienestar en adolescentes.",
        }
        for nombre, desc in datasets_info.items():
            with st.expander(nombre):
                st.write(desc)

    with col2:
        st.subheader("🛠️ Tecnologías Utilizadas")
        techs = [
            ("Python 3.x", "Lenguaje base"),
            ("Streamlit", "Interfaz interactiva"),
            ("Pandas", "Manipulación de datos"),
            ("NumPy", "Operaciones numéricas"),
            ("Plotly", "Gráficos interactivos"),
            ("Matplotlib", "Visualizaciones estáticas"),
            ("Seaborn", "Gráficos estadísticos"),
            ("GitHub", "Control de versiones"),
        ]
        for tech, desc in techs:
            st.markdown(f"- **{tech}** – {desc}")

    st.markdown("---")
    st.subheader("🔄 Flujo de la Aplicación")
    cols = st.columns(4)
    pasos = [
        ("📂", "1. Cargar", "Sube tu CSV o elige un dataset"),
        ("⚙️", "2. Procesar", "Limpieza y validación"),
        ("📈", "3. Visualizar", "Gráficos interactivos"),
        ("💡", "4. Insights", "Hallazgos para decisiones"),
    ]
    for col, (icon, titulo, desc) in zip(cols, pasos):
        with col:
            st.markdown(f"### {icon} {titulo}")
            st.caption(desc)


# ─────────────────────────────────────────────
#  SECCIÓN 2: CARGA Y PERFIL
# ─────────────────────────────────────────────
elif seccion == "📂 Carga y Perfil del Dataset":
    st.title("📂 Carga y Perfil del Dataset")

    if "df" not in st.session_state:
        st.warning("⚠️ Carga un dataset desde el menú lateral para continuar.")
        st.stop()

    df = st.session_state["df"]
    num_cols, cat_cols, bin_cols, date_cols = detectar_tipos(df)

    st.subheader("📌 Métricas Rápidas")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Numéricas", len(num_cols))
    c4.metric("Categóricas", len(cat_cols))
    c5.metric("Nulos", int(df.isnull().sum().sum()))
    c6.metric("Duplicados", int(df.duplicated().sum()))

    st.markdown("---")
    st.subheader("👁️ Vista Previa")
    n_filas = st.slider("Filas a mostrar:", 5, 50, 10)
    st.dataframe(df_a_str(df.head(n_filas)), width='stretch')

    st.subheader("🗃️ Información de Columnas")
    col_info = pd.DataFrame({
        "Columna": df.columns.tolist(),
        "Tipo": [str(t) for t in df.dtypes.values],
        "Nulos": df.isnull().sum().values,
        "% Nulos": (df.isnull().sum().values / len(df) * 100).round(2),
        "Únicos": df.nunique().values,
    })
    st.dataframe(col_info, width='stretch')

    st.subheader("🏷️ Clasificación de Variables")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Numéricas:**")
        st.write(", ".join(num_cols) if num_cols else "Ninguna")
        st.markdown("**Fechas:**")
        st.write(", ".join(date_cols) if date_cols else "Ninguna")
    with col_b:
        st.markdown("**Categóricas:**")
        st.write(", ".join(cat_cols) if cat_cols else "Ninguna")
        st.markdown("**Binarias:**")
        st.write(", ".join(bin_cols) if bin_cols else "Ninguna")

    if st.checkbox("Mostrar estadística descriptiva completa"):
        desc = df.describe(include="all").T
        desc.index = desc.index.astype(str)
        desc.columns = desc.columns.astype(str)
        st.dataframe(desc.round(3), width='stretch')

    st.subheader("🔎 Explorar Columnas Específicas")
    cols_selec = st.multiselect("Columnas:", df.columns.tolist(), default=df.columns[:5].tolist())
    if cols_selec:
        st.dataframe(df_a_str(df[cols_selec].head(20)), width='stretch')


# ─────────────────────────────────────────────
#  SECCIÓN 3: PROCESAMIENTO
# ─────────────────────────────────────────────
elif seccion == "⚙️ Procesamiento de Datos":
    st.title("⚙️ Procesamiento de Datos")

    if "df" not in st.session_state:
        st.warning("⚠️ Carga un dataset desde el menú lateral para continuar.")
        st.stop()

    df = st.session_state["df"].copy()
    num_cols, cat_cols, bin_cols, date_cols = detectar_tipos(df)

    if date_cols:
        df = convertir_fechas(df, date_cols)
        st.success(f"✅ Fechas convertidas: {', '.join(date_cols)}")

    # Nulos
    st.subheader("🕳️ Valores Faltantes")
    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    nulos_df = pd.DataFrame({"Nulos": nulos, "% Nulos": nulos_pct}).query("Nulos > 0")

    if nulos_df.empty:
        st.success("✅ Sin valores nulos.")
    else:
        st.dataframe(nulos_df, width='stretch')
        fig_nulos = px.bar(
            nulos_df.reset_index().rename(columns={"index": "Columna"}),
            x="Columna", y="% Nulos", color="% Nulos",
            color_continuous_scale="Reds", title="% de Nulos por Columna"
        )
        st.plotly_chart(fig_nulos, width='stretch')

    # Duplicados
    st.subheader("🔁 Duplicados")
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        st.success("✅ Sin filas duplicadas.")
    else:
        st.warning(f"⚠️ {n_dup} filas duplicadas ({n_dup/len(df)*100:.2f}%).")
        if st.checkbox("Ver duplicados"):
            st.dataframe(df_a_str(df[df.duplicated()]), width='stretch')

    # Outliers
    st.subheader("🎯 Detección de Outliers (IQR)")
    num_puras = [c for c in num_cols if c not in bin_cols]
    if num_puras:
        outlier_data = {col: detectar_outliers_iqr(df, col) for col in num_puras}
        outlier_df = pd.DataFrame(outlier_data.items(), columns=["Variable", "Outliers"]).sort_values("Outliers", ascending=False)
        st.dataframe(outlier_df, width='stretch')

        col_box = st.selectbox("Ver boxplot de:", num_puras)
        fig_box = px.box(df, y=col_box, title=f"Outliers: {col_box}", color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig_box, width='stretch')
    else:
        st.info("No hay variables numéricas.")

    # Filtros
    st.subheader("🔧 Filtros Dinámicos")
    df_filtrado = df.copy()

    if cat_cols:
        cat_filtro = st.selectbox("Filtrar por categórica:", ["— Ninguno —"] + cat_cols)
        if cat_filtro != "— Ninguno —":
            opciones = df[cat_filtro].dropna().unique().tolist()
            selec = st.multiselect(f"Valores de {cat_filtro}:", opciones, default=opciones[:3])
            if selec:
                df_filtrado = df_filtrado[df_filtrado[cat_filtro].isin(selec)]

    if num_puras:
        num_filtro = st.selectbox("Filtrar por rango numérico:", ["— Ninguno —"] + num_puras)
        if num_filtro != "— Ninguno —":
            min_v = float(df[num_filtro].min())
            max_v = float(df[num_filtro].max())
            rango = st.slider(f"Rango de {num_filtro}:", min_v, max_v, (min_v, max_v))
            df_filtrado = df_filtrado[df_filtrado[num_filtro].between(rango[0], rango[1])]

    st.info(f"Dataset filtrado: **{df_filtrado.shape[0]:,} filas**")
    if st.checkbox("Ver datos filtrados"):
        st.dataframe(df_a_str(df_filtrado.head(100)), width='stretch')

    st.session_state["df_filtrado"] = df_filtrado
    st.session_state["num_cols"] = num_cols
    st.session_state["cat_cols"] = cat_cols
    st.session_state["bin_cols"] = bin_cols
    st.session_state["date_cols"] = date_cols


# ─────────────────────────────────────────────
#  SECCIÓN 4: ANÁLISIS VISUAL
# ─────────────────────────────────────────────
elif seccion == "📈 Análisis Visual":
    st.title("📈 Análisis Visual")

    if "df" not in st.session_state:
        st.warning("⚠️ Carga un dataset desde el menú lateral para continuar.")
        st.stop()

    df_orig = st.session_state["df"].copy()
    df = st.session_state.get("df_filtrado", df_orig).copy()

    num_cols = st.session_state.get("num_cols", df.select_dtypes(include="number").columns.tolist())
    cat_cols = st.session_state.get("cat_cols", df.select_dtypes(include=["object", "category"]).columns.tolist())
    bin_cols = st.session_state.get("bin_cols", [c for c in num_cols if df[c].nunique() == 2])
    date_cols = st.session_state.get("date_cols", [])
    num_puras = [c for c in num_cols if c not in bin_cols]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    tabs = st.tabs(["📋 Resumen", "📊 Univariado", "🔗 Bivariado", "🌐 Multivariado", "📅 Temporal", "💡 Insights"])

    # ── TAB 1: RESUMEN ──
    with tabs[0]:
        st.subheader("📋 Resumen del Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{df.shape[0]:,}")
        c2.metric("Columnas", df.shape[1])
        c3.metric("Nulos totales", int(df.isnull().sum().sum()))
        c4.metric("Duplicados", int(df.duplicated().sum()))

        col_l, col_r = st.columns(2)
        with col_l:
            # Convertir dtypes a string para plotly
            tipo_counts = df.dtypes.astype(str).value_counts().reset_index()
            tipo_counts.columns = ["Tipo", "Cantidad"]
            fig_tipos = px.pie(tipo_counts, names="Tipo", values="Cantidad", title="Tipos de Datos")
            st.plotly_chart(fig_tipos, width='stretch')

        with col_r:
            nulos_pct2 = (df.isnull().sum() / len(df) * 100).reset_index()
            nulos_pct2.columns = ["Columna", "% Nulos"]
            nulos_pct2 = nulos_pct2[nulos_pct2["% Nulos"] > 0]
            if not nulos_pct2.empty:
                fig_n2 = px.bar(nulos_pct2, x="Columna", y="% Nulos", color="% Nulos",
                                color_continuous_scale="Oranges", title="% Nulos por Columna")
                st.plotly_chart(fig_n2, width='stretch')
            else:
                st.success("✅ Sin valores nulos.")

        if num_puras:
            st.subheader("📐 Estadística Descriptiva")
            desc = df[num_puras].describe().T.round(3)
            st.dataframe(desc, width='stretch')

    # ── TAB 2: UNIVARIADO ──
    with tabs[1]:
        st.subheader("📊 Análisis Univariado")
        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown("#### Numéricas")
            if num_puras:
                var_num = st.selectbox("Variable numérica:", num_puras, key="uni_num")
                tipo_g = st.radio("Gráfico:", ["Histograma", "Boxplot"], horizontal=True)
                if tipo_g == "Histograma":
                    bins = st.slider("Bins:", 10, 100, 30)
                    fig = px.histogram(df, x=var_num, nbins=bins, title=f"Distribución: {var_num}",
                                       color_discrete_sequence=["#636EFA"])
                    st.plotly_chart(fig, width='stretch')
                    st.caption(f"Media: {df[var_num].mean():.2f} | Mediana: {df[var_num].median():.2f} | Std: {df[var_num].std():.2f}")
                else:
                    fig = px.box(df, y=var_num, title=f"Boxplot: {var_num}", color_discrete_sequence=["#EF553B"])
                    st.plotly_chart(fig, width='stretch')
                    st.caption(f"Outliers (IQR): {detectar_outliers_iqr(df, var_num)}")
            else:
                st.info("Sin variables numéricas.")

        with col_der:
            st.markdown("#### Categóricas")
            if cat_cols:
                var_cat = st.selectbox("Variable categórica:", cat_cols, key="uni_cat")
                top_n = st.slider("Top N categorías:", 5, 30, 10)
                conteo = df[var_cat].value_counts().head(top_n).reset_index()
                conteo.columns = [var_cat, "Conteo"]
                tipo_cat = st.radio("Gráfico:", ["Barras", "Pie"], horizontal=True, key="uni_cat_g")
                if tipo_cat == "Barras":
                    fig_cat = px.bar(conteo, x=var_cat, y="Conteo", color="Conteo",
                                     color_continuous_scale="Blues", title=f"Top {top_n}: {var_cat}")
                else:
                    fig_cat = px.pie(conteo, names=var_cat, values="Conteo", title=f"Proporción: {var_cat}")
                st.plotly_chart(fig_cat, width='stretch')
            else:
                st.info("Sin variables categóricas.")

        if num_puras and st.checkbox("📐 Histogramas múltiples (Seaborn)"):
            vars_sel = st.multiselect("Variables:", num_puras, default=num_puras[:4])
            if vars_sel:
                fig_sns, axes = plt.subplots(1, len(vars_sel), figsize=(5 * len(vars_sel), 4))
                if len(vars_sel) == 1:
                    axes = [axes]
                for ax, col in zip(axes, vars_sel):
                    sns.histplot(df[col].dropna(), ax=ax, kde=True, color="#4C72B0")
                    ax.set_title(col)
                plt.tight_layout()
                st.pyplot(fig_sns)

    # ── TAB 3: BIVARIADO ──
    with tabs[2]:
        st.subheader("🔗 Análisis Bivariado")
        tipo_biv = st.selectbox("Tipo:", [
            "Numérico vs Numérico (Scatter)",
            "Numérico vs Categórico (Boxplot)",
            "Categórico vs Categórico (Barras agrupadas)"
        ])

        if tipo_biv == "Numérico vs Numérico (Scatter)":
            if len(num_puras) >= 2:
                c1, c2, c3 = st.columns(3)
                eje_x = c1.selectbox("Eje X:", num_puras, key="bx")
                eje_y = c2.selectbox("Eje Y:", num_puras, index=min(1, len(num_puras)-1), key="by")
                color_v = c3.selectbox("Color:", ["— Ninguno —"] + cat_cols, key="bc")
                df_s = df.sample(min(2000, len(df)), random_state=42)
                fig_sc = px.scatter(df_s, x=eje_x, y=eje_y,
                                    color=None if color_v == "— Ninguno —" else color_v,
                                    title=f"{eje_x} vs {eje_y}", opacity=0.6)
                st.plotly_chart(fig_sc, width='stretch')
                corr = df[[eje_x, eje_y]].corr().iloc[0, 1]
                st.caption(f"Correlación de Pearson: **{corr:.3f}**")
            else:
                st.info("Se necesitan ≥2 variables numéricas.")

        elif tipo_biv == "Numérico vs Categórico (Boxplot)":
            if num_puras and cat_cols:
                c1, c2 = st.columns(2)
                var_n = c1.selectbox("Numérica:", num_puras, key="bn")
                var_c = c2.selectbox("Categórica:", cat_cols, key="bc2")
                top_c = st.slider("Top N categorías:", 3, 20, 8)
                top_v = df[var_c].value_counts().head(top_c).index
                df_biv = df[df[var_c].isin(top_v)]
                fig_biv = px.box(df_biv, x=var_c, y=var_n, color=var_c, title=f"{var_n} por {var_c}")
                st.plotly_chart(fig_biv, width='stretch')
                if st.checkbox("Ver versión Seaborn"):
                    fig_s, ax = plt.subplots(figsize=(10, 4))
                    sns.boxplot(data=df_biv, x=var_c, y=var_n, palette="Set2", ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_s)
            else:
                st.info("Se necesita ≥1 numérica y ≥1 categórica.")

        else:
            if len(cat_cols) >= 2:
                c1, c2 = st.columns(2)
                cat1 = c1.selectbox("Variable 1:", cat_cols, key="cc1")
                cat2 = c2.selectbox("Variable 2:", cat_cols, index=min(1, len(cat_cols)-1), key="cc2")
                top_cc = st.slider("Top N (var 1):", 3, 15, 6)
                top_v1 = df[cat1].value_counts().head(top_cc).index
                ct = df[df[cat1].isin(top_v1)].groupby([cat1, cat2]).size().reset_index(name="Conteo")
                fig_cc = px.bar(ct, x=cat1, y="Conteo", color=cat2, barmode="group",
                                title=f"{cat1} vs {cat2}")
                st.plotly_chart(fig_cc, width='stretch')
            else:
                st.info("Se necesitan ≥2 variables categóricas.")

    # ── TAB 4: MULTIVARIADO ──
    with tabs[3]:
        st.subheader("🌐 Análisis Multivariado")
        sub = st.radio("Análisis:", ["Heatmap de Correlación", "Barras Apiladas", "Scatter 3D"], horizontal=True)

        if sub == "Heatmap de Correlación":
            if len(num_puras) >= 2:
                vars_c = st.multiselect("Variables:", num_puras, default=num_puras[:8])
                if len(vars_c) >= 2:
                    corr_m = df[vars_c].corr().round(3)
                    fig_h, ax = plt.subplots(figsize=(max(8, len(vars_c)), max(6, len(vars_c)-1)))
                    sns.heatmap(corr_m, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                                linewidths=0.5, ax=ax)
                    ax.set_title("Matriz de Correlación")
                    plt.tight_layout()
                    st.pyplot(fig_h)
                    st.caption("Valores cercanos a ±1 indican alta correlación.")
            else:
                st.info("No hay suficientes variables numéricas.")

        elif sub == "Barras Apiladas":
            if cat_cols and num_puras:
                c1, c2, c3 = st.columns(3)
                cat_ap = c1.selectbox("Eje X:", cat_cols, key="mcat")
                num_ap = c2.selectbox("Valor:", num_puras, key="mnum")
                col_ap = c3.selectbox("Color:", ["— Ninguno —"] + cat_cols, key="mcol")
                top_ap = st.slider("Top N:", 3, 20, 8)
                top_v = df[cat_ap].value_counts().head(top_ap).index
                df_ap = df[df[cat_ap].isin(top_v)]
                if col_ap == "— Ninguno —":
                    agg = df_ap.groupby(cat_ap)[num_ap].mean().reset_index()
                    fig_ap = px.bar(agg, x=cat_ap, y=num_ap, title=f"Promedio {num_ap} por {cat_ap}")
                else:
                    agg2 = df_ap.groupby([cat_ap, col_ap])[num_ap].mean().reset_index()
                    fig_ap = px.bar(agg2, x=cat_ap, y=num_ap, color=col_ap, barmode="stack",
                                    title=f"{num_ap} por {cat_ap} y {col_ap}")
                st.plotly_chart(fig_ap, width='stretch')

        else:
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
                st.plotly_chart(fig3d, width='stretch')
            else:
                st.info("Se necesitan ≥3 variables numéricas.")

    # ── TAB 5: TEMPORAL ──
    with tabs[4]:
        st.subheader("📅 Análisis Temporal")
        posibles_fechas = date_cols if date_cols else [c for c in df.columns if "date" in c or "time" in c]

        if not posibles_fechas:
            st.info("ℹ️ No se detectaron columnas de fecha en este dataset.")
        else:
            for col in posibles_fechas:
                df[col] = pd.to_datetime(df[col], errors="coerce")

            col_fecha = st.selectbox("Columna de fecha:", posibles_fechas)
            df_t = df.dropna(subset=[col_fecha]).copy()
            granularidad = st.radio("Granularidad:", ["Mes", "Trimestre", "Año"], horizontal=True)
            freq_map = {"Mes": "ME", "Trimestre": "QE", "Año": "YE"}

            if num_puras:
                var_temp = st.selectbox("Variable a analizar:", num_puras)
                serie = (df_t.set_index(col_fecha)[var_temp]
                         .resample(freq_map[granularidad]).mean().reset_index())
                serie.columns = [col_fecha, "Valor"]
                fig_temp = px.line(serie, x=col_fecha, y="Valor", markers=True,
                                   title=f"Evolución de {var_temp} por {granularidad}")
                fig_temp.update_traces(line_color="#00CC96", line_width=2)
                st.plotly_chart(fig_temp, width='stretch')
                st.caption(f"Mín: {serie['Valor'].min():.2f} | Máx: {serie['Valor'].max():.2f} | Prom: {serie['Valor'].mean():.2f}")

            if cat_cols:
                cat_temp = st.selectbox("Conteo por categoría:", ["— Ninguno —"] + cat_cols)
                if cat_temp != "— Ninguno —":
                    df_tc = df_t.copy()
                    freq_p = {"Mes": "M", "Trimestre": "Q", "Año": "Y"}
                    df_tc["periodo"] = df_tc[col_fecha].dt.to_period(freq_p[granularidad]).astype(str)
                    top_v = df_tc[cat_temp].value_counts().head(6).index
                    ct2 = df_tc[df_tc[cat_temp].isin(top_v)].groupby(["periodo", cat_temp]).size().reset_index(name="Conteo")
                    fig_tc = px.line(ct2, x="periodo", y="Conteo", color=cat_temp,
                                     title=f"{cat_temp} a lo largo del tiempo", markers=True)
                    st.plotly_chart(fig_tc, width='stretch')

    # ── TAB 6: INSIGHTS ──
    with tabs[5]:
        st.subheader("💡 Insights y Hallazgos Clave")
        nombre_ds = st.session_state.get("nombre_dataset", "Dataset cargado")
        st.markdown(f"**Dataset:** {nombre_ds} · **{df.shape[0]:,} filas · {df.shape[1]} columnas**")
        st.markdown("---")

        col_ins1, col_ins2 = st.columns(2)
        with col_ins1:
            st.markdown("#### 📌 Estadísticos Clave")
            if num_puras:
                for col in num_puras[:5]:
                    st.metric(col, f"{df[col].mean():.2f}", delta=f"std: {df[col].std():.2f}")
        with col_ins2:
            st.markdown("#### 📌 Categorías Más Frecuentes")
            if cat_cols:
                for col in cat_cols[:4]:
                    top1 = df[col].value_counts().idxmax()
                    pct = df[col].value_counts(normalize=True).max() * 100
                    st.metric(col, str(top1), delta=f"{pct:.1f}%")

        st.markdown("---")
        st.markdown("#### 🔍 Top 10 Correlaciones")
        if len(num_puras) >= 2:
            corr_flat = df[num_puras].corr().unstack().reset_index()
            corr_flat.columns = ["Var1", "Var2", "Correlación"]
            corr_flat = (corr_flat[corr_flat["Var1"] < corr_flat["Var2"]]
                         .sort_values("Correlación", key=abs, ascending=False))
            st.dataframe(corr_flat.head(10).reset_index(drop=True), width='stretch')
        else:
            st.info("Sin suficientes variables numéricas.")

        st.markdown("---")
        st.info(
            "📝 Los resultados son **exploratorios**. Los patrones identificados deben ser "
            "validados con técnicas estadísticas formales antes de tomar decisiones de negocio."
        )
        if st.checkbox("📥 Ver muestra del dataset"):
            st.dataframe(df_a_str(df.head(50)), width='stretch')
