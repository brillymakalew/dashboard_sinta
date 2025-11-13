import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ------------------------------
# Konstanta & Mapping
# ------------------------------

CATEGORY_LABELS = {
    "Score in Publication": "Publication",
    "Score in HKI": "HKI",
    "Score in Research": "Research",
    "Score in Community Service": "Community Service",
    "Score in SDM": "SDM",
    "Score in Kelembagaan": "Kelembagaan",
}

CATEGORY_ORDER = [
    "Publication",
    "Research",
    "Community Service",
    "HKI",
    "SDM",
    "Kelembagaan",
]


# ------------------------------
# Fungsi Load & Preprocess Data
# ------------------------------

@st.cache_data
def load_cluster_data(file):
    """Load Sinta Metric Cluster.xlsx dan siapkan afiliasi + detail kode."""
    xls = pd.read_excel(file, sheet_name=None)

    df_af = xls["afiliasi"].copy()
    df_detail = xls["detail_kode"].copy()

    # Tambahkan label kategori yang lebih simple
    df_detail["category"] = df_detail["kategori_score"].map(CATEGORY_LABELS)

    # Agregasi total per kategori per afiliasi
    cat_pivot = (
        df_detail
        .groupby(["nama_afiliasi", "category"])["total"]
        .sum()
        .unstack(fill_value=0)
    )

    # Hitung rank nasional berdasarkan sinta_score_overall
    af_ranked = df_af.sort_values("sinta_score_overall", ascending=False).reset_index(drop=True)
    af_ranked["rank_overall"] = af_ranked.index + 1

    df_af = df_af.merge(
        af_ranked[["nama_afiliasi", "rank_overall"]],
        on="nama_afiliasi",
        how="left"
    )

    # Hitung score minimal Top 10 dan gap ke Top 10
    top10_threshold = None
    if len(af_ranked) >= 10:
        top10_threshold = af_ranked.loc[9, "sinta_score_overall"]
        df_af["gap_to_top10"] = np.where(
            df_af["rank_overall"] <= 10,
            0,
            top10_threshold - df_af["sinta_score_overall"]
        )
    else:
        df_af["gap_to_top10"] = np.nan

    return df_af, df_detail, cat_pivot, top10_threshold


@st.cache_data
def load_metrics_detail(file):
    """Load Sinta Metrics Detail.xlsx (sheet metrics_details)."""
    xls = pd.read_excel(file, sheet_name=None)
    if "metrics_details" in xls:
        df = xls["metrics_details"].copy()
    else:
        # fallback: ambil sheet pertama jika nama sheet beda
        first_sheet_name = list(xls.keys())[0]
        df = xls[first_sheet_name].copy()
    return df


# ------------------------------
# Fungsi Helper Visualisasi
# ------------------------------

def plot_overall_ranking(df_af, selected_affiliation, top_n=15):
    df = df_af.sort_values("sinta_score_overall", ascending=False).head(top_n).copy()

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("sinta_score_overall:Q", title="SINTA Score Overall"),
            y=alt.Y("nama_afiliasi:N", sort="-x", title="Afiliasi"),
            opacity=alt.condition(
                alt.datum.nama_afiliasi == selected_affiliation,
                alt.value(1.0),
                alt.value(0.4),
            ),
            tooltip=[
                alt.Tooltip("nama_afiliasi:N", title="Afiliasi"),
                alt.Tooltip("rank_overall:Q", title="Peringkat"),
                alt.Tooltip("sinta_score_overall:Q", title="Skor Overall", format=","),
                alt.Tooltip("gap_to_top10:Q", title="Gap ke Top 10", format=","),
            ],
        )
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)


def plot_category_breakdown(cat_pivot, selected_affiliation):
    if selected_affiliation not in cat_pivot.index:
        st.warning("Data kategori tidak ditemukan untuk afiliasi ini.")
        return

    row = cat_pivot.loc[selected_affiliation]
    df_cat = (
        row.reset_index()
        .rename(columns={"index": "category", 0: "total"})
        if isinstance(row, pd.Series)
        else row.reset_index().rename(columns={"index": "category"})
    )

    df_cat.columns = ["category", "total"]
    df_cat = df_cat[df_cat["category"].notna()].copy()
    df_cat["share_percent"] = df_cat["total"] / df_cat["total"].sum() * 100
    df_cat["category"] = pd.Categorical(
        df_cat["category"],
        categories=CATEGORY_ORDER,
        ordered=True,
    )
    df_cat = df_cat.sort_values("category")

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            df_cat.style.format(
                {"total": "{:,.3f}", "share_percent": "{:,.2f}%"}
            ),
            use_container_width=True,
            height=300,
        )

    with col2:
        chart = (
            alt.Chart(df_cat)
            .mark_bar()
            .encode(
                x=alt.X("share_percent:Q", title="Persentase Kontribusi (%)"),
                y=alt.Y("category:N", sort="-x", title="Kategori"),
                tooltip=[
                    alt.Tooltip("category:N", title="Kategori"),
                    alt.Tooltip("total:Q", title="Total Skor", format=","),
                    alt.Tooltip("share_percent:Q", title="Kontribusi (%)", format=".2f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


def get_high_leverage_metrics(df_detail, selected_affiliation, top_k=30):
    df = df_detail[df_detail["nama_afiliasi"] == selected_affiliation].copy()
    if df.empty:
        return df

    df["category"] = df["kategori_score"].map(CATEGORY_LABELS)
    # Potensi kenaikan: weight * (1 - value)
    df["potential_gain"] = df["weight"] * (1 - df["value"])
    df = df.sort_values("potential_gain", ascending=False)

    return df.head(top_k)


def plot_high_leverage_bar(df_high):
    if df_high.empty:
        st.info("Tidak ada data high-leverage metrics untuk afiliasi ini.")
        return

    chart_data = df_high.copy()
    chart_data["label"] = chart_data["kode"] + " – " + chart_data["category"]

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("potential_gain:Q", title="Potensi Kenaikan (weight × (1 - value))"),
            y=alt.Y("label:N", sort="-x", title="Kode – Kategori"),
            tooltip=[
                alt.Tooltip("kode:N", title="Kode"),
                alt.Tooltip("nama:N", title="Nama Indikator"),
                alt.Tooltip("category:N", title="Kategori"),
                alt.Tooltip("weight:Q", title="Bobot"),
                alt.Tooltip("value:Q", title="Value (0–1)", format=".3f"),
                alt.Tooltip("total:Q", title="Total Saat Ini", format=","),
                alt.Tooltip("potential_gain:Q", title="Potensi Kenaikan", format=".3f"),
            ],
        )
        .properties(height=500)
    )
    st.altair_chart(chart, use_container_width=True)


def compare_universities_df(df_md, aff1, aff2, metric_col):
    """
    Bandingkan dua afiliasi per code, side by side.
    Kategori diambil dari kolom `area` pada Sinta Metrics Detail (Publikasi, Penelitian, dll).
    """
    df1 = df_md[df_md["affiliation_name"] == aff1].copy()
    df2 = df_md[df_md["affiliation_name"] == aff2].copy()

    # Data A (selected)
    base = df1[["code", "name", "area", metric_col]].rename(
        columns={metric_col: "score_selected", "area": "category"}
    )

    # Data B (compare)
    comp = df2[["code", "area", metric_col]].rename(
        columns={metric_col: "score_compare", "area": "category_b"}
    )

    # Merge per code
    df = pd.merge(base, comp, on="code", how="outer")

    # Isi nama indikator dari B kalau di A kosong
    if "name" in df.columns:
        name_map = df2.set_index("code")["name"]
        df["name"] = df["name"].fillna(df["code"].map(name_map))

    # Ambil kategori dari A, kalau kosong pakai kategori dari B
    df["category"] = df["category"].fillna(df["category_b"])
    df = df.drop(columns=["category_b"])

    # Isi NaN skor dengan 0 supaya bisa dihitung
    df["score_selected"] = df["score_selected"].fillna(0)
    df["score_compare"] = df["score_compare"].fillna(0)

    # Hitung selisih
    df["diff_abs"] = df["score_selected"] - df["score_compare"]
    df["diff_pct"] = np.where(
        df["score_compare"] != 0,
        df["diff_abs"] / df["score_compare"] * 100,
        np.nan,
    )

    return df


def color_diff(row):
    """
    Styling untuk tabel compare:
    - Hijau kalau score_selected > score_compare
    - Merah kalau score_selected < score_compare
    - Kosong kalau sama
    """
    diff = row["diff_abs"]
    if diff > 0:
        color = "background-color: #c6efce"  # hijau
    elif diff < 0:
        color = "background-color: #ffc7ce"  # merah
    else:
        color = ""
    return [
        color if col in ["score_selected", "score_compare", "diff_abs", "diff_pct"] else ""
        for col in row.index
    ]


# ------------------------------
# App Utama
# ------------------------------

def main():
    st.set_page_config(
        page_title="SINTA Analytics – BINUS University",
        layout="wide",
    )

    st.title("📊 SINTA Analytics App")
    st.caption(
        "Memvisualisasikan dan menganalisis data SINTA untuk melihat potensi "
        "kenaikan peringkat BINUS University (dan afiliasi lain)."
    )

    # Sidebar: upload file
    st.sidebar.header("📁 Data Input")

    cluster_file = st.sidebar.file_uploader(
        "Upload **Sinta Metric Cluster.xlsx**",
        type=["xlsx"],
        key="cluster",
        help="File yang berisi sheet `afiliasi` dan `detail_kode`.",
    )

    metrics_file = st.sidebar.file_uploader(
        "Upload **Sinta Metrics Detail v2.xlsx / Sinta Metrics Detail.xlsx**",
        type=["xlsx"],
        key="metrics",
        help="File yang berisi sheet `metrics_details` (nilai raw per kode).",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Jika tidak upload, app akan mencoba membaca file dengan nama yang sama "
        "di folder yang sama dengan `app.py`."
    )

    # Load data dengan fallback ke file lokal
    df_af = df_detail = cat_pivot = df_md = top10_threshold = None

    if cluster_file is not None:
        df_af, df_detail, cat_pivot, top10_threshold = load_cluster_data(cluster_file)
    else:
        try:
            df_af, df_detail, cat_pivot, top10_threshold = load_cluster_data(
                "Sinta Metric Cluster.xlsx"
            )
            st.sidebar.success("Menggunakan file lokal: Sinta Metric Cluster.xlsx")
        except Exception:
            st.error(
                "Tidak dapat membuka **Sinta Metric Cluster.xlsx**. "
                "Silakan upload file tersebut di sidebar."
            )
            st.stop()

    if metrics_file is not None:
        try:
            df_md = load_metrics_detail(metrics_file)
        except Exception:
            df_md = None
            st.sidebar.warning("Gagal membaca Sinta Metrics Detail (v2) yang diupload.")
    else:
        # Coba v2 dulu, lalu fallback ke versi lama
        try:
            df_md = load_metrics_detail("Sinta Metrics Detail v2.xlsx")
            st.sidebar.success("Menggunakan file lokal: Sinta Metrics Detail v2.xlsx")
        except Exception:
            try:
                df_md = load_metrics_detail("Sinta Metrics Detail.xlsx")
                st.sidebar.success("Menggunakan file lokal: Sinta Metrics Detail.xlsx")
            except Exception:
                df_md = None
                st.sidebar.info(
                    "File **Sinta Metrics Detail** tidak ditemukan. "
                    "Tab analisis metrics detail & compare akan terbatas."
                )

    # Mapping code -> category (untuk tab selain compare)
    code_cat_map = (
        df_detail[["kode", "kategori_score"]]
        .drop_duplicates()
        .rename(columns={"kode": "code"})
    )
    code_cat_map["category"] = code_cat_map["kategori_score"].map(CATEGORY_LABELS)

    # Pilih afiliasi (default: Universitas Bina Nusantara kalau ada)
    affiliations = sorted(df_af["nama_afiliasi"].unique().tolist())
    default_aff = (
        affiliations.index("Universitas Bina Nusantara")
        if "Universitas Bina Nusantara" in affiliations
        else 0
    )

    selected_affiliation = st.sidebar.selectbox(
        "Pilih afiliasi untuk dianalisis (sebagai basis BINUS / A):",
        options=affiliations,
        index=default_aff,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Tip: Pilih **Universitas Bina Nusantara** untuk fokus analisis ke BINUS University."
    )

    # Ambil row afiliasi terpilih
    aff_row = df_af[df_af["nama_afiliasi"] == selected_affiliation].iloc[0]

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview & Ranking",
            "🏫 Profil Afiliasi",
            "🎯 High-Leverage Metrics + Simulasi",
            "📈 Metrics Detail & Raw Values",
            "⚖️ Compare Universities",
        ]
    )

    # --------------------------
    # Tab 1 – Overview & Ranking
    # --------------------------
    with tab1:
        st.subheader("📊 Posisi Umum & Ranking Nasional")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Peringkat nasional (SINTA Score Overall)",
                f"{int(aff_row['rank_overall'])}",
            )
        with col2:
            st.metric(
                "Skor SINTA Overall",
                f"{int(aff_row['sinta_score_overall']):,}".replace(",", "."),
            )
        with col3:
            st.metric(
                "Skor SINTA 3 Tahun",
                f"{int(aff_row['sinta_score_3yr']):,}".replace(",", "."),
            )
        with col4:
            if not pd.isna(aff_row["gap_to_top10"]):
                gap = int(aff_row["gap_to_top10"])
                st.metric(
                    "Gap ke batas Top 10",
                    f"{gap:,}".replace(",", "."),
                    help="Selisih skor SINTA overall dibandingkan afiliasi peringkat 10 nasional.",
                )
            else:
                st.metric("Gap ke batas Top 10", "–")

        st.markdown("#### Ranking & gap terhadap Top N")

        max_top_n = min(40, len(df_af))
        top_n = st.slider(
            "Tampilkan berapa besar afiliasi teratas?",
            min_value=5,
            max_value=max_top_n,
            value=min(15, max_top_n),
        )

        plot_overall_ranking(df_af, selected_affiliation, top_n=top_n)

        st.markdown("#### Tabel Top 20 SINTA Score Overall")

        top20 = (
            df_af.sort_values("sinta_score_overall", ascending=False)
            .head(20)
            .copy()
        )
        cols_to_show = [
            "rank_overall",
            "nama_afiliasi",
            "sinta_score_overall",
            "sinta_score_3yr",
            "gap_to_top10",
        ]
        top20 = top20[cols_to_show]
        st.dataframe(
            top20.style.format(
                {
                    "sinta_score_overall": "{:,.0f}",
                    "sinta_score_3yr": "{:,.0f}",
                    "gap_to_top10": "{:,.0f}",
                }
            ),
            height=450,
            use_container_width=True,
        )

    # --------------------------
    # Tab 2 – Profil Afiliasi
    # --------------------------
    with tab2:
        st.subheader(f"🏫 Profil Afiliasi: {selected_affiliation}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Peringkat nasional",
                f"{int(aff_row['rank_overall'])}",
            )
        with col2:
            st.metric(
                "Skor SINTA Overall",
                f"{int(aff_row['sinta_score_overall']):,}".replace(",", "."),
            )
        with col3:
            st.metric(
                "Skor SINTA 3 Tahun",
                f"{int(aff_row['sinta_score_3yr']):,}".replace(",", "."),
            )

        st.markdown("#### Kontribusi per Kategori Skor SINTA")
        plot_category_breakdown(cat_pivot, selected_affiliation)

        st.markdown("#### Perbandingan kategori dengan afiliasi lain")

        cat_to_compare = st.selectbox(
            "Pilih kategori untuk dibandingkan:",
            options=CATEGORY_ORDER,
        )

        if cat_to_compare in cat_pivot.columns:
            df_compare = (
                cat_pivot[[cat_to_compare]]
                .reset_index()
                .rename(columns={cat_to_compare: "total"})
            )
            df_compare["rank_in_category"] = (
                df_compare["total"].rank(ascending=False, method="min").astype(int)
            )

            df_compare = df_compare.sort_values("total", ascending=False)
            top_k_cat = min(20, len(df_compare))
            df_view = df_compare.head(top_k_cat)

            chart = (
                alt.Chart(df_view)
                .mark_bar()
                .encode(
                    x=alt.X("total:Q", title=f"Total Skor – {cat_to_compare}"),
                    y=alt.Y("nama_afiliasi:N", sort="-x", title="Afiliasi"),
                    opacity=alt.condition(
                        alt.datum.nama_afiliasi == selected_affiliation,
                        alt.value(1.0),
                        alt.value(0.4),
                    ),
                    tooltip=[
                        alt.Tooltip("nama_afiliasi:N", title="Afiliasi"),
                        alt.Tooltip("total:Q", title="Total Skor", format=","),
                        alt.Tooltip(
                            "rank_in_category:Q", title="Peringkat di kategori"
                        ),
                    ],
                )
                .properties(height=500)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Kategori tersebut tidak ditemukan di data kategori.")

    # --------------------------
    # Tab 3 – High-Leverage Metrics + Simulasi
    # --------------------------
    with tab3:
        st.subheader(f"🎯 High-Leverage Metrics – {selected_affiliation}")

        df_high = get_high_leverage_metrics(df_detail, selected_affiliation, top_k=30)

        st.markdown(
            "Indikator di bawah ini adalah kombinasi **bobot tinggi** dan **value masih rendah**. "
            "Kalau nilai indikator ini bisa dinaikkan, potensi kenaikan skor kategori "
            "dan skor SINTA cukup besar."
        )

        plot_high_leverage_bar(df_high)

        st.markdown("#### Detail High-Leverage Metrics (Top 30)")
        if not df_high.empty:
            show_cols = [
                "kode",
                "nama",
                "category",
                "weight",
                "value",
                "total",
                "potential_gain",
            ]
            st.dataframe(
                df_high[show_cols].style.format(
                    {
                        "weight": "{:,.2f}",
                        "value": "{:,.3f}",
                        "total": "{:,.3f}",
                        "potential_gain": "{:,.3f}",
                    }
                ),
                use_container_width=True,
                height=500,
            )
        else:
            st.info("Tidak ada data high-leverage untuk afiliasi ini.")

        st.markdown("---")
        st.subheader("🧪 Simulasi Sederhana: Naikkan Nilai Indikator")

        st.caption(
            "Simulasi ini **tidak** menghitung ulang skor SINTA resmi, "
            "tapi memberi gambaran bagaimana perubahan value indikator "
            "mempengaruhi total skor per kategori."
        )

        if selected_affiliation not in cat_pivot.index:
            st.info("Tidak bisa melakukan simulasi karena data kategori tidak lengkap.")
        else:
            # Data detail untuk afiliasi terpilih
            df_aff_detail = df_detail[df_detail["nama_afiliasi"] == selected_affiliation].copy()
            df_aff_detail["category"] = df_aff_detail["kategori_score"].map(CATEGORY_LABELS)

            sim_categories = st.multiselect(
                "Pilih kategori yang ingin ditingkatkan:",
                options=CATEGORY_ORDER,
                default=["Publication", "HKI"],
            )

            delta_value = st.slider(
                "Naikkan value semua indikator di kategori tersebut sebesar:",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Misalnya 0.1 berarti value naik 0.1 (maksimum 1.0).",
            )

            if st.button("Hitung Simulasi", type="primary"):
                if not sim_categories:
                    st.warning("Pilih minimal satu kategori untuk simulasi.")
                else:
                    df_sim = df_aff_detail.copy()
                    mask = df_sim["category"].isin(sim_categories)

                    df_sim["value_sim"] = df_sim["value"]
                    df_sim.loc[mask, "value_sim"] = (
                        df_sim.loc[mask, "value_sim"] + delta_value
                    ).clip(upper=1.0)

                    df_sim["total_sim"] = df_sim["weight"] * df_sim["value_sim"]

                    # Agregasi kategori: sebelum dan sesudah
                    original_cat = (
                        df_aff_detail.groupby("category")["total"].sum().rename("original_total")
                    )
                    sim_cat = (
                        df_sim.groupby("category")["total_sim"].sum().rename("simulated_total")
                    )

                    df_cat_sim = (
                        pd.concat([original_cat, sim_cat], axis=1)
                        .fillna(0)
                        .reset_index()
                    )
                    df_cat_sim["change"] = df_cat_sim["simulated_total"] - df_cat_sim["original_total"]
                    df_cat_sim["change_percent"] = np.where(
                        df_cat_sim["original_total"] > 0,
                        df_cat_sim["change"] / df_cat_sim["original_total"] * 100,
                        np.nan,
                    )

                    df_cat_sim["category"] = pd.Categorical(
                        df_cat_sim["category"],
                        categories=CATEGORY_ORDER,
                        ordered=True,
                    )
                    df_cat_sim = df_cat_sim.sort_values("category")

                    st.markdown("##### Hasil Simulasi per Kategori")
                    st.dataframe(
                        df_cat_sim.style.format(
                            {
                                "original_total": "{:,.3f}",
                                "simulated_total": "{:,.3f}",
                                "change": "{:,.3f}",
                                "change_percent": "{:,.2f}%",
                            }
                        ),
                        use_container_width=True,
                        height=350,
                    )

                    st.caption(
                        "Interpretasi sederhana: semakin besar `change` dan `change_percent` "
                        "di kategori tertentu, semakin besar potensi kontribusi dari peningkatan "
                        "indikator-indikator di kategori tersebut."
                    )

    # --------------------------
    # Tab 4 – Metrics Detail & Raw Values
    # --------------------------
    with tab4:
        st.subheader("📈 Analisis dari Sinta Metrics Detail")

        if df_md is None:
            st.info(
                "Data **Sinta Metrics Detail** belum tersedia. "
                "Upload file Sinta Metrics Detail v2.xlsx / Sinta Metrics Detail.xlsx di sidebar."
            )
        else:
            aff_upper = selected_affiliation.upper()
            df_md_aff = df_md[df_md["affiliation_name"] == aff_upper].copy()

            if df_md_aff.empty:
                st.warning(
                    "Tidak ditemukan baris di Sinta Metrics Detail untuk afiliasi ini. "
                    "Pastikan penulisan nama afiliasi di file sama dengan yang di SINTA Metric Cluster."
                )
            else:
                # Join kategori (versi cluster) hanya untuk analisis tambahan
                df_md_aff = df_md_aff.merge(
                    code_cat_map[["code", "category"]],
                    on="code",
                    how="left",
                )

                st.caption(
                    f"Analisis berikut menggunakan **Sinta Metrics Detail** untuk afiliasi: `{aff_upper}`."
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    total_overall = df_md_aff["sinta_v3_overall_total"].sum()
                    st.metric("Total skor overall (sum semua kode)", f"{total_overall:,.0f}")
                with col2:
                    total_3yr = df_md_aff["sinta_v3_3yr_total"].sum()
                    st.metric("Total skor 3-year (sum semua kode)", f"{total_3yr:,.0f}")
                with col3:
                    recent_ratio = (
                        total_3yr / total_overall * 100 if total_overall > 0 else np.nan
                    )
                    st.metric(
                        "Proporsi skor 3 tahun terakhir",
                        f"{recent_ratio:,.1f}%" if not np.isnan(recent_ratio) else "–",
                    )

                st.markdown("#### Top 15 Indikator Penyumbang Skor Terbesar (Overall Total)")

                df_md_aff["overall_share"] = df_md_aff["sinta_v3_overall_total"] / max(
                    df_md_aff["sinta_v3_overall_total"].sum(), 1
                )

                top_overall = (
                    df_md_aff.sort_values("sinta_v3_overall_total", ascending=False)
                    .head(15)
                    .copy()
                )

                st.dataframe(
                    top_overall[
                        [
                            "code",
                            "name",
                            "area",
                            "category",
                            "weight",
                            "sinta_v3_overall_value",
                            "sinta_v3_overall_total",
                            "overall_share",
                        ]
                    ].style.format(
                        {
                            "weight": "{:,.2f}",
                            "sinta_v3_overall_value": "{:,.0f}",
                            "sinta_v3_overall_total": "{:,.0f}",
                            "overall_share": "{:.2%}",
                        }
                    ),
                    use_container_width=True,
                    height=350,
                )

                st.markdown("#### Indikator yang Paling 'Baru' (Fokus 3 Tahun Terakhir)")

                df_md_aff["recent_ratio"] = np.where(
                    df_md_aff["sinta_v3_overall_total"] > 0,
                    df_md_aff["sinta_v3_3yr_total"] / df_md_aff["sinta_v3_overall_total"],
                    np.nan,
                )

                recent_top = (
                    df_md_aff[df_md_aff["sinta_v3_3yr_total"] > 0]
                    .sort_values("recent_ratio", ascending=False)
                    .head(15)
                    .copy()
                )

                st.dataframe(
                    recent_top[
                        [
                            "code",
                            "name",
                            "area",
                            "category",
                            "weight",
                            "sinta_v3_overall_total",
                            "sinta_v3_3yr_total",
                            "recent_ratio",
                        ]
                    ].style.format(
                        {
                            "weight": "{:,.2f}",
                            "sinta_v3_overall_total": "{:,.0f}",
                            "sinta_v3_3yr_total": "{:,.0f}",
                            "recent_ratio": "{:.2%}",
                        }
                    ),
                    use_container_width=True,
                    height=350,
                )

                st.markdown("---")
                st.markdown("#### Tabel Nilai Raw per Kode (Filterable)")

                filter_text = st.text_input(
                    "Filter berdasarkan kode atau nama indikator (opsional):",
                    value="",
                ).strip()

                df_view = df_md_aff.copy()
                if filter_text:
                    mask = (
                        df_view["code"].astype(str).str.contains(filter_text, case=False)
                        | df_view["name"].astype(str).str.contains(filter_text, case=False)
                    )
                    df_view = df_view[mask]

                cols_show = [
                    "code",
                    "name",
                    "area",
                    "category",
                    "weight",
                    "sinta_v3_overall_value",
                    "sinta_v3_overall_total",
                    "sinta_v3_3yr_value",
                    "sinta_v3_3yr_total",
                ]
                cols_show = [c for c in cols_show if c in df_view.columns]

                st.dataframe(
                    df_view[cols_show].style.format(
                        {
                            "weight": "{:,.2f}",
                            "sinta_v3_overall_value": "{:,.0f}",
                            "sinta_v3_overall_total": "{:,.0f}",
                            "sinta_v3_3yr_value": "{:,.0f}",
                            "sinta_v3_3yr_total": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                    height=450,
                )

    # --------------------------
    # Tab 5 – Compare Universities
    # --------------------------
    with tab5:
        st.subheader("⚖️ Bandingkan Dua Universitas per Metric")

        if df_md is None:
            st.info(
                "Data **Sinta Metrics Detail** belum tersedia, jadi fitur compare belum bisa digunakan. "
                "Upload Sinta Metrics Detail v2.xlsx / Sinta Metrics Detail.xlsx di sidebar dulu."
            )
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                aff_a = st.selectbox(
                    "Universitas A (acuan / hijau):",
                    options=affiliations,
                    index=default_aff,
                )
            with col_b:
                # default B: kampus tepat di atas BINUS kalau ada, kalau tidak index 0
                default_b_idx = 0
                if "Universitas Bina Nusantara" in affiliations:
                    idx_binus = affiliations.index("Universitas Bina Nusantara")
                    default_b_idx = max(0, idx_binus - 1)
                aff_b = st.selectbox(
                    "Universitas B (pembanding / baseline):",
                    options=affiliations,
                    index=default_b_idx,
                )

            metric_type = st.selectbox(
                "Pilih jenis skor untuk dibandingkan:",
                options=[
                    "Overall Total (sinta_v3_overall_total)",
                    "3-Year Total (sinta_v3_3yr_total)",
                    "Overall Value (sinta_v3_overall_value)",
                ],
            )

            metric_col_map = {
                "Overall Total (sinta_v3_overall_total)": "sinta_v3_overall_total",
                "3-Year Total (sinta_v3_3yr_total)": "sinta_v3_3yr_total",
                "Overall Value (sinta_v3_overall_value)": "sinta_v3_overall_value",
            }
            metric_col = metric_col_map[metric_type]

            # Kategori filter diambil dari kolom `area` pada Sinta Metrics Detail v2
            area_options = sorted(df_md["area"].dropna().unique().tolist())
            cat_filter = st.multiselect(
                "Filter kategori / area (opsional):",
                options=area_options,
                default=[],
                help="Kategori diambil dari kolom `area` (Publikasi, Penelitian, Pengabdian Kepada Masyarakat, dll).",
            )

            show_n = st.slider(
                "Tampilkan berapa metric dengan selisih absolut terbesar?",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
            )

            if st.button("Bandingkan", type="primary"):
                aff_a_upper = aff_a.upper()
                aff_b_upper = aff_b.upper()

                df_comp = compare_universities_df(
                    df_md, aff_a_upper, aff_b_upper, metric_col
                )

                # Filter kategori jika dipilih
                if cat_filter:
                    df_comp = df_comp[df_comp["category"].isin(cat_filter)]

                # Sort berdasarkan selisih absolut terbesar
                df_comp = df_comp.sort_values(
                    "diff_abs", key=lambda s: s.abs(), ascending=False
                )

                # Ambil top N
                df_comp = df_comp.head(show_n)

                # Ringkasan
                better = (df_comp["diff_abs"] > 0).sum()
                worse = (df_comp["diff_abs"] < 0).sum()
                equal = (df_comp["diff_abs"] == 0).sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Metric di mana {aff_a} lebih tinggi", better)
                with col2:
                    st.metric(f"Metric di mana {aff_a} lebih rendah", worse)
                with col3:
                    st.metric("Metric sama / imbang", equal)

                st.markdown(
                    f"#### Perbandingan per Metric ({metric_type})\n"
                    f"**Hijau**: skor {aff_a} lebih tinggi, **Merah**: skor {aff_a} lebih rendah."
                )

                # Siapkan dataframe yang akan ditampilkan
                df_show = df_comp[
                    [
                        "code",
                        "name",
                        "category",   # ini = area, misalnya Publikasi / Penelitian / dll
                        "score_selected",
                        "score_compare",
                        "diff_abs",
                        "diff_pct",
                    ]
                ].copy()

                st.caption(f"score_selected = {aff_a}  |  score_compare = {aff_b}")

                styler = (
                    df_show.style
                    .format(
                        {
                            "score_selected": "{:,.2f}",
                            "score_compare": "{:,.2f}",
                            "diff_abs": "{:,.2f}",
                            "diff_pct": "{:,.2f}%",
                        }
                    )
                    .apply(color_diff, axis=1)
                )

                st.dataframe(styler, use_container_width=True, height=600)


if __name__ == "__main__":
    main()
