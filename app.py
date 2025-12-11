import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from itertools import combinations
from collections import Counter

# 1. Cấu hình trang
st.set_page_config(layout="wide", page_title="Market Intelligence Dashboard", page_icon="📍")

st.title("📍 Báo cáo Phân tích Đối thủ Đào tạo")

# CSS UI
st.markdown("""
<style>
    .info-card {
        background-color: #ffffff;
        padding: 20px; border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-top: 5px solid #ff4b4b;
    }
    .card-title { color: #ff4b4b; font-size: 20px; font-weight: bold; margin-bottom: 10px; }
    .card-row { margin-bottom: 8px; font-size: 15px; }
    .card-label { font-weight: 600; color: #555; }
    .tooltip {
        position: relative; display: inline-block;
        border-bottom: 1px dotted black; color: #856404;
        background-color: #fff3cd; padding: 5px 10px;
        border-radius: 5px; border: 1px solid #ffeeba;
        font-weight: bold; cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden; width: 350px; background-color: #333;
        color: #fff; text-align: left; border-radius: 6px;
        padding: 10px; position: absolute; z-index: 1;
        top: 125%; left: 50%; margin-left: -175px;
        opacity: 0; transition: opacity 0.3s;
        font-size: 12px; font-weight: normal; white-space: pre-wrap;
    }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# 2. Load Data
@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv("data.csv")
        total_records = len(df_raw)
        
        # Xử lý toạ độ
        if 'lat' in df_raw.columns and 'lon' in df_raw.columns:
            df_raw['lat'] = pd.to_numeric(df_raw['lat'], errors='coerce')
            df_raw['lon'] = pd.to_numeric(df_raw['lon'], errors='coerce')
            df_clean = df_raw.dropna(subset=['lat', 'lon']).copy()
            df_missing = df_raw[df_raw['lat'].isna() | df_raw['lon'].isna()].copy()
            df_clean['lat'] = df_clean['lat'].round(6)
            df_clean['lon'] = df_clean['lon'].round(6)
        else:
            return None, None, 0

        # Xử lý Quận (District)
        if 'District' in df_clean.columns:
            df_clean['District'] = df_clean['District'].astype(str).str.strip().replace('nan', 'Chưa xác định')
        else:
            df_clean['District'] = 'Chưa xác định'

        # Xử lý Tags (Type1, Type2)
        def split_tags(text):
            if pd.isna(text) or str(text).strip() == '': return []
            return [t.strip() for t in str(text).split(',')]

        for col in ['type1', 'type2']:
            list_col = f"{col.capitalize()}_List"
            if col in df_clean.columns:
                df_clean[list_col] = df_clean[col].apply(split_tags)
            else:
                df_clean[list_col] = []
        
        df_clean['Service_Count'] = df_clean['Type1_List'].apply(len)
        return df_clean, df_missing, total_records

    except FileNotFoundError:
        return None, None, 0

df, df_missing, total_records = load_data()
if df is None: st.error("⚠️ Không tìm thấy file data.csv"); st.stop()

# ==================================================
# SIDEBAR: BỘ LỌC
# ==================================================
with st.sidebar:
    st.header("🎛️ Bộ lọc")
    
    st.subheader("📍 Khu vực")
    all_districts = sorted(df['District'].unique())
    filter_mode = st.radio("Chế độ:", ["Xem tất cả (Trừ...)", "Chỉ xem (Chọn...)"], horizontal=True, label_visibility="collapsed")
    
    if filter_mode == "Xem tất cả (Trừ...)":
        excluded_districts = st.multiselect("⛔ Ẩn khu vực:", options=all_districts)
        final_selected_districts = [d for d in all_districts if d not in excluded_districts]
    else:
        included_districts = st.multiselect("✅ Chọn khu vực:", options=all_districts, default=all_districts[:3] if len(all_districts)>3 else all_districts)
        final_selected_districts = included_districts

    st.markdown("---")

    all_type1 = sorted(list(set([tag for sublist in df['Type1_List'] for tag in sublist])))
    with st.expander("🎭 Lọc Môn Đào Tạo", expanded=False):
        if 'selected_type1' not in st.session_state: st.session_state['selected_type1'] = all_type1
        if st.button("Chọn hết môn"): st.session_state['selected_type1'] = all_type1
        if st.button("Bỏ chọn môn"): st.session_state['selected_type1'] = []
        selected_type1 = st.multiselect("Chọn môn:", options=all_type1, key='selected_type1')

    all_type2 = sorted(list(set([tag for sublist in df['Type2_List'] for tag in sublist])))
    with st.expander("👥 Lọc Đối Tượng", expanded=False):
        selected_type2 = st.multiselect("Chọn đối tượng:", options=all_type2, default=all_type2)

    def filter_row(row):
        ok_dist = row['District'] in final_selected_districts
        ok_type1 = True if not selected_type1 else (not set(row['Type1_List']).isdisjoint(selected_type1) if row['Type1_List'] else False)
        ok_type2 = True if not selected_type2 else (not set(row['Type2_List']).isdisjoint(selected_type2) if row['Type2_List'] else False)
        return ok_dist and ok_type1 and ok_type2

    df_filtered = df[df.apply(filter_row, axis=1)].reset_index(drop=True)

# ==================================================
# MAIN TABS
# ==================================================
tab1, tab2 = st.tabs(["🗺️ BẢN ĐỒ & TRA CỨU", "📈 PHÂN TÍCH CHUYÊN SÂU"])

# --- TAB 1: BẢN ĐỒ ---
with tab1:
    c1, c2 = st.columns([3, 1])
    with c1: 
        missing_cnt = len(df_missing)
        if missing_cnt > 0:
            missing_names = "\n- ".join(df_missing['Name'].astype(str).tolist())
            st.markdown(f"""<div class="tooltip">⚠️ {missing_cnt} cơ sở thiếu toạ độ<span class="tooltiptext">{missing_names}</span></div>""", unsafe_allow_html=True)
        else:
            st.success(f"✅ Dữ liệu đầy đủ (Hiển thị: {len(df_filtered)}/{len(df)})")
    with c2: st.metric("Tổng hiển thị", f"{len(df_filtered)}")
    
    if not df_filtered.empty:
        center_lat = df_filtered['lat'].mean()
        center_lon = df_filtered['lon'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        marker_cluster = MarkerCluster().add_to(m)

        for idx, row in df_filtered.iterrows():
            color = 'red' if row['Service_Count'] > 1 else 'blue'
            popup_simple = f"<b>{row['Name']}</b><br>{row['District']}"
            folium.Marker(
                [row['lat'], row['lon']],
                popup=popup_simple, tooltip=row['Name'],
                icon=folium.Icon(color=color, icon='star' if row['Service_Count']>1 else 'info-sign')
            ).add_to(marker_cluster)

        map_output = st_folium(m, height=500, use_container_width=True)
    else:
        st.warning("Không có dữ liệu hiển thị bản đồ."); map_output = None

    st.markdown("---")

    col_list, col_detail = st.columns([1, 1])
    selected_row = None
    if map_output and map_output['last_object_clicked']:
        clicked_lat = map_output['last_object_clicked']['lat']
        clicked_lon = map_output['last_object_clicked']['lng']
        found = df_filtered[(df_filtered['lat'].round(5) == round(clicked_lat, 5)) & (df_filtered['lon'].round(5) == round(clicked_lon, 5))]
        if not found.empty: selected_row = found.iloc[0]

    with col_list:
        st.subheader("📋 Danh sách Cơ sở")
        selection = st.dataframe(
            df_filtered[['Name', 'District', 'type1']],
            column_config={"Name": "Tên Trung tâm", "District": "Khu vực", "type1": "Môn học"},
            hide_index=True, use_container_width=True, height=400, on_select="rerun", selection_mode="single-row"
        )
        if selection and len(selection.selection.rows) > 0:
            selected_idx = selection.selection.rows[0]
            selected_row = df_filtered.iloc[selected_idx]

    with col_detail:
        st.subheader("🏢 Thông tin chi tiết")
        if selected_row is not None:
            with st.container():
                web_btn = f"""<a href="{selected_row['Website']}" target="_blank" style="text-decoration:none; background:#ff4b4b; color:white; padding:5px 10px; border-radius:5px; margin-right:5px;">🌐 Website</a>""" if isinstance(selected_row.get('Website'), str) and 'http' in str(selected_row['Website']) else ""
                fb_btn = f"""<a href="{selected_row['link']}" target="_blank" style="text-decoration:none; background:#1877F2; color:white; padding:5px 10px; border-radius:5px; margin-right:5px;">📘 Facebook</a>""" if isinstance(selected_row.get('link'), str) and 'http' in str(selected_row['link']) else ""
                map_btn = f"""<a href="{selected_row['Map_Link']}" target="_blank" style="text-decoration:none; background:#34A853; color:white; padding:5px 10px; border-radius:5px;">🗺️ Chỉ đường</a>""" if isinstance(selected_row.get('Map_Link'), str) and 'http' in str(selected_row['Map_Link']) else ""
                note_html = f"""<div style="background:#fff3cd; color:#856404; padding:8px; border-radius:5px; margin-top:10px; font-size:14px;">📝 <b>Ghi chú:</b> {selected_row['note']}</div>""" if isinstance(selected_row.get('note'), str) and len(str(selected_row['note'])) > 1 else ""

                st.markdown(f"""
                <div class="info-card">
                    <div class="card-title">{selected_row['Name']}</div>
                    <div class="card-row"><span class="card-label">📍 Khu vực:</span> {selected_row['District']}</div>
                    <div class="card-row"><span class="card-label">🏠 Địa chỉ:</span> {selected_row.get('Address', 'Chưa cập nhật')}</div>
                    <hr style="margin:10px 0; border-top:1px dashed #ccc;">
                    <div class="card-row"><span class="card-label">🎭 Môn đào tạo:</span><br>{selected_row['type1']}</div>
                    <div class="card-row"><span class="card-label">👥 Đối tượng:</span><br>{selected_row['type2']}</div>
                    {note_html}
                    <div style="margin-top:15px;">{web_btn} {fb_btn} {map_btn}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Chọn một cơ sở để xem chi tiết.")

# --- TAB 2: PHÂN TÍCH CHUYÊN SÂU ---
with tab2:
    if not df_filtered.empty:
        # ==========================
        # SECTION 1: OVERVIEW
        # ==========================
        st.subheader("1. 📊 Tổng quan & Cơ cấu")
        
        c_seg1, c_seg2 = st.columns(2)
        
        with c_seg1:
            st.markdown("**👥 Cơ cấu Đối tượng Học viên (Segment)**")
            
            def classify_segment(tags_list):
                tags_str = str(tags_list).lower()
                has_kid = 'nhí' in tags_str or 'trẻ' in tags_str or 'bé' in tags_str
                has_adult = 'lớn' in tags_str or 'người lớn' in tags_str
                
                if has_kid and has_adult: return "Đào tạo Cả hai (Mix)"
                elif has_kid: return "Chuyên Đào tạo Nhí"
                elif has_adult: return "Chuyên Đào tạo Người lớn"
                else: return "Chưa xác định"

            seg_series = df_filtered['Type2_List'].apply(classify_segment)
            count_t2 = seg_series.value_counts().reset_index()
            count_t2.columns = ['Phân khúc', 'Số lượng']
            
            fig_t2 = px.pie(count_t2, values='Số lượng', names='Phân khúc', hole=0.5,
                            color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_t2, use_container_width=True)

        with c_seg2:
            st.markdown("**🎭 Thị phần Môn Đào tạo**")
            df_explode_type1 = df_filtered.explode('Type1_List')
            if not df_explode_type1.empty:
                count_t1 = df_explode_type1['Type1_List'].value_counts().reset_index().head(10) # Top 10
                count_t1.columns = ['Môn học', 'Số lượng']
                fig_t1 = px.bar(count_t1, x='Số lượng', y='Môn học', orientation='h', 
                                text_auto=True, color='Số lượng', color_continuous_scale='Blues')
                fig_t1.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_t1, use_container_width=True)

        st.markdown("---")

        # ==========================
        # SECTION 2: STRATEGY
        # ==========================
        st.subheader("2. 🔗 Phân tích Chiến lược Sản phẩm")
        
        col_matrix, col_top_centers = st.columns([1, 1])
        
        # --- LEFT: COMBO MATRIX ---
        with col_matrix:
            st.markdown("**🧩 Ma trận Combo Môn học**")
            st.caption("Các môn học nào thường đi kèm với nhau?")
            
            transactions = df_filtered['Type1_List'].tolist()
            pairs = []
            for t in transactions:
                cleaned_t = [x for x in t if x in all_type1]
                if len(cleaned_t) > 1:
                    pairs.extend(combinations(sorted(cleaned_t), 2))
            
            if pairs:
                pair_counts = Counter(pairs)
                nodes = sorted(list(set([item for sublist in pairs for item in sublist])))
                matrix_df = pd.DataFrame(0, index=nodes, columns=nodes)
                
                for (a, b), count in pair_counts.items():
                    matrix_df.at[a, b] = count
                    matrix_df.at[b, a] = count
                
                fig_mx = px.imshow(matrix_df, text_auto=True, color_continuous_scale='Reds', aspect="auto")
                st.plotly_chart(fig_mx, use_container_width=True)
            else:
                st.info("Không đủ dữ liệu đa môn để vẽ ma trận tương quan.")

        # --- RIGHT: TOP CENTERS ---
        with col_top_centers:
            st.markdown("**🏆 Top Trung tâm Đa dạng Dịch vụ**")
            st.caption("Các đơn vị cung cấp nhiều loại hình đào tạo nhất")
            
            top_centers = df_filtered[['Name', 'Service_Count']].sort_values(by='Service_Count', ascending=False).head(15)
            
            if not top_centers.empty:
                fig_top = px.bar(
                    top_centers,
                    x='Service_Count',
                    y='Name',
                    orientation='h',
                    text_auto=True,
                    color='Service_Count',
                    color_continuous_scale='YlOrRd',
                    labels={'Service_Count': 'Số lượng môn', 'Name': 'Tên trung tâm'}
                )
                fig_top.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu.")

        st.markdown("---")
        
        # ==========================
        # SECTION 3: GEOGRAPHY (HEATMAP ONLY)
        # ==========================
        st.subheader("3. 🔥 Bản đồ nhiệt Thị trường")
        st.caption("Cường độ cạnh tranh tại các Khu vực")
        
        heat_data = pd.crosstab(df_explode_type1['District'], df_explode_type1['Type1_List'])
        if not heat_data.empty:
            fig_heat = px.imshow(heat_data, text_auto=True, aspect="auto", color_continuous_scale="Oranges",
                                 labels=dict(x="Môn học", y="Khu vực", color="Số lượng"))
            fig_heat.update_xaxes(side="top")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu phân bố.")

        st.markdown("---")

        # ==========================
        # SECTION 4: BRANDING
        # ==========================
        st.subheader("4. 🏷️ Xu hướng Đặt tên (Branding)")
        st.caption("Các từ khoá xuất hiện nhiều nhất trong tên Thương hiệu")
        
        text_data = " ".join(df_filtered['Name'].astype(str).tolist()).lower()
        stopwords = ['trung', 'tâm', 'đào', 'tạo', 'nghệ', 'thuật', 'âm', 'nhạc', 'music', 'center', 'hà', 'nội', 'của', 'và', 'lớp', 'học', 'dạy', 'clb', 'câu', 'lạc', 'bộ']
        words = text_data.split()
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        word_counts = Counter(filtered_words).most_common(20)
        wc_df = pd.DataFrame(word_counts, columns=['Keyword', 'Frequency'])
        
        if not wc_df.empty:
            fig_wc = px.bar(wc_df, x='Keyword', y='Frequency', color='Frequency', 
                            color_continuous_scale='Viridis', title="Top Keywords")
            st.plotly_chart(fig_wc, use_container_width=True)
        else:
            st.info("Không đủ dữ liệu text để phân tích.")

        st.markdown("---")

        # ==========================
        # SECTION 5: TOP DISTRICTS (NEW, AT THE BOTTOM)
        # ==========================
        st.subheader("5. 🏙️ Xếp hạng Khu vực (District)")
        st.caption("Danh sách các Quận/Huyện có nhiều trung tâm nhất")
        
        dist_counts = df_filtered['District'].value_counts().reset_index()
        dist_counts.columns = ['Khu vực', 'Số lượng']
        
        if not dist_counts.empty:
            # Xử lý max_val an toàn tránh lỗi NaN
            max_val = dist_counts['Số lượng'].max()
            if pd.isna(max_val) or max_val == 0: max_val = 1
            
            st.dataframe(
                dist_counts,
                column_config={
                    "Khu vực": st.column_config.TextColumn("Tên Quận/Huyện"),
                    "Số lượng": st.column_config.ProgressColumn(
                        "Số lượng cơ sở",
                        format="%d",
                        min_value=0,
                        max_value=int(max_val),
                    ),
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
        else:
            st.info("Chưa có dữ liệu khu vực.")

    else:
        st.warning("Vui lòng chọn bộ lọc để xem phân tích dữ liệu.")