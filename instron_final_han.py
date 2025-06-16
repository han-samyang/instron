import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- 0. 앱 기본 정보 ---
st.set_page_config(page_title="Analysis System for Polymer Stress Relaxation", layout="wide")
st.title("🔬 Analysis System for Polymer Stress Relaxation")
st.markdown("재료의 히스테리시스 데이터를 분석하고, 자동 분석 또는 수동 계산기 기능을 사용해 보세요.")

# --- st.session_state 초기화 ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# --- 정보 섹션 ---
with st.expander("ℹ️ 앱 사용 가이드 및 이론"):
    st.subheader("1. 히스테리시스 면적 (에너지 손실)")
    st.markdown("그래프 루프의 면적은 한 사이클 동안 재료 내에서 열에너지로 소산된 양을 나타냅니다. 이 앱에서는 **NumPy의 사다리꼴 공식(Trapezoidal Rule, `np.trapz`)**을 이용한 수치 적분으로 면적을 계산합니다.")
    st.markdown("계산식: `로딩 면적 - |언로딩 면적|`")
    st.info("단위: 응력(y축)이 MPa, 변형률(x축)이 **단위 없음(%)**으로 계산되므로, 면적의 단위는 **MJ/m³ (메가줄 퍼 세제곱미터)**가 됩니다.")
    st.markdown("**중요**: 언로딩 면적은 하강으로 인해 음수로 계산되지만, 면적은 이론적으로 음수가 불가능하므로 절댓값을 취합니다.")
    st.subheader("2. 회복탄성률 (Resilience)")
    st.markdown("자동 분석에서는 최대 변형률 50% 지점의 응력 회복률을, 수동 계산기에서는 사용자가 직접 입력한 값으로 응력 유지율을 계산합니다.")
    st.subheader("3. 응력 유지율 (Stress Retention Rate)")
    st.markdown("응력 유지율은 (ε_p / ε_max) × 100으로 계산됩니다. 여기서:")
    st.markdown("- **ε_max**: Load 곡선에서 응력이 0.4 MPa 이하인 변형률을 최대 변형률에서 뺀 후, 그 값의 50%를 최대 변형률에서 뺀 지점(W)에서의 응력(MPa)")
    st.markdown("- **ε_p**: Unload 곡선에서 W 지점 변형률에서의 응력(MPa)")

# --- 1. 데이터 처리 함수들 ---
@st.cache_data
def generate_demo_data():
    num_cycles=5
    full_df_list = []
    for i in range(1, num_cycles + 1):
        points_per_leg=100
        current_max_strain = (150 - (i-1)*10)
        permanent_set = (i-1) * 2
        loading_strain = np.linspace(permanent_set, current_max_strain, points_per_leg)
        loading_stress = 25 * (((loading_strain-permanent_set)/100)**0.6) * (1 - (i-1) * 0.05)
        unloading_strain = np.linspace(current_max_strain, permanent_set + 1, points_per_leg)
        max_stress = loading_stress[-1]
        unloading_stress = max_stress * (((unloading_strain - (permanent_set+1)) / (current_max_strain - (permanent_set+1)))**1.2)
        unloading_stress[unloading_stress < 0] = 0
        cycle_df = pd.DataFrame({
            'Strain': np.concatenate([loading_strain, unloading_strain]),
            'Stress (MPa)': np.concatenate([loading_stress, unloading_stress])
        })
        cycle_df['Cycle'] = i
        full_df_list.append(cycle_df)
    return pd.concat(full_df_list, ignore_index=True)

def read_csv_file(uploaded_file):
    uploaded_file.seek(0)
    try:
        first_line = uploaded_file.readline().decode('utf-8', errors='ignore')
        uploaded_file.seek(0)
        if '시간' in first_line and '(s)' in first_line:
            df_raw = pd.read_csv(uploaded_file, header=None, encoding='utf-8')
            header = df_raw.iloc[0].astype(str) + " " + df_raw.iloc[1].astype(str)
            df = df_raw[2:].copy()
            df.columns = header
        else:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        st.warning("UTF-8 인코딩 실패. CP949로 재시도합니다.")
        uploaded_file.seek(0)
        first_line = uploaded_file.readline().decode('cp949', errors='ignore')
        uploaded_file.seek(0)
        if '시간' in first_line and '(s)' in first_line:
            df_raw = pd.read_csv(uploaded_file, header=None, encoding='cp949')
            header = df_raw.iloc[0].astype(str) + " " + df_raw.iloc[1].astype(str)
            df = df_raw[2:].copy()
            df.columns = header
        else:
            df = pd.read_csv(uploaded_file, encoding='cp949')
    except Exception as e:
        raise ValueError(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")

    df.reset_index(drop=True, inplace=True)
    return df

def detect_cycles(df, strain_col):
    strain = pd.to_numeric(df[strain_col], errors='coerce')
    valleys = df[(strain.shift(1) > strain) & (strain.shift(-1) > strain)]
    valley_indices = valleys.index.tolist()

    if not valley_indices:
        df['Cycle'] = 1
        return df, 1

    df['Cycle'] = 0
    start_idx = 0
    cycle_count = 1
    for end_idx in valley_indices:
        df.loc[start_idx:end_idx-1, 'Cycle'] = cycle_count
        start_idx = end_idx
        cycle_count += 1
    
    if start_idx < len(df) - 1:
        df.loc[start_idx:, 'Cycle'] = cycle_count

    num_cycles = df['Cycle'].max()
    return df, int(num_cycles)


@st.cache_data
def parse_data(df, strain_col, stress_col, cycle_col=None, is_demo=False):
    if is_demo:
        df_with_cycles = generate_demo_data()
        cycle_col = 'Cycle'
    else:
        df_parsed = df.copy()
        df_parsed[strain_col] = pd.to_numeric(df_parsed[strain_col], errors='coerce')
        df_parsed[stress_col] = pd.to_numeric(df_parsed[stress_col], errors='coerce')
        df_parsed.dropna(subset=[strain_col, stress_col], inplace=True)

        # [수정] 음수 응력(Stress) 값과 관련 노이즈 제거
        df_parsed = df_parsed[df_parsed[stress_col] >= 0].copy()
        df_parsed.reset_index(drop=True, inplace=True)

        # [수정] 기울기 기반으로 시작점 노이즈 제거
        if len(df_parsed) > 20: 
            stress_smooth = df_parsed[stress_col].rolling(window=5, center=True, min_periods=1).mean()
            stress_threshold = max(0.05, stress_smooth.max() * 0.01) # 최소 0.05MPa 또는 최대치의 1%
            
            first_valid_idx = (stress_smooth > stress_threshold).idxmax()
            
            if first_valid_idx > 0:
                df_parsed = df_parsed.loc[first_valid_idx:].copy()

        df_parsed.reset_index(drop=True, inplace=True)

        if cycle_col is None:
            df_with_cycles, _ = detect_cycles(df_parsed, strain_col)
            cycle_col = 'Cycle'
        else:
            df_with_cycles = df_parsed
            
        df_with_cycles[strain_col] = df_with_cycles[strain_col] * 100

    df_renamed = df_with_cycles.rename(columns={strain_col: 'Strain', stress_col: 'Stress (MPa)', cycle_col: 'Cycle'})
    df_renamed['Strain'] = pd.to_numeric(df_renamed['Strain'], errors='coerce')
    df_renamed['Stress (MPa)'] = pd.to_numeric(df_renamed['Stress (MPa)'], errors='coerce')
    df_renamed.dropna(inplace=True)
    all_cycles_data = {}
    unique_cycles = sorted(df_renamed['Cycle'].unique())
    for cycle_num in unique_cycles:
        if cycle_num == 0: continue
        cycle_df = df_renamed[df_renamed['Cycle'] == cycle_num].copy().reset_index(drop=True)
        if not cycle_df.empty:
            all_cycles_data[f'Cycle {cycle_num}'] = cycle_df
    return all_cycles_data

def analyze_cycle(df):
    if df.empty or len(df) < 3: return None
        
    max_strain_idx = df['Strain'].idxmax()
    loading_df = df.iloc[:max_strain_idx + 1].copy()
    unloading_df = df.iloc[max_strain_idx:].copy()
    
    if loading_df.empty or unloading_df.empty or len(loading_df) < 2 or len(unloading_df) < 2: return None
    
    # === 수정된 히스테리시스 면적 계산 ===
    w_loading = np.trapz(loading_df['Stress (MPa)'], loading_df['Strain'] / 100)
    w_unloading = np.trapz(unloading_df['Stress (MPa)'], unloading_df['Strain'] / 100)
    
    # 히스테리시스 면적 = 로딩 면적 - |언로딩 면적|
    # 언로딩 면적은 하강으로 인해 음수가 나오므로 절댓값을 취함
    hysteresis_area = w_loading - abs(w_unloading)

    max_strain = df['Strain'].max()
    target_strain_resilience = max_strain / 2
    loading_stress_at_target = np.interp(target_strain_resilience, loading_df['Strain'], loading_df['Stress (MPa)'])
    unloading_stress_at_target = np.interp(target_strain_resilience, np.flip(unloading_df['Strain']), np.flip(unloading_df['Stress (MPa)']))
    stress_resilience = (unloading_stress_at_target / loading_stress_at_target) * 100 if loading_stress_at_target > 0 else 0

    filtered_loading_df = loading_df[loading_df['Stress (MPa)'] > 0.4]
    if not filtered_loading_df.empty:
        min_filtered_strain = filtered_loading_df['Strain'].min()
        filtered_range = max_strain - min_filtered_strain
        w_strain_value = max_strain - (filtered_range / 2) 
        ε_max = np.interp(w_strain_value, loading_df['Strain'], loading_df['Stress (MPa)'])
        ε_max_strain = w_strain_value
    else:
        ε_max = 0; ε_max_strain = 0
    
    if not unloading_df.empty and len(unloading_df) > 1:
        ε_p = np.interp(ε_max_strain, np.flip(unloading_df['Strain']), np.flip(unloading_df['Stress (MPa)']))
        ε_p_strain = ε_max_strain
    else:
        ε_p = 0; ε_p_strain = ε_max_strain
    
    stress_retention = (ε_p / ε_max) * 100 if ε_max > 0 else 0

    return {
        "Max Stress (MPa)": df['Stress (MPa)'].max(), "Max Strain": max_strain,
        "Hysteresis Area (MJ/m³)": hysteresis_area, "Resilience (%)": stress_resilience,
        "W_loading (MJ/m³)": w_loading, "W_unloading (MJ/m³)": w_unloading,
        "Target Strain for Resilience (%)": target_strain_resilience,
        "Loading Stress at Target (MPa)": loading_stress_at_target,
        "Unloading Stress at Target (MPa)": unloading_stress_at_target,
        "ε_max (MPa)": ε_max, "ε_p (MPa)": ε_p,
        "Stress Retention Rate (%)": stress_retention,
        "ε_max Strain": ε_max_strain, "ε_p Strain": ε_p_strain
    }

def display_auto_calculation_details(selected_data):
    st.header(f"🔍 자동 분석 결과: `{selected_data['Cycle']}`")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("에너지 손실 (Hysteresis Area)")
        w_load = selected_data['W_loading (MJ/m³)']; w_unload = selected_data['W_unloading (MJ/m³)']
        st.metric(label="로딩 구간 면적 (W_loading)", value=f"{w_load:,.4f} MJ/m³")
        st.metric(label="언로딩 구간 면적 (W_unloading)", value=f"{w_unload:,.4f} MJ/m³", help="변형률이 감소하므로 적분값이 음수로 계산됩니다.")
        st.metric(label="언로딩 면적 절댓값", value=f"{abs(w_unload):,.4f} MJ/m³", help="면적은 이론적으로 음수가 불가능하므로 절댓값을 사용합니다.")
        st.metric(label="➡️ 히스테리시스 면적", value=f"{selected_data['Hysteresis Area (MJ/m³)']:,.4f} MJ/m³", help=f"계산: {w_load:,.4f} - {abs(w_unload):,.4f} = {w_load - abs(w_unload):,.4f}")
    with col2:
        st.subheader("회복탄성률 (at 50% max strain)")
        target_strain_val = selected_data['Target Strain for Resilience (%)']; loading_stress_val = selected_data['Loading Stress at Target (MPa)']; unloading_stress_val = selected_data['Unloading Stress at Target (MPa)']; resilience_val = selected_data['Resilience (%)']
        st.metric(label=f"A. Loading Stress at x={target_strain_val:.2f}%", value=f"{loading_stress_val:.3f} MPa")
        st.metric(label=f"B. Unloading Stress at x={target_strain_val:.2f}%", value=f"{unloading_stress_val:.3f} MPa")
        st.metric(label="🔄 회복탄성률", value=f"{resilience_val:.2f} %", help=f"계산: ({unloading_stress_val:.3f} / {loading_stress_val:.3f}) * 100")
    
    st.subheader("응력 유지율 (Stress Retention Rate)")
    ε_max = selected_data['ε_max (MPa)']; ε_p = selected_data['ε_p (MPa)']; ε_max_strain = selected_data['ε_max Strain']
    st.metric(label=f"W 지점 변형률", value=f"{ε_max_strain:.2f} %")
    st.metric(label=f"A. Loading 응력 at W (ε_max)", value=f"{ε_max:.3f} MPa")
    st.metric(label=f"B. Unloading 응력 at W (ε_p)", value=f"{ε_p:.3f} MPa")
    st.metric(label="응력 유지율", value=f"{(ε_p/ε_max * 100) if ε_max > 0 else 0:.2f} %", help=f"계산: ({ε_p:.3f} / {ε_max:.3f}) * 100")

# --- 앱 메인 로직 ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.sidebar.header("📂 Data Source")
    use_demo_data = st.sidebar.toggle("데모 데이터 사용", value=False, key="demo_toggle")
    if use_demo_data:
        if st.sidebar.button("데모 데이터로 분석 시작", use_container_width=True):
            with st.spinner('데이터 분석 중...'):
                st.session_state.processed_data = parse_data(None, None, None, is_demo=True)
                st.session_state.uploaded_file_name = "Demo Data"
            st.rerun()
    else:
        uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
        if uploaded_file:
            raw_df = read_csv_file(uploaded_file)
            st.sidebar.header("🔗 Column & Cycle Settings")
            file_columns = raw_df.columns.tolist()
            auto_detect = st.sidebar.checkbox("사이클 자동 감지", value=True, help="변형률의 국소 최솟값을 기준으로 사이클을 자동으로 나눕니다.")
            cycle_col_select = None
            if not auto_detect:
                cycle_col_select = st.sidebar.selectbox("Cycle Column", file_columns)
            
            try: default_strain_col = [col for col in file_columns if "인장변형" in col][0]; default_strain_idx = file_columns.index(default_strain_col)
            except IndexError: default_strain_idx = 0
            try: default_stress_col = [col for col in file_columns if "인장 강도" in col][0]; default_stress_idx = file_columns.index(default_stress_col)
            except IndexError: default_stress_idx = 1
                
            strain_col_select = st.sidebar.selectbox("Strain Column", file_columns, index=default_strain_idx)
            stress_col_select = st.sidebar.selectbox("Stress (MPa) Column", file_columns, index=default_stress_idx)
            
            if st.sidebar.button("데이터 분석 시작", use_container_width=True):
                with st.spinner('데이터 분석 중...'):
                    st.session_state.processed_data = parse_data(raw_df, strain_col_select, stress_col_select, cycle_col=cycle_col_select)
                    st.session_state.uploaded_file_name = uploaded_file.name
                st.rerun()
    st.info("데이터를 업로드하고 사이드바에서 [데이터 분석 시작] 버튼을 누르세요.")

else:
    data = st.session_state.processed_data
    if not data:
        st.error("데이터에서 사이클을 찾을 수 없습니다. 다른 파일을 사용하거나 사이클 자동 감지를 해제하고 직접 지정해주세요.")
        if st.button("새 파일로 다시 분석하기"):
            for key in list(st.session_state.keys()):
                if key != 'theme': del st.session_state[key]
            st.rerun()
        st.stop()

    analysis_results = {name: analyze_cycle(df) for name, df in data.items() if df is not None}
    analysis_results = {k: v for k, v in analysis_results.items() if v is not None}
    
    if not analysis_results:
        st.error("분석 가능한 사이클이 없습니다. 데이터 형식을 확인해주세요.")
        if st.button("새 파일로 다시 분석하기"):
            for key in list(st.session_state.keys()):
                if key != 'theme': del st.session_state[key]
            st.rerun()
        st.stop()
        
    analysis_df = pd.DataFrame(analysis_results).T.reset_index().rename(columns={'index': 'Cycle'})
    analysis_df['Cumulative Hysteresis (MJ/m³)'] = analysis_df['Hysteresis Area (MJ/m³)'].cumsum()

    st.sidebar.header("📊 Chart Controls")
    if st.session_state.get('uploaded_file_name'):
        st.sidebar.success(f"현재 분석 중인 파일: **{st.session_state.uploaded_file_name}**")
    else:
        st.sidebar.success("현재 데모 데이터로 분석 중입니다.")
    
    cycle_names = list(data.keys())
    selected_cycles = st.sidebar.multiselect("보고 싶은 Cycle을 선택하세요:", options=cycle_names, default=cycle_names, key="cycle_selector")

    if st.sidebar.button("새 파일로 다시 분석하기", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            if key not in ['theme']: del st.session_state[key]
        st.rerun()

    st.header("📈 Hysteresis Loop Visualization")
    fig = go.Figure()
    if not selected_cycles:
        st.warning("사이드바에서 하나 이상의 Cycle을 선택해주세요.")
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, cycle_name in enumerate(selected_cycles):
            if cycle_name in data:
                df = data[cycle_name]
                color = colors[i % len(colors)]
                max_strain_idx = df['Strain'].idxmax()
                loading_df = df.iloc[:max_strain_idx + 1]
                unloading_df = df.iloc[max_strain_idx:]
                
                # === 기울기 기반 노이즈 제거: 임계점 낮춤 (덜 aggressive) ===
                clean_loading_df = loading_df.copy()
                clean_unloading_df = unloading_df.copy()
                
                if len(loading_df) > 1:
                    loading_stress_diff = np.diff(loading_df['Stress (MPa)'])
                    slope_threshold = 0.002  # 0.01 → 0.002로 낮춤 (노이즈 제거 완화)
                    valid_loading_mask = loading_stress_diff > slope_threshold
                    
                    loading_start_idx = 0
                    for j, is_rising in enumerate(valid_loading_mask):
                        if is_rising:
                            loading_start_idx = j
                            break
                    
                    clean_loading_df = loading_df.iloc[loading_start_idx:].copy()
                    
                    if len(clean_loading_df) > 0:
                        loading_start_strain = clean_loading_df['Strain'].iloc[0]
                        mask = unloading_df['Strain'] >= loading_start_strain
                        clean_unloading_df = unloading_df[mask].copy()
                
                # === 선택한 그래프는 자동으로 색칠 ===
                show_fill = True  # 선택된 그래프는 항상 색칠
                
                # === 'toself' 방식으로 히스테리시스 면적을 정확하게 시각화 ===
                if len(clean_loading_df) > 0 and len(clean_unloading_df) > 0 and show_fill:
                    # --- 1. 히스테리시스 루프를 위한 닫힌 경로 데이터 생성 ---
                    # Loading 경로 (앞부분)와 Unloading 경로 (뒷부분)를 합쳐 하나의 폴리곤을 만듭니다.
                    # Unloading 데이터는 이미 Strain이 감소하는 순서이므로 그대로 이어붙이면 됩니다.
                    loop_x = list(clean_loading_df['Strain']) + list(clean_unloading_df['Strain'])
                    loop_y = list(clean_loading_df['Stress (MPa)']) + list(clean_unloading_df['Stress (MPa)'])
                    
                    # --- 2. 닫힌 경로 내부를 채우는 trace 추가 (핵심!) ---
                    # 이 trace는 오직 '채우기' 역할만 담당합니다. 선(line)은 보이지 않게 처리합니다.
                    fig.add_trace(go.Scatter(
                        x=loop_x,
                        y=loop_y,
                        fill="toself",  # 🎯 이 옵션이 닫힌 경로의 내부를 채워줍니다.
                        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),  # 선 색을 투명하게 만듦
                        hoverinfo="none",  # 마우스를 올려도 정보가 나타나지 않음
                        showlegend=False,  # 범례에 표시하지 않음
                        name=f'{cycle_name}_fill',  # 고유 이름 부여
                        legendgroup=cycle_name  # 같은 그룹으로 묶어서 함께 제어
                    ))
                
                # --- 3. 기존의 Loading/Unloading 선을 그대로 그려줍니다 (시각적 표현) ---
                # Loading 곡선 (실선)
                if len(clean_loading_df) > 0:
                    fig.add_trace(go.Scatter(
                        x=clean_loading_df['Strain'],
                        y=clean_loading_df['Stress (MPa)'],
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f'{cycle_name} - Loading',
                        legendgroup=cycle_name,
                        hovertemplate="<b>%{fullData.name}</b><br>Strain: %{x:.2f}%<br>Stress: %{y:.3f} MPa<extra></extra>"
                    ))
                    
                    # Unloading 곡선 (점선)
                    fig.add_trace(go.Scatter(
                        x=clean_unloading_df['Strain'],
                        y=clean_unloading_df['Stress (MPa)'],
                        mode='lines',
                        line=dict(color=color, width=2, dash='dash'),
                        name=f'{cycle_name} - Unloading',
                        legendgroup=cycle_name,
                        hovertemplate="<b>%{fullData.name}</b><br>Strain: %{x:.2f}%<br>Stress: %{y:.3f} MPa<extra></extra>"
                    ))
                else:
                    # fallback: 노이즈 제거 실패 시 원래 방식
                    fig.add_trace(go.Scatter(x=loading_df['Strain'], y=loading_df['Stress (MPa)'], mode='lines', line=dict(color=color), name=f'{cycle_name} - Loading', legendgroup=cycle_name, hovertemplate="<b>%{fullData.name}</b><br>Strain: %{x:.2f}%<br>Stress: %{y:.3f} MPa<extra></extra>"))
                    fig.add_trace(go.Scatter(x=unloading_df['Strain'], y=unloading_df['Stress (MPa)'], mode='lines', line=dict(color=color, dash='dash'), name=f'{cycle_name} - Unloading', legendgroup=cycle_name, hovertemplate="<b>%{fullData.name}</b><br>Strain: %{x:.2f}%<br>Stress: %{y:.3f} MPa<extra></extra>"))

                if cycle_name in analysis_results:
                    cycle_analysis = analysis_results[cycle_name]
                    if cycle_analysis:
                        fig.add_trace(go.Scatter(x=[cycle_analysis['ε_max Strain']], y=[cycle_analysis['ε_max (MPa)']], mode='markers', marker=dict(symbol='circle', size=10, color=color, line=dict(width=2, color='black')), name=f'{cycle_name} - ε_max Point', legendgroup=cycle_name, showlegend=False, hovertemplate="<b>ε_max Point</b><br>Strain: %{x:.2f}%<br>Stress: %{y:.3f} MPa<extra></extra>"))
                        fig.add_trace(go.Scatter(x=[cycle_analysis['ε_p Strain']], y=[cycle_analysis['ε_p (MPa)']], mode='markers', marker=dict(symbol='square', size=10, color=color, line=dict(width=2, color='black')), name=f'{cycle_name} - ε_p Point', legendgroup=cycle_name, showlegend=False, hovertemplate="<b>ε_p Point</b><br>Strain: %{x:.2f}%<br>Stress: %{y:.3f} MPa<extra></extra>"))

        fig.update_layout(xaxis_title="Strain (%)", yaxis_title="Stress (MPa)", legend_title="Cycles", hovermode="closest", showlegend=True, xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True), yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True))

    st.plotly_chart(fig, use_container_width=True)

    st.header("📊 Analysis Summary")
    tab1, tab2, tab3 = st.tabs(["📋 Analysis Results", "📈 Hysteresis Area per Cycle", "📈 Cumulative Energy Loss"])
    with tab1:
        if not analysis_df.empty:
            cols_to_drop = ["W_loading (MJ/m³)", "W_unloading (MJ/m³)", "Target Strain for Resilience (%)", "Loading Stress at Target (MPa)", "Unloading Stress at Target (MPa)", "ε_max Strain", "ε_p Strain"]
            st.dataframe(analysis_df.drop(columns=cols_to_drop, errors='ignore'), key="analysis_table", selection_mode="single-row", on_select="rerun", hide_index=True, use_container_width=True,
                column_config={
                    "Hysteresis Area (MJ/m³)": st.column_config.NumberColumn(format="%.4f"), "Resilience (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "ε_max (MPa)": st.column_config.NumberColumn(format="%.3f"), "ε_p (MPa)": st.column_config.NumberColumn(format="%.3f"),
                    "Stress Retention Rate (%)": st.column_config.NumberColumn(format="%.2f%%"), "Cumulative Hysteresis (MJ/m³)": st.column_config.NumberColumn(format="%.4f")
                }
            )
            table_selection = st.session_state.get("analysis_table", {}).get("selection", {"rows": []})
            if table_selection["rows"]:
                selected_data = analysis_df.iloc[table_selection["rows"][0]]
                display_auto_calculation_details(selected_data)
            else:
                st.info("위 표에서 행을 선택하면, 해당 사이클의 상세 계산 과정을 볼 수 있습니다.")
    with tab2:
        fig_avg = px.bar(analysis_df, x='Cycle', y='Hysteresis Area (MJ/m³)', title="Cycle별 히스테리시스 면적", color='Cycle', text_auto='.4f')
        fig_avg.update_layout(showlegend=False)
        st.plotly_chart(fig_avg, use_container_width=True)
    with tab3:
        fig_cumulative = px.line(analysis_df, x='Cycle', y='Cumulative Hysteresis (MJ/m³)', title="사이클별 누적 에너지 손실", markers=True, labels={'Cumulative Hysteresis (MJ/m³)': '누적 에너지 손실 (MJ/m³)'})
        fig_cumulative.update_traces(line_color='#ff7f0e', marker=dict(size=8))
        st.plotly_chart(fig_cumulative, use_container_width=True)

    with st.container(border=True):
        st.header("🔬 수동 계산기 (값 직접 입력)")
        manual_cycle = st.selectbox("1. 계산의 기준이 될 사이클을 선택하세요:", options=cycle_names, key="manual_cycle_select")
        if manual_cycle:
            cycle_df = data[manual_cycle]
            st.markdown("#### 2. 아래 표 또는 그래프에서 값을 찾아 직접 입력하세요.")
            st.dataframe(cycle_df, use_container_width=True, height=250)
            st.markdown("#### 3. ε_max와 ε_p 값을 입력하세요.")
            col1, col2 = st.columns(2)
            with col1:
                ε_max_manual = st.number_input("ε_max (Loading Stress, MPa)", key="manual_ε_max", value=0.0, format="%.4f")
            with col2:
                ε_p_manual = st.number_input("ε_p (Unloading Stress, MPa)", key="manual_ε_p", value=0.0, format="%.4f")
            
            st.markdown("#### 4. 계산 결과")
            if ε_p_manual > 0 and ε_max_manual > 0:
                stress_retention_manual = (ε_p_manual / ε_max_manual) * 100
                st.metric(label="계산된 응력 유지율", value=f"{stress_retention_manual:.2f} %", help=f"계산: ({ε_p_manual:.4f} / {ε_max_manual:.4f}) * 100")
            elif ε_max_manual == 0 and ε_p_manual > 0:
                st.error("ε_max (Loading Stress)가 0입니다. 계산할 수 없습니다.")
            else:
                st.info("ε_max와 ε_p 값을 입력하면 결과가 표시됩니다.")