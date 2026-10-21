import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA E UX
# ==========================================
st.set_page_config(
    page_title="PredMaint AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. GERAÇÃO DE DADOS SINTÉTICOS
# ==========================================
@st.cache_data
def gerar_dados_fabrica():
    np.random.seed(42)
    
    # Máquinas em funcionamento Normal (Cluster 1: Equipamentos pesados)
    freq_1 = np.random.normal(50, 4, 150)
    temp_1 = np.random.normal(70, 5, 150)
    
    # Máquinas em funcionamento Normal (Cluster 2: Equipamentos leves)
    freq_2 = np.random.normal(30, 3, 100)
    temp_2 = np.random.normal(45, 4, 100)
    
    # Outliers Intencionais (Anomalias espalhadas)
    freq_out = np.random.uniform(10, 80, 20)
    temp_out = np.random.uniform(20, 100, 20)
    
    freq = np.concatenate([freq_1, freq_2, freq_out])
    temp = np.concatenate([temp_1, temp_2, temp_out])
    
    df = pd.DataFrame({
        'ID_Maquina': [f"MAQ-{i:03d}" for i in range(1, len(freq) + 1)],
        'Frequencia_Vibracao': freq,
        'Pico_Temperatura': temp
    })
    return df

# Função para o Easter Egg (Logo do GitHub)
def gerar_dados_github_logo():
    # Matriz simplificada do Octocat
    logo_grid = [
        "  #       #  ",
        " ###     ### ",
        "#############",
        "#############",
        "###  ###  ###",
        "###############",
        " ############# ",
        "   #######   "
    ]
    xs, ys = [], []
    for r_idx, row in enumerate(logo_grid):
        for c_idx, char in enumerate(row):
            if char == '#':
                # Escalonando e posicionando no centro do gráfico
                xs.append((c_idx * 3) + 20) 
                ys.append(((7 - r_idx) * 4) + 40)
                
    return pd.DataFrame({
        'ID_Maquina': [f"GH-OUTLIER-{i}" for i in range(len(xs))],
        'Frequencia_Vibracao': xs,
        'Pico_Temperatura': ys,
        'Cluster': -1, # Todos forçados como anomalia (preto)
        'Status': 'Anomalia Crítica (GitHub)'
    })

# Carregar base padrão
df_base = gerar_dados_fabrica()

# ==========================================
# 3. SIDEBAR (CONTROLES DO USUÁRIO)
# ==========================================
with st.sidebar:
    st.title("⚙️ Parâmetros do Sensor")
    st.write("Ajuste a sensibilidade da Inteligência Artificial para detecção de anomalias.")
    
    st.divider()
    
    # Parâmetros do DBSCAN
    MAX_EPS = 3.0
    MAX_MIN_SAMPLES = 20
    
    eps_val = st.slider(
        "Raio de Vizinhança (eps):", 
        min_value=0.1, 
        max_value=MAX_EPS, 
        value=0.5, 
        step=0.1,
        help="Controla a distância máxima para que duas máquinas sejam consideradas do mesmo cluster. Valores menores tornam o modelo mais rigoroso."
    )
    
    min_samp = st.slider(
