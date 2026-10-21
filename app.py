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
    
    # Parâmetros do DBSCAN (Com tooltips para UX)
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
        "Amostras Mínimas (min_samples):", 
        min_value=2, 
        max_value=MAX_MIN_SAMPLES, 
        value=5, 
        step=1,
        help="Número mínimo de máquinas próximas para formar um cluster saudável."
    )
    
    st.divider()
    st.info("💡 **Dica de UX:** Mova ambos os sliders para o valor **máximo** para ver algo especial no gráfico!")

# ==========================================
# 4. LÓGICA DO MODELO (DBSCAN E EASTER EGG)
# ==========================================
is_easter_egg = (eps_val == MAX_EPS and min_samp == MAX_MIN_SAMPLES)

if is_easter_egg:
    # Ativa o Easter Egg gerando apenas a logo do Github como anomalias
    df_result = gerar_dados_github_logo()
    total_maq = len(df_result)
    anomalias_count = total_maq
else:
    # Lógica Normal Preditiva
    df_result = df_base.copy()
    
    # Normalização é crucial para o DBSCAN funcionar bem em escalas diferentes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_result[['Frequencia_Vibracao', 'Pico_Temperatura']])
    
    # Aplicação do algoritmo
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samp)
    df_result['Cluster'] = dbscan.fit_predict(X_scaled)
    
    # Mapear status para UX da Tabela
    df_result['Status'] = df_result['Cluster'].apply(lambda x: 'Anomalia (Manutenção)' if x == -1 else 'Operação Normal')
    
    total_maq = len(df_result)
    anomalias_count = len(df_result[df_result['Cluster'] == -1])

# ==========================================
# 5. TELA PRINCIPAL (DASHBOARD)
# ==========================================
st.title("🏭 Dashboard de Manutenção Preditiva")
st.markdown("Monitoramento de vibração e temperatura do maquinário em tempo real utilizando algoritmo não-supervisionado (**DBSCAN**).")

# Métricas de Resumo (KPIs rápidos no topo)
col1, col2, col3 = st.columns(3)
col1.metric("Total de Máquinas Monitoradas", total_maq)
col2.metric("Anomalias Detectadas", anomalias_count, delta_color="inverse", delta=anomalias_count)
col3.metric("Saúde da Planta", f"{100 - (anomalias_count/total_maq)*100:.1f}%", "- Risco" if anomalias_count > 10 else "+ Estável")

st.divider()

# ==========================================
# 6. VISUALIZAÇÃO GRÁFICA (SEABORN)
# ==========================================
st.subheader("📡 Distribuição de Sensores e Detecção")

# Configurando as cores (Preto para -1, paleta categórica para os demais)
clusters_unicos = sorted(df_result['Cluster'].unique())
paleta_cores = {}
cor_normal = sns.color_palette("Set1", n_colors=10)
idx_cor = 0

for c in clusters_unicos:
    if c == -1:
        paleta_cores[c] = 'black'  # Outliers sempre pretos
    else:
        paleta_cores[c] = cor_normal[idx_cor % 10]
        idx_cor += 1

# Plotagem
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=df_result, 
    x='Frequencia_Vibracao', 
    y='Pico_Temperatura', 
    hue='Cluster', 
    palette=paleta_cores,
    s=80 if not is_easter_egg else 120, # Bolinhas maiores no easter egg
    marker='o' if not is_easter_egg else 's', # Quadrados para simular pixels no logo
    alpha=0.8,
    ax=ax
)

# Ajustes visuais no gráfico
ax.set_title("Vibração vs Temperatura" if not is_easter_egg else "EASTER EGG UNLOCKED: Octocat Anômalo!", fontweight='bold')
ax.set_xlabel("Frequência de Vibração (Hz)")
ax.set_ylabel("Pico de Temperatura (°C)")
fig.patch.set_facecolor('none') 
ax.set_facecolor('none')

if is_easter_egg:
    ax.legend_.remove() # Esconde a legenda no easter egg
else:
    plt.legend(title='Grupo (Cluster) | -1 = Outlier', bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)

st.divider()

# ==========================================
# 7. TABELA INTERATIVA DE AÇÃO
# ==========================================
st.subheader("🚨 Máquinas com Anomalias (Ação Requerida)")
st.write("Abaixo estão listados os equipamentos que o algoritmo identificou como fora do padrão de operação e requerem inspeção.")

df_anomalias = df_result[df_result['Cluster'] == -1].copy()

if not df_anomalias.empty:
    st.dataframe(
        df_anomalias[['ID_Maquina', 'Frequencia_Vibracao', 'Pico_Temperatura', 'Status']].style.format({
            "Frequencia_Vibracao": "{:.2f} Hz",
            "Pico_Temperatura": "{:.2f} °C"
        }).applymap(lambda x: "background-color: #ffcccc; color: black", subset=['Status']),
        use_container_width=True
    )
else:
    st.success("Tudo certo! Nenhuma anomalia detectada com os parâmetros atuais de sensibilidade.")
