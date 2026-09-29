import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def carregar_e_limpar_dados(caminho_transacoes: str, caminho_cotacoes: str):
    df_transacoes = pd.read_csv(caminho_transacoes, encoding='latin1')
    
    df_transacoes['valor'] = df_transacoes.groupby('estado_cliente')['valor'].transform(
        lambda group: group.fillna(group.median())
    )
    
    df_transacoes['plataforma'] = 'Mobile'
    
    df_cotacoes = pd.read_csv(caminho_cotacoes, encoding='utf-8')
    df_cotacoes['cotacao_usd'] = df_cotacoes['cotacao_usd'].ffill().bfill()
    
    return df_transacoes, df_cotacoes

def engenharia_de_dados_temporais(df_transacoes: pd.DataFrame, df_cotacoes: pd.DataFrame):
    df_transacoes['data_transacao'] = pd.to_datetime(df_transacoes['data_transacao'])
    df_transacoes['data_transacao'] = df_transacoes['data_transacao'].dt.tz_localize('America/Sao_Paulo')
    
    df_transacoes['dia_semana'] = df_transacoes['data_transacao'].dt.day_name()
    df_transacoes['mes'] = df_transacoes['data_transacao'].dt.month
    
    df_transacoes = df_transacoes.drop_duplicates(keep='first').copy()
    
    df_cotacoes['data'] = pd.to_datetime(df_cotacoes['data'])
    df_cotacoes['data'] = df_cotacoes['data'].dt.tz_localize('America/Sao_Paulo')
    
    return df_transacoes, df_cotacoes

def filtrar_transacoes_criticas(df: pd.DataFrame) -> pd.DataFrame:
    condicao_mes = (df['mes'] == 9)
    condicao_estado = (df['estado_cliente'] == 'SP') | (df['estado_cliente'] == 'RJ')
    condicao_valor = (df['valor'] > 5000.0)
    
    return df[condicao_mes & condicao_estado & condicao_valor].copy()

def mapear_risco_e_pivotalizar(df: pd.DataFrame, risco_dict: dict) -> pd.DataFrame:
    df['nivel_risco'] = df['id_cliente'].map(risco_dict)
    
    tabela_pivot = pd.pivot_table(
        df,
        values='valor',
        index='mes',
        columns='nivel_risco',
        aggfunc='sum',
        fill_value=0,
        margins=True,
        margins_name='Total'
    )
    return tabela_pivot

def detectar_outliers_zscore(df: pd.DataFrame, limiar: float = 2.5) -> pd.DataFrame:
    mean_estado = df.groupby('estado_cliente')['valor'].transform('mean')
    std_estado = df.groupby('estado_cliente')['valor'].transform('std')
    
    df['z_score'] = (df['valor'] - mean_estado) / std_estado
    
    return df[df['z_score'] > limiar].copy()

def gerar_grafico(df_transacoes: pd.DataFrame, caminho_saida: str = 'relatorio_diario.png'):
    df_diario = (
        df_transacoes.set_index('data_transacao')
        .resample('D')['valor']
        .sum()
        .reset_index()
    )
    
    df_diario['media_movel_7d'] = df_diario['valor'].rolling(window=7, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_diario['data_transacao'], df_diario['valor'], label='Total Diário (R$)', color='#1f77b4', alpha=0.7)
    ax.plot(df_diario['data_transacao'], df_diario['media_movel_7d'], label='Média Móvel 7 Dias', color='#ff7f0e', linewidth=2.5)
    
    ax.set_ylim(bottom=0)
    ax.set_title('Desempenho Diário de Transações e Média Móvel de 7 Dias', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Data', fontsize=11)
    ax.set_ylabel('Valor Total (R$)', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    fig.savefig(caminho_saida, dpi=300)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")

if __name__ == '__main__':
    risco_dict = {
        'C100': 'Baixo',
        'C101': 'Alto',
        'C102': 'Médio',
        'C103': 'Alto',
        'C104': 'Baixo'
    }

    print("Iniciando processamento...")
    
    df_trans, df_cot = carregar_e_limpar_dados('transacoes.csv', 'cotacoes.csv')
    df_trans, df_cot = engenharia_de_dados_temporais(df_trans, df_cot)
    
    df_critico = filtrar_transacoes_criticas(df_trans)
    print(f"\nTransações críticas (Setembro, SP/RJ, > R$5000): {len(df_critico)} registros")
    
    tabela_pivot = mapear_risco_e_pivotalizar(df_trans, risco_dict)
    print("\nTabela Dinâmica (Mês x Risco):")
    print(tabela_pivot)
    
    anomalias = detectar_outliers_zscore(df_trans, limiar=2.5)
    print(f"\nAnomalias/Fraudes detectadas (Z-Score > 2.5): {len(anomalias)} registros")
    
    gerar_grafico(df_trans)
    print("\nProcessamento finalizado!")