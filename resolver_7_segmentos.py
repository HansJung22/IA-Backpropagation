# /projeto_backpropagation/resolver_7_segmentos.py

import numpy as np
from src import RedeNeural, carregar_dados_7_segmentos
from src.dados_util import carregar_dados_segmentos_ruido
import random

def adicionar_ruido(x):
    """
    Simula uma falha de segmento (Requisito RF-5).
    Inverte (flip) um único bit aleatório na entrada 'x'.
    """
    x_ruidoso = np.copy(x)
    # Escolhe um índice aleatório (0 a 6) para inverter
    idx_ruido = random.randint(0, len(x_ruidoso) - 1)
    
    # Inverte o bit (0 -> 1 ou 1 -> 0)
    x_ruidoso[idx_ruido] = 1.0 - x_ruidoso[idx_ruido]
    
    return x_ruidoso, idx_ruido

def interpretar_previsao(previsao_one_hot):
    """
    Converte a saída one-hot (vetor de 10 posições) no dígito previsto.
    O dígito previsto é o índice do neurônio com a maior ativação.
    """
    return np.argmax(previsao_one_hot)

def obter_configuracoes_usuario():
    """Solicita ao usuário os parâmetros de configuração."""
    print("--- Configuração da Rede Neural para 7-Segmentos ---")
    
    # Taxa de aprendizado configurável
    while True:
        try:
            taxa = float(input("Digite a taxa de aprendizado (ex: 0.05): "))
            if taxa > 0:
                break
            print("Por favor, digite um número positivo.")
        except ValueError:
            print("Entrada inválida.")
            
    while True:
        try:
            epocas = int(input("Digite o número de épocas de treinamento (ex: 5000): "))
            if epocas > 0:
                break
            print("Por favor, digite um número inteiro positivo.")
        except ValueError:
            print("Entrada inválida.")

    print("-" * 50)
    return taxa, epocas

def main():
    # 1. Carregar Dados
    X, y = carregar_dados_7_segmentos()
    
    # 2. Obter Configurações
    taxa_aprendizado, epocas = obter_configuracoes_usuario()
    
    # 3. Definir Arquitetura (Fixa pelo PDF: 7-5-10)
    # 7 entradas, 5 ocultos, 10 saídas (corrigido)
    arquitetura_7seg = [7, 5, 10]
    
    # 4. Instanciar e Treinar a Rede
    print(f"Iniciando treinamento com arquitetura {arquitetura_7seg} e taxa={taxa_aprendizado}...")
    rede_7seg = RedeNeural(arquitetura_7seg, taxa_aprendizado=taxa_aprendizado)
    rede_7seg.treinar(X, y, epocas)
    print("Treinamento concluído.")
    print("-" * 50)

    # 5. Testar com Dados Limpos (Sem ruído)
    print("--- Resultados da Previsão (Dados Limpos) ---")
    print("Estes são os 'prints de entrada e saída' para o relatório.")
    acertos = 0
    for i in range(len(X)):
        x_teste = X[i]
        y_esperado_vetor = y[i]
        
        digito_esperado = interpretar_previsao(y_esperado_vetor)
        
        previsao_raw = rede_7seg.prever(x_teste)
        digito_previsto = interpretar_previsao(previsao_raw)
        
        status = "CORRETO" if digito_previsto == digito_esperado else "INCORRETO"
        if status == "CORRETO":
            acertos += 1
            
        print(f"Dígito: {digito_esperado} | Previsto: {digito_previsto} | Status: {status}")
    
    acurácia = acertos * 100 / len(X)
    print("-" * 50)
    if acurácia == 100.0:
        print("->RESULTADO SEM RUIDO: Acertou")
    else:
        print("->RESULTADO SEM RUIDO: Errou")
    print("-" * 50)
    X, y = carregar_dados_segmentos_ruido()
    # 6. Testar Robustez com Ruído
    print("--- Resultados da Previsão (Teste com Ruído)  ---")
    acertos_ruido = 0
    for i in range(len(X)):
        x_original = X[i]
        y_esperado_vetor = y[i]
        digito_esperado = interpretar_previsao(y_esperado_vetor)
        
        previsao_raw_ruido = rede_7seg.prever(x_original)
        digito_previsto_ruido = interpretar_previsao(previsao_raw_ruido)

        status = "CORRETO" if digito_previsto_ruido == digito_esperado else "INCORRETO"
        if status == "CORRETO":
            acertos_ruido += 1

        print(f"Dígito: {digito_esperado} | Previsto: {digito_previsto_ruido} | Status: {status}")
    acurácia = acertos_ruido * 100 / len(X)
    print("-" * 50)
    if acurácia == 100.0:
        print("->RESULTADO COM RUIDO: Acertou")
    else:
        print("->RESULTADO COM RUIDO: Errou")
    print("-" * 50)

if __name__ == "__main__":
    main()