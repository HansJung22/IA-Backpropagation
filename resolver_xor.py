# /projeto_backpropagation/resolver_xor.py

import numpy as np
from src import RedeNeural, carregar_dados_xor

def obter_configuracoes_usuario():
    """Solicita ao usuário os parâmetros de configuração."""
    print("--- Configuração da Rede Neural para o XOR ---")
    
    # Requisito RU-5: Número de neurônios ocultos configurável
    while True:
        try:
            n_oculta = int(input("Digite o número de neurônios na camada oculta (ex: 2): "))
            if n_oculta > 0:
                break
            print("Por favor, digite um número inteiro positivo.")
        except ValueError:
            print("Entrada inválida.")

    # Requisito RU-4: Taxa de aprendizado configurável
    while True:
        try:
            taxa = float(input("Digite a taxa de aprendizado (ex: 0.1): "))
            if taxa > 0:
                break
            print("Por favor, digite um número positivo.")
        except ValueError:
            print("Entrada inválida.")
            
    while True:
        try:
            epocas = int(input("Digite o número de épocas de treinamento (ex: 10000): "))
            if epocas > 0:
                break
            print("Por favor, digite um número inteiro positivo.")
        except ValueError:
            print("Entrada inválida.")

    print("-" * 45)
    return n_oculta, taxa, epocas

def main():
    # 1. Carregar Dados
    X, y = carregar_dados_xor()
    
    
    # 2. Obter Configurações
    n_oculta, taxa_aprendizado, epocas = obter_configuracoes_usuario()
    
    # 3. Definir Arquitetura (Entrada=2, Oculta=n_oculta, Saída=1)
    arquitetura_xor = [2, n_oculta, 1]
    
    # 4. Instanciar e Treinar a Rede
    print(f"Iniciando treinamento com arquitetura {arquitetura_xor} e taxa={taxa_aprendizado}...")
    rede_xor = RedeNeural(arquitetura_xor, taxa_aprendizado=taxa_aprendizado)
    rede_xor.treinar(X, y, epocas)
    print("Treinamento concluído.")
    print("-" * 45)

    # 5. Testar e Exibir Resultados
    print("--- Resultados da Previsão (XOR) ---")
    acerto = 0
    for x_teste, y_esperado in zip(X, y):
        previsao_raw = rede_xor.prever(x_teste)
        
        # Como a saída é Sigmoide (0 a 1), arredondamos para 0 ou 1
        previsao_arredondada = 1 if previsao_raw[0][0] > 0.5 else 0
        
        print(f"Entrada: {x_teste}")
        print(f"  Esperado: {y_esperado[0]}")
        #print(f"  Previsto (Raw): {previsao_raw[0][0]:.6f}")
        print(f"  Previsto (Final): {previsao_arredondada}")
        print("---")
        

if __name__ == "__main__":
    main()