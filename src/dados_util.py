# /projeto_backpropagation/src/dados_util.py

import numpy as np

def carregar_dados_xor():
    """
    Carrega o conjunto de dados para o problema XOR.
    
    Retorna:
        tuple: (X_xor, y_xor)
            X_xor (np.ndarray): Entradas do XOR (shape 4x2).
            y_xor (np.ndarray): Saídas esperadas do XOR (shape 4x1).
    """
    X_xor = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)
    
    y_xor = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)
    
    return X_xor, y_xor

def carregar_dados_7_segmentos():
    """
    Carrega o conjunto de dados para o problema do display de 7 segmentos.
    
    Os dados (entrada e saída) são baseados na tabela fornecida no
    documento da atividade.
    
    Retorna:
        tuple: (X_7seg, y_7seg)
            X_7seg (np.ndarray): Entradas (7 segmentos) para os 10 dígitos (shape 10x7).
            y_7seg (np.ndarray): Saídas esperadas (one-hot) (shape 10x10).
    """
    
    # Entradas: [a, b, c, d, e, f, g]
    # (Transcrito da tabela no PDF)
    X_7seg = np.array([
        # Dígito 0
        [1, 1, 1, 1, 1, 1, 0],
        # Dígito 1
        [0, 1, 1, 0, 0, 0, 0],
        # Dígito 2
        [1, 1, 0, 1, 1, 0, 1],
        # Dígito 3
        [1, 1, 1, 1, 0, 0, 1],
        # Dígito 4
        [0, 1, 1, 0, 0, 1, 1],
        # Dígito 5
        [1, 0, 1, 1, 0, 1, 1],
        # Dígito 6
        [1, 0, 1, 1, 1, 1, 1],
        # Dígito 7
        [1, 1, 1, 0, 0, 0, 0],
        # Dígito 8
        [1, 1, 1, 1, 1, 1, 1],
        # Dígito 9
        [1, 1, 1, 1, 0, 1, 1]
    ], dtype=float)
    
    # Saídas: One-hot encoding para 10 classes (0-9)
    # (Transcrito da tabela no PDF)
    y_7seg = np.array([
        # 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # 2
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # 3
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # 4
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # 5
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        # 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        # 9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=float)

    return X_7seg, y_7seg

def carregar_dados_segmentos_ruido():
    """
    Carrega o conjunto de dados para o problema do display de 7 segmentos.
    
    Os dados (entrada e saída) são baseados na tabela fornecida no
    documento da atividade.
    
    Retorna:
        tuple: (X_7seg, y_7seg)
            X_7seg (np.ndarray): Entradas (7 segmentos) para os 10 dígitos (shape 10x7).
            y_7seg (np.ndarray): Saídas esperadas (one-hot) (shape 10x10).
    """
    
    # Entradas: [a, b, c, d, e, f, g]
    # (Transcrito da tabela no PDF)
    X_7seg = np.array([
        # Dígito 0
        [1, 1, 1, 1, 1, 1, 1],
        # Dígito 1
        [0, 0, 1, 0, 0, 0, 0],
        # Dígito 2
        [1, 1, 1, 1, 1, 0, 1],
        # Dígito 3
        [0, 1, 1, 1, 0, 0, 1],
        # Dígito 4
        [0, 1, 1, 0, 1, 1, 1],
        # Dígito 5
        [1, 0, 1, 1, 0, 1, 0],
        # Dígito 6
        [0, 0, 1, 1, 1, 1, 1]
    ], dtype=float)
    
    # Saídas: One-hot encoding para 10 classes (0-9)
    # (Transcrito da tabela no PDF)
    y_7seg = np.array([
        # 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # 2
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # 3
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # 4
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # 5
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        # 6
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    ], dtype=float)

    return X_7seg, y_7seg

if __name__ == '__main__':
    # Teste rápido para verificar se os dados carregam corretamente
    
    print("--- Teste XOR ---")
    X_xor, y_xor = carregar_dados_xor()
    print("Entradas (X):")
    print(X_xor)
    print("Saídas (y):")
    print(y_xor)
    print("-" * 20)

    print("--- Teste 7 Segmentos ---")
    X_7seg, y_7seg = carregar_dados_7_segmentos()
    print(f"Entradas (X) shape: {X_7seg.shape}")
    print("Exemplo (Dígito 8):")
    print(X_7seg[8])
    print(f"Saídas (y) shape: {y_7seg.shape}")
    print("Exemplo (Dígito 8):")
    print(y_7seg[8])
    print("-" * 20)