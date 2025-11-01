# /projeto_backpropagation/src/rede_neural.py

import numpy as np

class RedeNeural:
    """
    Implementação de uma Rede Neural com backpropagation "do zero".

    Contexto da Atividade:
    - Algoritmo: Backpropagation
    - Atualização de Pesos: Estocástica (a cada instância)
    - Função de Ativação: Sigmoide (em todas as camadas)
    - Função de Custo: Erro Quadrático Médio (MSE)
    - Implementação: Python + NumPy
    """

    def __init__(self, arquitetura, taxa_aprendizado=0.1):
        """
        Inicializa a rede neural.

        Args:
            arquitetura (list): Uma lista de inteiros definindo o número 
                                de neurônios em cada camada.
                                Ex: [7, 5, 10] (Entrada, Oculta, Saída)
            taxa_aprendizado (float): A taxa de aprendizado (eta) para 
                                      a atualização dos pesos.
        """
        self.arquitetura = arquitetura
        self.taxa_aprendizado = taxa_aprendizado
        self.num_camadas = len(arquitetura)
        
        # Listas para armazenar pesos e biases
        # self.pesos[i] conecta a camada i com a camada i+1
        # self.biases[i] são os biases da camada i+1
        self.pesos = []
        self.biases = []

        # Inicializa os pesos e biases
        for i in range(self.num_camadas - 1):
            # Shape dos Pesos: (neuronios_camada_seguinte, neuronios_camada_anterior)
            # Inicialização aleatória pequena (para quebrar simetria)
            w = np.random.randn(arquitetura[i+1], arquitetura[i]) * 0.1
            self.pesos.append(w)
            
            # Shape dos Biases: (neuronios_camada_seguinte, 1)
            # Inicialização com zeros
            b = np.zeros((arquitetura[i+1], 1))
            self.biases.append(b)

    # --- Funções de Ativação e Custo ---

    def _sigmoid(self, z):
        # Adicionado np.clip para evitar overflow/warning em np.exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _sigmoid_derivada(self, a):
        """
        Derivada da função Sigmoide.
        Nota: Recebe 'a' (a saída da sigmoide) para eficiência.
        a = sigmoid(z)
        sigmoid'(z) = a * (1 - a)
        """
        return a * (1 - a)

    def _mse(self, y_real, y_previsto):
        return np.mean((y_real - y_previsto) ** 2)

    # --- Funções Auxiliares ---

    def _formatar_vetor_coluna(self, v, dim):
        """Garante que o vetor de entrada/saída tenha o formato (dim, 1)."""
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        if v.shape == (dim,): # Converte de (dim,) para (dim, 1)
            v = v.reshape(dim, 1)
        return v

    # --- Núcleo do Algoritmo ---

    def forward_pass(self, x):
        """
        Executa a propagação direta (forward pass) para uma única instância 'x'.
        
        Args:
            x (np.ndarray): Vetor de entrada (formato coluna (N_entrada, 1)).
        
        Returns:
            tuple: (ativacoes, zs)
                ativacoes (list): Lista de vetores de ativação (saídas) 
                                  de cada camada (incluindo a entrada).
                zs (list): Lista de vetores de entrada ponderada (z) 
                           para cada camada (exceto a entrada).
        """
        ativacoes = [x] # Armazena a ativação da camada 0 (entrada)
        zs = []         # Armazena os 'z' (w.x + b)

        a = x
        # Itera da primeira camada oculta até a camada de saída
        for w, b in zip(self.pesos, self.biases):
            # z = (W * a_anterior) + b
            z = (w @ a) + b
            zs.append(z)
            
            # a = sigmoid(z)
            a = self._sigmoid(z)
            ativacoes.append(a)
            
        return ativacoes, zs

    def backpropagate_e_atualizar(self, y, ativacoes, zs):
        """
        Executa a retropropagação (backward pass) E atualiza os pesos
        para uma única instância (x, y). (Atualização Estocástica)
        
        Args:
            y (np.ndarray): Vetor de saída esperado (formato coluna (N_saida, 1)).
            ativacoes (list): Lista de ativações do forward_pass.
            zs (list): Lista de 'z' do forward_pass.
        """
        
        # Listas para armazenar os gradientes (derivadas parciais)
        nabla_w = [np.zeros(w.shape) for w in self.pesos]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # 1. Calcular o Erro (delta) da Camada de Saída (L)
        
        y_previsto = ativacoes[-1] # a^L
        
        # Derivada do Custo (MSE) em relação à ativação de saída (a^L)
        # dE/da^L = (a^L - y)
        # Usamos (a^L - y) pois é a derivada de 1/2 * (a^L - y)^2,
        # que é a forma comum do MSE para derivação.
        derivada_custo = (y_previsto - y)
        
        # Derivada da sigmoide em relação a z^L
        derivada_sig_L = self._sigmoid_derivada(y_previsto)
        
        # Delta da camada de saída (delta^L = dE/dz^L)
        delta = derivada_custo * derivada_sig_L
        
        # 2. Gradientes da Camada de Saída
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ ativacoes[-2].T # (delta^L) * (a^{L-1})^T

        # 3. Retropropagar o Erro
        # Itera de trás para frente, começando da penúltima camada
        # (a última camada oculta)
        for l in range(2, self.num_camadas):
            # l=2 -> penúltima camada (índice -2)
            # l=3 -> antepenúltima camada (índice -3)
            
            # z da camada atual (z^l)
            z = zs[-l]
            # ativação da camada atual (a^l)
            a = ativacoes[-l]
            
            # Derivada da sigmoide para a camada atual
            sp = self._sigmoid_derivada(a) # sigmoid'(z^l)
            
            # delta^l = (W^{l+1})^T * (delta^{l+1}) * sigmoid'(z^l)
            delta = (self.pesos[-l+1].T @ delta) * sp
            
            # 4. Gradientes da Camada Oculta Atual
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ ativacoes[-l-1].T

        # 5. Atualizar Pesos e Biases (Atualização Estocástica)
        # W = W - eta * nabla_w
        # b = b - eta * nabla_b
        for i in range(len(self.pesos)):
            self.pesos[i] = self.pesos[i] - (self.taxa_aprendizado * nabla_w[i])
            self.biases[i] = self.biases[i] - (self.taxa_aprendizado * nabla_b[i])

    # --- Métodos Públicos de Treinamento e Previsão ---

    def treinar(self, X_treino, y_treino, epocas):
        """
        Treina a rede neural usando atualização estocástica.

        Args:
            X_treino (list ou np.ndarray): Lista de todas as entradas de treino.
            y_treino (list ou np.ndarray): Lista de todas as saídas esperadas.
            epocas (int): Número de épocas para treinar.
        
        Returns:
            list: Histórico do erro (MSE) por época.
        """
        historia_erro = []
        
        dim_entrada = self.arquitetura[0]
        dim_saida = self.arquitetura[-1]
        
        for e in range(epocas):
            erro_total_epoca = 0
            
            # Embaralhar os dados a cada época (melhora a convergência)
            indices = np.arange(len(X_treino))
            np.random.shuffle(indices)
            X_shuffled = [X_treino[i] for i in indices]
            y_shuffled = [y_treino[i] for i in indices]
            
            # Loop de atualização Estocástica (RU-3)
            # Atualiza os pesos para CADA instância
            for x, y in zip(X_shuffled, y_shuffled):
                
                # Garantir que x e y sejam vetores-coluna
                x_col = self._formatar_vetor_coluna(x, dim_entrada)
                y_col = self._formatar_vetor_coluna(y, dim_saida)
                
                # 1. Forward Pass
                ativacoes, zs = self.forward_pass(x_col)
                
                # 2. Backpropagation e Atualização Estocástica
                self.backpropagate_e_atualizar(y_col, ativacoes, zs)
                
                # Acumula o erro da instância
                erro_total_epoca += self._mse(y_col, ativacoes[-1])
            
            # Calcula o erro médio da época
            mse_epoca = erro_total_epoca / len(X_treino)
            historia_erro.append(mse_epoca)
            
                
        return historia_erro

    def prever(self, x):
        """
        Faz uma previsão para uma única entrada 'x'.

        Args:
            x (list ou np.ndarray): A entrada a ser prevista.
        
        Returns:
            np.ndarray: A ativação da camada de saída (previsão).
        """
        # Garantir formato de vetor coluna
        x_col = self._formatar_vetor_coluna(x, self.arquitetura[0])
        
        # Executa o forward pass
        ativacoes, _ = self.forward_pass(x_col)
        
        # Retorna a última ativação (a previsão)
        return ativacoes[-1]