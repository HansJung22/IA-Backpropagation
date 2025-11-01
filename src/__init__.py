# /projeto_backpropagation/src/__init__.py

"""
Pacote 'src' para o projeto de implementação do Backpropagation.

Este pacote contém os módulos principais da aplicação:
- rede_neural: Contém a classe RedeNeural.
- dados_util: Contém funções para carregar os datasets.
"""

# Expõe a classe RedeNeural para facilitar a importação
# Agora podemos fazer: from src import RedeNeural
from .rede_neural import RedeNeural

# Expõe as funções de carregamento de dados
# Agora podemos fazer: from src import carregar_dados_xor, carregar_dados_7_segmentos
from .dados_util import carregar_dados_xor, carregar_dados_7_segmentos

# Opcional: Define o que é importado com 'from src import *'
__all__ = [
    'RedeNeural',
    'carregar_dados_xor',
    'carregar_dados_7_segmentos'
]