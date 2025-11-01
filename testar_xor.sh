#!/bin/bash

# ===================================================================
# SCRIPT DE TESTE - PROBLEMA XOR
# Executa a rede neural para o problema XOR com parâmetros definidos
# e ANEXA o output em 'resultados_xor.txt'.
# ===================================================================

# Defina o nome do arquivo de saída
ARQUIVO_DE_SAIDA="resultados_xor.txt"

# Informa ao usuário (no terminal) que o processo começou
echo "Iniciando teste do XOR... Resultados serão ANEXADOS em: $ARQUIVO_DE_SAIDA"

# Inicia o bloco de redirecionamento para o arquivo
# (Note o '>>' para ANEXAR)
{
    # --- Parâmetros do Teste XOR ---
    XOR_OCULTA=2
    XOR_TAXA=0.2
    XOR_EPOCAS=5000
    
    echo "================================="
    echo "Executando Teste: XOR"
    echo "Data: $(date)"
    echo "Parâmetros: Oculta=${XOR_OCULTA}, Taxa=${XOR_TAXA}, Épocas=${XOR_EPOCAS}"
    echo "================================="
    
    # ORDEM IMPORTANTE:
    # 1º input() do resolver_xor.py pede: n_oculta
    # 2º input() do resolver_xor.py pede: taxa
    # 3º input() do resolver_xor.py pede: epocas
    
    echo -e "${XOR_OCULTA}\n${XOR_TAXA}\n${XOR_EPOCAS}" | python resolver_xor.py
    
    echo ""
    echo "Teste XOR concluído."

} >> "$ARQUIVO_DE_SAIDA" 2>&1 
# ^
# |--- MUDANÇA PRINCIPAL AQUI (de > para >>)

# Informa ao usuário (no terminal) que o processo terminou.
echo "Execução do XOR concluída. Verifique $ARQUIVO_DE_SAIDA."