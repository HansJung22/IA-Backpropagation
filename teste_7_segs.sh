#!/bin/bash

# ===================================================================
# SCRIPT DE TESTE - PROBLEMA 7-SEGMENTOS
# Executa a rede neural para o problema de 7 segmentos 
# e ANEXA o output em 'resultados_7_segmentos.txt'.
# ===================================================================

# Defina o nome do arquivo de saída
ARQUIVO_DE_SAIDA="resultados_7_segmentos.txt"

# Informa ao usuário (no terminal) que o processo começou
echo "Iniciando teste do 7-Segmentos... Resultados serão ANEXADOS em: $ARQUIVO_DE_SAIDA"

# Inicia o bloco de redirecionamento para o arquivo
# (Note o '>>' para ANEXAR)
{
    # --- Parâmetros do Teste 7-Segmentos ---
    SEG_TAXA=0.2
    SEG_EPOCAS=10000
    
    echo "================================="
    echo "Executando Teste: 7-Segmentos"
    echo "Data: $(date)"
    echo "Parâmetros: Taxa=${SEG_TAXA}, Épocas=${SEG_EPOCAS}"
    echo "=================================" 
    
    echo -e "${SEG_TAXA}\n${SEG_EPOCAS}" | python resolver_7_segmentos.py
    
    echo ""
    echo "Teste 7-Segmentos concluído."

} >> "$ARQUIVO_DE_SAIDA" 2>&1
# ^
# |--- MUDANÇA PRINCIPAL AQUI (de > para >>)

# Informa ao usuário (no terminal) que o processo terminou.
echo "Execução do 7-Segmentos concluída. Verifique $ARQUIVO_DE_SAIDA."