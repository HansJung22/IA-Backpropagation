#!/bin/bash

# ===================================================================
# SCRIPT DE REPETIÇÃO - TESTE XOR
# Executa o 'testar_xor.sh' N vezes para análise estatística.
# ===================================================================

# --- CONFIGURAÇÃO ---
# Defina o número de repetições (N)
N_EXECUCOES=30

# O arquivo de log (deve ser o MESMO do script 'testar_xor.sh')
ARQUIVO_DE_SAIDA="resultados_xor.txt"
# --- FIM DA CONFIGURAÇÃO ---


echo "Iniciando teste de robustez do XOR..."
echo "O script './testar_xor.sh' será executado $N_EXECUCOES vezes."
echo "Os resultados serão anexados em: $ARQUIVO_DE_SAIDA"
echo ""

# Limpa o arquivo de log antigo ANTES de começar o loop
# (Se você quiser começar um novo lote de testes)
# Comente a linha abaixo se quiser manter os logs de lotes anteriores.
echo "Limpando log antigo... (Iniciando lote de $N_EXECUCOES execuções)" > $ARQUIVO_DE_SAIDA


# Loop 'for' (estilo C) para repetir N vezes
for ((i=1; i<=$N_EXECUCOES; i++))
do
    # Imprime o progresso no TERMINAL (isso não vai para o arquivo de log)
    echo "Executando (XOR) teste $i de $N_EXECUCOES..."

    # Chama o script de teste. 
    # O 'testar_xor.sh' já está configurado para ANEXAR (>>) ao log.
    ./testar_xor.sh
done

echo ""
echo "Teste de robustez do XOR concluído."
echo "Resultados salvos em $ARQUIVO_DE_SAIDA"