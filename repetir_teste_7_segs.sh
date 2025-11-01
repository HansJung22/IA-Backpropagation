#!/bin/bash

# ===================================================================
# SCRIPT DE REPETIÇÃO - TESTE 7-SEGMENTOS
# Executa o 'teste_7_segs.sh' N vezes para análise estatística.
# ===================================================================

# --- CONFIGURAÇÃO ---
# Defina o número de repetições (N)
N_EXECUCOES=30

# O arquivo de log (deve ser o MESMO do script 'teste_7_segs.sh')
ARQUIVO_DE_SAIDA="resultados_7_segmentos.txt"
# --- FIM DA CONFIGURAÇÃO ---


echo "Iniciando teste de robustez do 7-Segmentos..."
echo "O script './teste_7_segs.sh' será executado $N_EXECUCOES vezes."
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
    echo "Executando (7-Seg) teste $i de $N_EXECUCOES..."

    # Chama o script de teste. 
    # O 'teste_7_segs.sh' já está configurado para ANEXAR (>>) ao log.
    ./teste_7_segs.sh
done

echo ""
echo "Teste de robustez do 7-Segmentos concluído."
echo "Resultados salvos em $ARQUIVO_DE_SAIDA"