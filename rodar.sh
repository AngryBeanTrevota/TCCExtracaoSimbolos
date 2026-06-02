#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "uso: ./rodar.sh fosforos.txt"
    exit 1
fi

TEXTO="textosBase/$1"
NOME="${1%.txt}"

mkdir -p output/clusters output/passagens output/prompts output/interpretacoes

source venv/bin/activate

echo "==> extraindo objetos e passagens..."
python3 extrator_objetos.py "$TEXTO"

mv "textosBase/${NOME}_clusters.json"  output/clusters/
mv "textosBase/${NOME}_passagens.json" output/passagens/

echo ""
echo "==> gerando prompts..."
python3 interpretar.py "output/passagens/${NOME}_passagens.json" output/prompts

echo ""
echo "arquivos salvos:"
echo "  output/clusters/${NOME}_clusters.json"
echo "  output/passagens/${NOME}_passagens.json"
echo "  output/prompts/${NOME}_prompts.txt"
echo "  (output/interpretacoes/, salvar respostas dos modelos aqui)"
