import json
import sys
from pathlib import Path

PROMPT_TEMPLATE = Path("prompt.txt").read_text(encoding="utf-8")

MODELOS_WEB = [
    "ChatGPT (chat.openai.com)",
    "Claude (claude.ai)",
    "Gemini (gemini.google.com)",
]


def _montar_prompt(objetos):
    lista = ", ".join(objetos)
    formato = "\n\n".join(
        f"{obj}:\na. [interpretation]\nb. [interpretation]\nc. [interpretation]"
        for obj in objetos
    )
    return PROMPT_TEMPLATE.format(objects=lista, format=formato)


def gerar_prompts(caminho_passagens, dir_prompts):
    caminho_passagens = Path(caminho_passagens)
    nome = caminho_passagens.stem.replace("_passagens", "")

    with open(caminho_passagens, encoding="utf-8") as f:
        passagens = json.load(f)

    saida_txt = Path(dir_prompts) / f"{nome}_prompts.txt"
    objetos = list(passagens.keys())
    prompt = _montar_prompt(objetos)

    print(f"\n{'='*60}")
    print(f"CONTO: {nome}  |  objetos: {', '.join(objetos)}")
    print(f"{'='*60}")
    print(f"\nAnexe: output/passagens/{nome}_passagens.json")
    print(f"Cole nos modelos: {', '.join(MODELOS_WEB)}\n")
    print(prompt)

    saida_txt.write_text(
        f"Anexar: output/passagens/{nome}_passagens.json\n\n{prompt}",
        encoding="utf-8"
    )
    print(f"\nPrompt salvo → {saida_txt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python3 interpretar.py <arquivo_passagens.json>")
        sys.exit(1)
    gerar_prompts(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
