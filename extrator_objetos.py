import spacy
import sys

def main(caminho_arquivo):
    nlp = spacy.load('pt_core_news_sm')

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        texto = f.read()

    doc = nlp(texto)

    pessoas = {ent.text.lower() for ent in doc.ents if ent.label_ == "PER"}

    objetos = []
    for token in doc:
        if token.dep_ == "obj" and token.text.lower() not in pessoas:
            objetos.append(token.text)

    objetos = sorted(set(objetos))

    print("Objetos encontrados:")
    for obj in objetos:
        print(f"  - {obj}")
    print(f"\nTotal: {len(objetos)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 exploracao_simples.py <arquivo.txt>")
        sys.exit(1)
    main(sys.argv[1])
