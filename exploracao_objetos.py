"""
Extração de diferentes tipos de entidades usando NER

Este script mostra como extrair objetos (PRODUCT) e outras entidades
além de pessoas.
"""

import spacy

def main():
    # Carregar modelo NER do spaCy
    nlp = spacy.load('en_core_web_sm')

    # Texto de exemplo com vários tipos de entidades
    texto_exemplo = """
    Anne Shirley loved her iPhone and always carried it in her brown leather bag.
    She drove a Toyota Corolla to Green Gables.
    Matthew gave her a beautiful necklace and a copy of Hamlet.
    They lived in Prince Edward Island, Canada.
    """

    print("=== EXTRAÇÃO DE DIFERENTES ENTIDADES ===\n")
    print("Texto original:")
    print(texto_exemplo)
    print("\n" + "="*50 + "\n")

    # Processar com NER
    doc = nlp(texto_exemplo)

    # Separar por tipo de entidade
    pessoas = []
    objetos = []
    locais = []
    organizacoes = []

    print("Entidades encontradas:\n")
    for ent in doc.ents:
        print(f"  '{ent.text}' -> {ent.label_}")

        if ent.label_ == "PERSON":
            pessoas.append(ent.text)
        elif ent.label_ == "PRODUCT":
            objetos.append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            locais.append(ent.text)
        elif ent.label_ == "ORG":
            organizacoes.append(ent.text)

    print("\n" + "="*50 + "\n")
    print("RESUMO POR CATEGORIA:")
    print(f"\nPessoas (PERSON): {pessoas}")
    print(f"Objetos (PRODUCT): {objetos}")
    print(f"Locais (GPE/LOC): {locais}")
    print(f"Organizações (ORG): {organizacoes}")

    # Se você quiser APENAS objetos, descomente abaixo:
    print("\n" + "="*50 + "\n")
    print("APENAS OBJETOS:")
    doc2 = nlp(texto_exemplo)
    for ent in doc2.ents:
        if ent.label_ == "PRODUCT":
            print(f"  - '{ent.text}'")

if __name__ == "__main__":
    main()
