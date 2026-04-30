import json
import spacy
import sys
from collections import Counter, defaultdict
from fastcoref import FCoref

# textos de entrada são em inglês, usa en_core_web_sm
# motivo: fastcoref (correferência) só funciona com modelos treinados em inglês.
# tentamos pt_core_news_sm + coreferee/fastcoref para português, mas:
#   - coreferee não suporta português e é incompatível com spaCy 3.8+
#   - fastcoref usa biu-nlp/f-coref treinado no OntoNotes (inglês)
#   - não existe biblioteca estável de correferência para português hoje

PERSON_ROLES = {
    'girl', 'boy', 'child', 'man', 'woman', 'lady', 'gentleman',
    'father', 'mother', 'son', 'daughter', 'grandmother', 'grandfather',
    'brother', 'sister', 'uncle', 'aunt', 'cousin', 'nephew', 'niece',
    'friend', 'neighbor', 'stranger', 'servant', 'master', 'mistress',
    'king', 'queen', 'prince', 'princess', 'lord', 'sir', 'madam',
    'merchant', 'soldier', 'doctor', 'priest', 'witch', 'orphan',
    'seller', 'granddaughter', 'passerby',
}

# LIMITAÇÃO: abstrato vs concreto não é filtrado automaticamente.
# Substantivos como "cold", "hunger", "misery" aparecem junto com objetos físicos.
# Ideia pra resolver barato: usar um SLM (Phi-3-mini, Gemma-2-2b via Ollama, ou Claude Haiku)
# pra classificar cada substantivo como concreto ou abstrato.
# Prompt simples: "Is '{word}' a concrete physical object? Answer only yes or no."

N_SECOES = 10


def _find_cluster_head(cluster_spans, doc, pessoas):
    # 1º passo: se qualquer span do cluster tem raiz em pessoas, ignora o cluster inteiro
    # (ex: "the poor little thing" está no cluster da menina junto com "girl" → ignora tudo)
    for start_char, end_char in cluster_spans:
        span = doc.char_span(start_char, end_char)
        if span is None:
            continue
        root = span.root
        if root.pos_ == "NOUN" and root.lemma_.lower() in pessoas:
            return None
    # 2º passo: acha o primeiro span com raiz substantivo concreto
    for start_char, end_char in cluster_spans:
        span = doc.char_span(start_char, end_char)
        if span is None:
            continue
        root = span.root
        if root.pos_ == "NOUN" and not root.is_stop and root.lemma_.lower() not in pessoas:
            return root.lemma_.lower()
    return None


def _contar_direto(doc, pessoas, tamanho_secao):
    freq = Counter()
    secoes = defaultdict(set)
    for token in doc:
        if token.pos_ != "NOUN" or token.is_stop or token.is_punct:
            continue
        lemma = token.lemma_.lower()
        if lemma in pessoas:
            continue
        secao = min(token.i // tamanho_secao, N_SECOES - 1)
        freq[lemma] += 1
        secoes[lemma].add(secao)
    return freq, secoes


def _aplicar_correferencia(doc, clusters_chars, pessoas, tamanho_secao, freq, secoes):
    # para cada cluster, acha o substantivo-cabeça e soma as menções pronominais a ele
    resolucoes = []  # (pronome_texto, head), pra debug/teste
    for cluster in clusters_chars:
        head = _find_cluster_head(cluster, doc, pessoas)
        if head is None:
            continue
        for start_char, end_char in cluster:
            span = doc.char_span(start_char, end_char)
            if span is None:
                continue
            # pula spans que já contêm o próprio substantivo-cabeça (já foram contados)
            if any(t.lemma_.lower() == head and t.pos_ == "NOUN" for t in span):
                continue
            # só conta pronomes como menções coreferentes
            if not any(t.pos_ == "PRON" for t in span):
                continue
            secao = min(span[0].i // tamanho_secao, N_SECOES - 1)
            freq[head] += 1
            secoes[head].add(secao)
            resolucoes.append((span.text.strip(), head))
    return freq, secoes, resolucoes


def main(caminho_arquivo):
    nlp = spacy.load('en_core_web_sm')

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        texto = f.read()

    doc = nlp(texto)
    tamanho_secao = max(1, len(doc) // N_SECOES)

    pessoas = {ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"}
    pessoas |= PERSON_ROLES

    # contagem sem correferência (baseline)
    freq_sem, secoes_sem = _contar_direto(doc, pessoas, tamanho_secao)

    # correferência
    print("loading coreference model...", flush=True)
    coref = FCoref()
    preds = coref.predict(texts=[texto])
    clusters_chars = preds[0].get_clusters(as_strings=False)
    clusters_str = preds[0].get_clusters(as_strings=True)

    # salva clusters em JSON ao lado do arquivo de entrada
    nome_base = caminho_arquivo.rsplit('.', 1)[0]
    caminho_clusters = nome_base + '_clusters.json'
    with open(caminho_clusters, 'w', encoding='utf-8') as f:
        json.dump(clusters_str, f, ensure_ascii=False, indent=2)
    print(f"clusters saved → {caminho_clusters}")

    # aplica correferência
    freq_com = Counter(freq_sem)
    secoes_com = defaultdict(set, {k: set(v) for k, v in secoes_sem.items()})
    freq_com, secoes_com, resolucoes = _aplicar_correferencia(
        doc, clusters_chars, pessoas, tamanho_secao, freq_com, secoes_com
    )

    # mostra o que foi resolvido (útil pra verificar se tá certo)
    if resolucoes:
        print("\n--- coreference resolutions (pronouns → noun) ---")
        for pronome, head in resolucoes:
            print(f"  '{pronome}'  →  {head}")

    print("\n--- top 3 WITHOUT coreference ---")
    for obj, freq in freq_sem.most_common(3):
        print(f"  {freq}x, {obj}")

    print("\n--- top 3 WITH coreference ---")
    for obj, freq in freq_com.most_common(3):
        dist = len(secoes_com[obj])
        print(f"  {freq}x in {dist}/{N_SECOES} sections, {obj}")

    print(f"\n{'object':<20} {'freq':>5}  {'dist':>7}  (sections/{N_SECOES})")
    print("-" * 50)
    for obj, freq in freq_com.most_common():
        dist = len(secoes_com[obj])
        print(f"{obj:<20} {freq:>5}  {dist:>4}/{N_SECOES}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 extrator_objetos.py <file.txt>")
        sys.exit(1)
    main(sys.argv[1])
