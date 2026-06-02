import json
import os
import random
import spacy
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from fastcoref import FCoref
from nltk.corpus import wordnet as wn

# textos de entrada são em inglês, usa en_core_web_sm
# motivo: fastcoref (correferência) só funciona com modelos treinados em inglês.
# tentamos pt_core_news_sm + coreferee/fastcoref para português, mas:
#   - coreferee não suporta português e é incompatível com spaCy 3.8+
#   - fastcoref usa biu-nlp/f-coref treinado no OntoNotes (inglês)
#   - não existe biblioteca estável de correferência para português hoje

@lru_cache(maxsize=None)
def _is_person_role(word):
    # usa só o primeiro synset (sentido mais comum) pra evitar falsos positivos
    # ex: "stone" tem um synset de pessoa (jurista americano) mas não é o sentido primário
    syns = wn.synsets(word, pos=wn.NOUN)
    if not syns:
        return False
    return any(s.name() == 'person.n.01' for s in syns[0].hypernym_paths()[0])

# LIMITAÇÃO: abstrato vs concreto não é filtrado automaticamente.
# Substantivos como "cold", "hunger", "misery" aparecem junto com objetos físicos.
# Ideia pra resolver barato: usar um SLM (Phi-3-mini, Gemma-2-2b via Ollama, ou Claude Haiku)
# pra classificar cada substantivo como concreto ou abstrato.
# Prompt simples: "Is '{word}' a concrete physical object? Answer only yes or no."

N_SECOES = 10


def _is_pessoa(lemma, pessoas_ner):
    return lemma in pessoas_ner or _is_person_role(lemma)


def _find_cluster_head(cluster_spans, doc, pessoas_ner):
    # 1º passo: se qualquer span do cluster tem raiz em pessoas, ignora o cluster inteiro
    # (ex: "the poor little thing" está no cluster da menina junto com "girl" → ignora tudo)
    for start_char, end_char in cluster_spans:
        span = doc.char_span(start_char, end_char)
        if span is None:
            continue
        root = span.root
        if root.pos_ == "NOUN" and _is_pessoa(root.lemma_.lower(), pessoas_ner):
            return None
    # 2º passo: acha o primeiro span com raiz substantivo concreto
    for start_char, end_char in cluster_spans:
        span = doc.char_span(start_char, end_char)
        if span is None:
            continue
        root = span.root
        if root.pos_ == "NOUN" and not root.is_stop and not _is_pessoa(root.lemma_.lower(), pessoas_ner):
            return root.lemma_.lower()
    return None


def _contar_direto(doc, pessoas_ner, tamanho_secao):
    freq = Counter()
    secoes = defaultdict(set)
    for token in doc:
        if token.pos_ != "NOUN" or token.is_stop or token.is_punct:
            continue
        lemma = token.lemma_.lower()
        if _is_pessoa(lemma, pessoas_ner):
            continue
        secao = min(token.i // tamanho_secao, N_SECOES - 1)
        freq[lemma] += 1
        secoes[lemma].add(secao)
    return freq, secoes


def _aplicar_correferencia(doc, clusters_chars, pessoas_ner, tamanho_secao, freq, secoes):
    # para cada cluster, acha o substantivo-cabeça e soma as menções pronominais a ele
    resolucoes = []  # (pronome_texto, head), pra debug/teste
    for cluster in clusters_chars:
        head = _find_cluster_head(cluster, doc, pessoas_ner)
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


def _extrair_passagens(doc, candidatos, clusters_chars, pessoas_ner, tamanho_secao):
    sents = list(doc.sents)
    tok_to_sent = {tok.i: i for i, sent in enumerate(sents) for tok in sent}

    passagens = {}
    for obj in candidatos:
        # coleta (secao, idx_sentenca) de menções diretas e pronomes coreferentes
        mencoes = set()

        for tok in doc:
            if tok.pos_ == "NOUN" and tok.lemma_.lower() == obj:
                sent_i = tok_to_sent.get(tok.i)
                if sent_i is not None:
                    secao = min(tok.i // tamanho_secao, N_SECOES - 1)
                    mencoes.add((secao, sent_i))

        for cluster in clusters_chars:
            if _find_cluster_head(cluster, doc, pessoas_ner) != obj:
                continue
            for start_char, end_char in cluster:
                span = doc.char_span(start_char, end_char)
                if span is None:
                    continue
                if any(t.lemma_.lower() == obj and t.pos_ == "NOUN" for t in span):
                    continue
                if not any(t.pos_ == "PRON" for t in span):
                    continue
                sent_i = tok_to_sent.get(span[0].i)
                if sent_i is not None:
                    secao = min(span[0].i // tamanho_secao, N_SECOES - 1)
                    mencoes.add((secao, sent_i))

        # uma menção por seção, distribui pelo texto em vez de empilhar tudo
        by_secao = defaultdict(list)
        for secao, sent_i in mencoes:
            by_secao[secao].append(sent_i)

        trechos = []
        for secao in sorted(by_secao):
            sent_i = min(by_secao[secao])
            inicio = max(0, sent_i - 1)
            fim = min(len(sents) - 1, sent_i + 1)
            trecho = " ".join(s.text.strip() for s in sents[inicio:fim + 1])
            trechos.append(trecho)

        passagens[obj] = trechos
    return passagens


def _selecionar_candidatos(freq, n=3, pool_size=7):
    # sempre inclui o mais frequente + 2 aleatorios do top 5
    pool = freq.most_common(pool_size)
    if len(pool) <= n:
        return [w for w, _ in pool]
    top, restantes = pool[0][0], [w for w, _ in pool[1:]]
    aleatorios = random.SystemRandom().sample(restantes, n - 1)
    return [top] + aleatorios


def main(caminho_arquivo):
    nlp = spacy.load('en_core_web_sm')

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        texto = f.read()

    doc = nlp(texto)
    tamanho_secao = max(1, len(doc) // N_SECOES)

    pessoas_ner = {ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"}

    # contagem sem correferência (baseline)
    freq_sem, secoes_sem = _contar_direto(doc, pessoas_ner, tamanho_secao)

    # correferência
    print("loading coreference model...", flush=True)
    coref = FCoref()
    preds = coref.predict(texts=[texto])
    clusters_chars = preds[0].get_clusters(as_strings=False)
    clusters_str = preds[0].get_clusters(as_strings=True)

    nome_historia = os.path.splitext(os.path.basename(caminho_arquivo))[0]
    out_clusters = os.path.join('output', 'clusters')
    out_passagens = os.path.join('output', 'passagens')
    os.makedirs(out_clusters, exist_ok=True)
    os.makedirs(out_passagens, exist_ok=True)

    caminho_clusters = os.path.join(out_clusters, nome_historia + '_clusters.json')
    with open(caminho_clusters, 'w', encoding='utf-8') as f:
        json.dump(clusters_str, f, ensure_ascii=False, indent=2)
    print(f"clusters saved → {caminho_clusters}")

    # aplica correferência
    freq_com = Counter(freq_sem)
    secoes_com = defaultdict(set, {k: set(v) for k, v in secoes_sem.items()})
    freq_com, secoes_com, resolucoes = _aplicar_correferencia(
        doc, clusters_chars, pessoas_ner, tamanho_secao, freq_com, secoes_com
    )

    # mostra o que foi resolvido (útil pra verificar se tá certo)
    if resolucoes:
        print("\n--- coreference resolutions (pronouns → noun) ---")
        for pronome, head in resolucoes:
            print(f"  '{pronome}'  →  {head}")

    print("\n--- top 3 WITHOUT coreference ---")
    for obj, freq in freq_sem.most_common(3):
        print(f"  {freq}x, {obj}")

    candidatos = _selecionar_candidatos(freq_com)
    print("\n--- selected candidates (weighted random from top 7) ---")
    for obj in candidatos:
        dist = len(secoes_com[obj])
        print(f"  {freq_com[obj]}x in {dist}/{N_SECOES} sections, {obj}")

    print(f"\n{'object':<20} {'freq':>5}  {'dist':>7}  (sections/{N_SECOES})")
    print("-" * 50)
    for obj, freq in freq_com.most_common():
        dist = len(secoes_com[obj])
        print(f"{obj:<20} {freq:>5}  {dist:>4}/{N_SECOES}")

    # step 5: extrai passagens para cada candidato e salva
    passagens = _extrair_passagens(doc, candidatos, clusters_chars, pessoas_ner, tamanho_secao)
    caminho_passagens = os.path.join(out_passagens, nome_historia + '_passagens.json')
    with open(caminho_passagens, 'w', encoding='utf-8') as f:
        json.dump(passagens, f, ensure_ascii=False, indent=2)
    print(f"\npassagens saved → {caminho_passagens}")

    print("\n--- passages per candidate ---")
    for obj, trechos in passagens.items():
        print(f"\n  [{obj}], {len(trechos)} excerpt(s)")
        for i, t in enumerate(trechos, 1):
            print(f"    {i}. {t[:120]}{'...' if len(t) > 120 else ''}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 extrator_objetos.py <file.txt>")
        sys.exit(1)
    main(sys.argv[1])
