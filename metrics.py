"""
Moltbook Analysis — Focused Metrics

Three metrics for characterizing agent-agent interaction at scale:

1. Agent Behavioral Entropy
   - How much does an agent vary its output across different posts?
   - Low entropy / high self-similarity = template/spam agent
   - High entropy / low self-similarity = context-adaptive agent

2. Information Saturation
   - As more agents comment on a post, does each new comment add info?
   - Measures marginal information gain at each comment position
   - Tells us whether multi-agent "discussion" produces diminishing returns

3. Post-Comment Relevance
   - Is a comment specific to the post it appears under?
   - Or could it be pasted under any post (spam/generic)?
   - Measured as specificity: NCD(comment, actual_post) vs NCD(comment, random_posts)
"""

import numpy as np
import zlib
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization, strip punctuation."""
    if not text:
        return []
    return [
        w.strip(".,!?;:\"'()[]{}—–-…`~#@$%^&*+=/\\|<>")
        for w in text.lower().split()
        if w.strip(".,!?;:\"'()[]{}—–-…`~#@$%^&*+=/\\|<>")
    ]


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _zlen(text: str) -> int:
    return len(zlib.compress(text.encode('utf-8'), level=6))


def ncd(a: str, b: str) -> float:
    """
    Normalized Compression Distance.
    ≈ 0: texts share most information
    ≈ 1: texts are informationally independent
    """
    if not a or not b:
        return np.nan
    ca, cb = _zlen(a), _zlen(b)
    cab = _zlen(a + " " + b)
    denom = max(ca, cb)
    return (cab - min(ca, cb)) / denom if denom > 0 else 0.0


# ── 1. Agent Behavioral Entropy ──────────────────────────────────────

def agent_token_entropy(comments: list[str]) -> float:
    """
    Shannon entropy of an agent's pooled vocabulary across all comments.
    H(agent) = -Σ p(w) log2 p(w)

    Higher = more diverse vocabulary across contexts.
    Lower = repetitive/templated output.
    """
    all_tokens = []
    for c in comments:
        all_tokens.extend(tokenize(c))
    if not all_tokens:
        return 0.0
    counts = Counter(all_tokens)
    total = sum(counts.values())
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    return float(-np.sum(probs * np.log2(probs)))


def agent_self_ncd(comments: list[str], n_pairs: int = 50,
                   seed: int = 42) -> float:
    """
    Average NCD between random pairs of an agent's own comments
    (on different posts).

    Low self-NCD = agent produces similar text everywhere (template).
    High self-NCD = agent's output varies across contexts.
    """
    if len(comments) < 2:
        return np.nan
    rng = np.random.RandomState(seed)
    n = min(n_pairs, len(comments) * (len(comments) - 1) // 2)
    ncds = []
    seen = set()
    attempts = 0
    while len(ncds) < n and attempts < n * 5:
        i, j = rng.randint(0, len(comments), size=2)
        if i == j:
            attempts += 1
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        val = ncd(comments[i], comments[j])
        if not np.isnan(val):
            ncds.append(val)
        attempts += 1
    return float(np.mean(ncds)) if ncds else np.nan


def agent_bigram_entropy(comments: list[str]) -> float:
    """
    Shannon entropy of an agent's pooled bigram distribution.
    Captures phrase-level diversity (more sensitive to templates than unigrams).
    """
    all_bigrams = []
    for c in comments:
        all_bigrams.extend(_ngrams(tokenize(c), 2))
    if not all_bigrams:
        return 0.0
    counts = Counter(all_bigrams)
    total = sum(counts.values())
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    return float(-np.sum(probs * np.log2(probs)))


def agent_unique_bigram_ratio(comments: list[str]) -> float:
    """
    Fraction of unique bigrams out of total bigrams across all comments.
    Low = highly repetitive phrasing. High = diverse phrasing.
    """
    all_bigrams = []
    for c in comments:
        all_bigrams.extend(_ngrams(tokenize(c), 2))
    if not all_bigrams:
        return np.nan
    return len(set(all_bigrams)) / len(all_bigrams)


# ── 2. Information Saturation ────────────────────────────────────────

def incremental_info_gain_ngrams(existing_tokens: list[str],
                                  new_tokens: list[str],
                                  n: int = 1) -> float:
    """
    Fraction of n-grams in new_comment that are NOT in existing text.
    1.0 = entirely new content. 0.0 = complete repetition.
    """
    new_ng = _ngrams(new_tokens, n) if n > 1 else new_tokens
    if not new_ng:
        return np.nan
    existing_ng = set(_ngrams(existing_tokens, n) if n > 1 else existing_tokens)
    novel = sum(1 for ng in new_ng if ng not in existing_ng)
    return novel / len(new_ng)


def incremental_info_gain_compression(existing_text: str,
                                       new_comment: str) -> float:
    """
    Compression-based information gain.
    IG = C(existing + new) - C(existing)

    Measures how many additional compressed bytes the new comment adds.
    Normalized by C(new) to give a ratio:
      IG_norm = (C(existing + new) - C(existing)) / C(new)

    ≈ 1: new comment is entirely novel relative to existing
    ≈ 0: new comment is fully redundant
    """
    if not new_comment:
        return np.nan
    if not existing_text:
        return 1.0
    c_existing = _zlen(existing_text)
    c_new = _zlen(new_comment)
    c_combined = _zlen(existing_text + " " + new_comment)
    if c_new == 0:
        return np.nan
    return (c_combined - c_existing) / c_new


def compute_saturation_curve(comments_ordered: list[str]) -> dict:
    """
    For a post's comments ordered by time, compute info gain at each position.

    Returns dict with:
      'unigram_gains': list of unigram novelty at each position
      'bigram_gains': list of bigram novelty at each position
      'compression_gains': list of compression-based IG at each position
      'cumulative_unique_unigrams': list of cumulative unique token count
      'cumulative_unique_bigrams': list of cumulative unique bigram count
    """
    unigram_gains = []
    bigram_gains = []
    compression_gains = []
    cum_unique_uni = []
    cum_unique_bi = []

    existing_tokens = []
    existing_text = ""
    all_unigrams_seen = set()
    all_bigrams_seen = set()

    for comment in comments_ordered:
        tokens = tokenize(comment)

        # Incremental gains
        ug = incremental_info_gain_ngrams(existing_tokens, tokens, n=1)
        bg = incremental_info_gain_ngrams(existing_tokens, tokens, n=2)
        cg = incremental_info_gain_compression(existing_text, comment)

        unigram_gains.append(ug)
        bigram_gains.append(bg)
        compression_gains.append(cg)

        # Cumulative unique counts
        all_unigrams_seen.update(tokens)
        all_bigrams_seen.update(_ngrams(tokens, 2))
        cum_unique_uni.append(len(all_unigrams_seen))
        cum_unique_bi.append(len(all_bigrams_seen))

        # Extend existing
        existing_tokens.extend(tokens)
        existing_text = existing_text + " " + comment if existing_text else comment

    return {
        'unigram_gains': unigram_gains,
        'bigram_gains': bigram_gains,
        'compression_gains': compression_gains,
        'cumulative_unique_unigrams': cum_unique_uni,
        'cumulative_unique_bigrams': cum_unique_bi,
    }


# ── 3. Post-Comment Relevance ───────────────────────────────────────

def post_comment_relevance(post_text: str, comment_text: str) -> float:
    """
    NCD between a post and a comment.
    Lower = more shared information = more relevant.
    Higher = more independent = less relevant / generic.
    """
    return ncd(post_text, comment_text)


def specificity_score(comment: str, actual_post: str,
                      random_posts: list[str]) -> float:
    """
    How much more relevant is a comment to its actual post vs. random posts?

    S = mean(NCD(comment, random_post)) - NCD(comment, actual_post)

    S > 0: comment is more similar to its post than to random posts (relevant)
    S ≈ 0: comment is equally similar to any post (generic/spam)
    S < 0: comment is less similar to its post than random (misplaced)

    NOTE: NCD is unreliable for short texts (< ~50 words) due to compression
    overhead dominating the signal.  Prefer specificity_score_lexical for
    typical social-media comment lengths.
    """
    ncd_actual = ncd(comment, actual_post)
    if np.isnan(ncd_actual):
        return np.nan
    ncd_randoms = [ncd(comment, rp) for rp in random_posts if rp]
    ncd_randoms = [v for v in ncd_randoms if not np.isnan(v)]
    if not ncd_randoms:
        return np.nan
    return float(np.mean(ncd_randoms) - ncd_actual)


# ── 4. Lexical Post-Comment Relevance ──────────────────────────────────

STOPWORDS = frozenset({
    # articles
    'a', 'an', 'the',
    # pronouns
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours',
    'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    # be / have / do
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    # modals
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might',
    'must',
    # prepositions
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'up', 'down',
    # conjunctions
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    # determiners / quantifiers
    'this', 'that', 'these', 'those', 'all', 'each', 'every', 'any',
    'some', 'no', 'not', 'few', 'more', 'most', 'other', 'such', 'only',
    'own', 'same',
    # common adverbs
    'very', 'just', 'also', 'too', 'than', 'then', 'now', 'here', 'there',
    'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom',
    'if', 'about', 'again', 'further', 'once',
})


def content_tokens(text: str) -> list[str]:
    """Tokenize and remove stopwords, keeping only content words."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Jaccard similarity between two token sets.  0 = no overlap, 1 = identical."""
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return np.nan
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def specificity_score_lexical(comment: str, actual_post: str,
                              random_posts: list[str]) -> float:
    """
    Lexical specificity: content-word Jaccard overlap with actual post
    vs. average overlap with random posts.

    S = Jaccard(comment, actual_post) - mean(Jaccard(comment, random_posts))

    Uses content words only (stopwords removed) so that shared function
    words do not inflate similarity for unrelated texts.

    S > 0: comment shares more content vocabulary with its post than random
    S ≈ 0: comment is equally (ir)relevant to any post  (generic)
    S < 0: comment is less relevant to its post than to random posts
    """
    c_tok = content_tokens(comment)
    a_tok = content_tokens(actual_post)
    sim_actual = jaccard(c_tok, a_tok)
    if np.isnan(sim_actual):
        return np.nan
    sims_random = []
    for rp in random_posts:
        if rp:
            s = jaccard(c_tok, content_tokens(rp))
            if not np.isnan(s):
                sims_random.append(s)
    if not sims_random:
        return np.nan
    return float(sim_actual - np.mean(sims_random))
