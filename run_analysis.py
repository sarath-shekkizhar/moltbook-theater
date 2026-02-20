"""
Run six analyses on the combined Moltbook dataset.

1. Agent Behavioral Entropy — per-agent diversity
2. Information Saturation — marginal info gain per comment position
3. Post-Comment Relevance — specificity of comments to their posts
4. Semantic Relevance — embedding-based specificity (validates Jaccard)
5. LLM-as-Judge Validation — ground-truth quality assessment
6. Nested Reply Analysis — top-level vs nested engagement comparison

Outputs:
  cache/agent_entropy.parquet
  cache/saturation_curves.parquet
  cache/relevance_scores.parquet
  cache/semantic_relevance.parquet
  cache/judge_results.parquet
  cache/nested_reply_analysis.parquet
  paper-tex/figures/*.pdf
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from collections import Counter

from metrics import (
    agent_token_entropy, agent_self_ncd, agent_bigram_entropy,
    agent_unique_bigram_ratio, compute_saturation_curve,
    post_comment_relevance, specificity_score, specificity_score_lexical,
    tokenize, ncd, content_tokens, jaccard,
)
from embedding_metrics import (
    embed_texts_batch, cosine_similarity, cosine_similarity_batch,
    semantic_specificity, semantic_saturation_curve,
)
from llm_judge import (
    judge_pair, run_judge_batch, compute_inter_rater_reliability,
)

CACHE = "cache"
FIG_DIR = "paper-tex/figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PAL = sns.color_palette("deep")

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_combined():
    posts = pd.read_parquet(f"{CACHE}/combined_posts.parquet")
    comments = pd.read_parquet(f"{CACHE}/combined_comments.parquet")
    posts['full_text'] = posts['title'].fillna('') + '\n' + posts['body'].fillna('')
    return posts, comments


# ═══════════════════════════════════════════════════════════════════════
# 1. AGENT BEHAVIORAL ENTROPY
# ═══════════════════════════════════════════════════════════════════════

def analyze_agent_entropy(comments, min_comments=10, max_agents=5000):
    """Compute entropy metrics for agents with enough comments."""
    print("\n" + "=" * 70)
    print("1. AGENT BEHAVIORAL ENTROPY")
    print("=" * 70)

    agent_counts = comments.groupby('author_id').size()
    eligible = agent_counts[agent_counts >= min_comments]
    print(f"  Agents with >= {min_comments} comments: {len(eligible):,} "
          f"(of {agent_counts.shape[0]:,} total)")

    if len(eligible) > max_agents:
        eligible = eligible.sample(max_agents, random_state=42)
        print(f"  Sampled {max_agents} agents for analysis")

    agent_comments = comments[comments['author_id'].isin(eligible.index)]
    grouped = agent_comments.groupby('author_id')['content'].apply(list)

    results = []
    t0 = time.time()
    for i, (agent_id, texts) in enumerate(grouped.items()):
        texts = [str(t) for t in texts if t and str(t).strip()]
        if len(texts) < min_comments:
            continue

        row = {
            'agent_id': agent_id,
            'n_comments': len(texts),
            'token_entropy': agent_token_entropy(texts),
            'bigram_entropy': agent_bigram_entropy(texts),
            'unique_bigram_ratio': agent_unique_bigram_ratio(texts),
            'self_ncd': agent_self_ncd(texts, n_pairs=30),
            'mean_comment_len': np.mean([len(tokenize(t)) for t in texts]),
        }
        results.append(row)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(grouped)} agents ({elapsed:.1f}s)")

    df = pd.DataFrame(results)
    elapsed = time.time() - t0
    print(f"  Computed entropy for {len(df):,} agents in {elapsed:.1f}s")

    print(f"\n  Token entropy:       mean={df['token_entropy'].mean():.2f}  "
          f"median={df['token_entropy'].median():.2f}  "
          f"std={df['token_entropy'].std():.2f}")
    print(f"  Bigram entropy:      mean={df['bigram_entropy'].mean():.2f}  "
          f"median={df['bigram_entropy'].median():.2f}")
    print(f"  Unique bigram ratio: mean={df['unique_bigram_ratio'].mean():.3f}  "
          f"median={df['unique_bigram_ratio'].median():.3f}")
    print(f"  Self-NCD:            mean={df['self_ncd'].mean():.3f}  "
          f"median={df['self_ncd'].median():.3f}")

    low_entropy = (df['self_ncd'] < 0.5).sum()
    mid_entropy = ((df['self_ncd'] >= 0.5) & (df['self_ncd'] < 0.8)).sum()
    high_entropy = (df['self_ncd'] >= 0.8).sum()
    print(f"\n  Agent classification by self-NCD:")
    print(f"    Template/spam (self-NCD < 0.5): {low_entropy} "
          f"({low_entropy/len(df)*100:.1f}%)")
    print(f"    Moderate variation (0.5-0.8):   {mid_entropy} "
          f"({mid_entropy/len(df)*100:.1f}%)")
    print(f"    Context-adaptive (>= 0.8):      {high_entropy} "
          f"({high_entropy/len(df)*100:.1f}%)")

    df.to_parquet(f"{CACHE}/agent_entropy.parquet", index=False)
    return df


def plot_agent_entropy(df):
    """Generate figures for agent behavioral entropy."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Self-NCD distribution
    ax = axes[0]
    sns.histplot(df['self_ncd'].dropna(), bins=50, ax=ax, color=PAL[0],
                 edgecolor='white')
    ax.axvline(df['self_ncd'].median(), color='crimson', linestyle='--',
               label=f'median = {df["self_ncd"].median():.3f}')
    ax.set_xlabel('Self-NCD')
    ax.set_ylabel('Number of Agents')
    ax.set_title('(a) Agent Self-Similarity')
    ax.legend(fontsize=9)

    # (b) Token entropy distribution
    ax = axes[1]
    sns.histplot(df['token_entropy'].dropna(), bins=50, ax=ax, color=PAL[1],
                 edgecolor='white')
    ax.axvline(df['token_entropy'].median(), color='crimson', linestyle='--',
               label=f'median = {df["token_entropy"].median():.2f}')
    ax.set_xlabel('Token Entropy (bits)')
    ax.set_ylabel('Number of Agents')
    ax.set_title('(b) Vocabulary Diversity')
    ax.legend(fontsize=9)

    # (c) Scatter
    ax = axes[2]
    sns.scatterplot(x='self_ncd', y='token_entropy', data=df, ax=ax,
                    alpha=0.25, s=10, color=PAL[0], linewidth=0)
    ax.set_xlabel('Self-NCD')
    ax.set_ylabel('Token Entropy (bits)')
    ax.set_title('(c) Self-NCD vs. Vocabulary Diversity')

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/agent_entropy.pdf")
    plt.close()
    print(f"  Saved {FIG_DIR}/agent_entropy.pdf")


# ═══════════════════════════════════════════════════════════════════════
# 2. INFORMATION SATURATION
# ═══════════════════════════════════════════════════════════════════════

def analyze_saturation(posts, comments, min_comments=5, max_position=30,
                       max_posts=20000):
    """Compute information saturation curves across posts."""
    print("\n" + "=" * 70)
    print("2. INFORMATION SATURATION CURVE")
    print("=" * 70)

    cpp = comments.groupby('post_id').size()
    eligible_posts = cpp[cpp >= min_comments].index
    print(f"  Posts with >= {min_comments} comments: {len(eligible_posts):,}")

    if len(eligible_posts) > max_posts:
        rng = np.random.RandomState(42)
        eligible_posts = rng.choice(eligible_posts, max_posts, replace=False)
        print(f"  Sampled {max_posts} posts")

    subset = comments[comments['post_id'].isin(eligible_posts)].copy()
    subset = subset.sort_values(['post_id', 'created_at'])

    all_uni_gains = {k: [] for k in range(max_position)}
    all_bi_gains = {k: [] for k in range(max_position)}
    all_comp_gains = {k: [] for k in range(max_position)}
    all_cum_uni = {k: [] for k in range(max_position)}
    all_cum_bi = {k: [] for k in range(max_position)}

    t0 = time.time()
    n_processed = 0
    for post_id, group in subset.groupby('post_id'):
        texts = [str(c) for c in group['content'].values
                 if c and str(c).strip()]
        if len(texts) < min_comments:
            continue

        texts = texts[:max_position]
        curve = compute_saturation_curve(texts)

        for k in range(len(texts)):
            all_uni_gains[k].append(curve['unigram_gains'][k])
            all_bi_gains[k].append(curve['bigram_gains'][k])
            all_comp_gains[k].append(curve['compression_gains'][k])
            all_cum_uni[k].append(curve['cumulative_unique_unigrams'][k])
            all_cum_bi[k].append(curve['cumulative_unique_bigrams'][k])

        n_processed += 1
        if n_processed % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {n_processed:,} posts ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Processed {n_processed:,} posts in {elapsed:.1f}s")

    rows = []
    for k in range(max_position):
        if not all_uni_gains[k]:
            break
        rows.append({
            'position': k,
            'n_posts': len(all_uni_gains[k]),
            'unigram_gain_mean': np.nanmean(all_uni_gains[k]),
            'unigram_gain_median': np.nanmedian(all_uni_gains[k]),
            'unigram_gain_std': np.nanstd(all_uni_gains[k]),
            'bigram_gain_mean': np.nanmean(all_bi_gains[k]),
            'bigram_gain_median': np.nanmedian(all_bi_gains[k]),
            'compression_gain_mean': np.nanmean(all_comp_gains[k]),
            'compression_gain_median': np.nanmedian(all_comp_gains[k]),
            'compression_gain_std': np.nanstd(all_comp_gains[k]),
            'cum_unique_uni_mean': np.nanmean(all_cum_uni[k]),
            'cum_unique_bi_mean': np.nanmean(all_cum_bi[k]),
        })
    df = pd.DataFrame(rows)

    print(f"\n  Saturation curve (mean info gain at each position):")
    print(f"  {'Pos':>4s} {'n_posts':>8s} {'uni_gain':>10s} "
          f"{'bi_gain':>10s} {'comp_gain':>10s}")
    for _, r in df.iterrows():
        print(f"  {int(r['position']):4d} {int(r['n_posts']):8d} "
              f"{r['unigram_gain_mean']:10.3f} "
              f"{r['bigram_gain_mean']:10.3f} "
              f"{r['compression_gain_mean']:10.3f}")

    df.to_parquet(f"{CACHE}/saturation_curves.parquet", index=False)
    return df


def plot_saturation(df):
    """Generate information saturation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    df = df[df['n_posts'] >= 100].copy()

    # (a) Lexical gain curves
    ax = axes[0]
    ax.plot(df['position'], df['unigram_gain_mean'], 'o-', color=PAL[0],
            markersize=4, label='Unigram')
    ax.fill_between(df['position'],
                    df['unigram_gain_mean'] - df['unigram_gain_std'],
                    df['unigram_gain_mean'] + df['unigram_gain_std'],
                    alpha=0.15, color=PAL[0])
    ax.plot(df['position'], df['bigram_gain_mean'], 's-', color=PAL[1],
            markersize=4, label='Bigram')
    ax.set_xlabel('Comment Position')
    ax.set_ylabel('Fraction Novel N-grams')
    ax.set_title('(a) Lexical Information Gain')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # (b) Compression gain
    ax = axes[1]
    ax.plot(df['position'], df['compression_gain_mean'], 'o-',
            color=PAL[2], markersize=4)
    ax.fill_between(df['position'],
                    df['compression_gain_mean'] - df['compression_gain_std'],
                    df['compression_gain_mean'] + df['compression_gain_std'],
                    alpha=0.15, color=PAL[2])
    ax.set_xlabel('Comment Position')
    ax.set_ylabel('Normalized Compression IG')
    ax.set_title('(b) Compression Information Gain')
    ax.set_ylim(0, 1.3)

    # (c) Cumulative vocabulary
    ax = axes[2]
    ax.plot(df['position'], df['cum_unique_uni_mean'], 'o-',
            color=PAL[0], markersize=4, label='Unigrams')
    ax.plot(df['position'], df['cum_unique_bi_mean'], 's-',
            color=PAL[1], markersize=4, label='Bigrams')
    ax.set_xlabel('Comment Position')
    ax.set_ylabel('Cumulative Unique Count')
    ax.set_title('(c) Cumulative Vocabulary Growth')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/saturation_curve.pdf")
    plt.close()
    print(f"  Saved {FIG_DIR}/saturation_curve.pdf")


# ═══════════════════════════════════════════════════════════════════════
# 3. POST-COMMENT RELEVANCE
# ═══════════════════════════════════════════════════════════════════════

def analyze_relevance(posts, comments, n_sample=50000, n_random=10):
    """Compute relevance and specificity scores for post-comment pairs.

    Reports both Jaccard-based (lexical) and NCD-based specificity.
    Jaccard on content words is the primary metric; NCD is kept for
    comparison but is unreliable for short texts.
    """
    print("\n" + "=" * 70)
    print("3. POST-COMMENT RELEVANCE")
    print("=" * 70)

    posts_dedup = posts.drop_duplicates('id', keep='first')
    post_texts = posts_dedup.set_index('id')['full_text']

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(comments), min(n_sample, len(comments)),
                            replace=False)
    sample = comments.iloc[sample_idx].copy()

    sample['post_text'] = sample['post_id'].map(post_texts)
    sample = sample.dropna(subset=['post_text', 'content'])
    print(f"  Sampled {len(sample):,} (post, comment) pairs")

    # Pool of random posts for specificity baseline
    all_post_texts = post_texts.dropna().values
    random_pool_idx = rng.choice(len(all_post_texts),
                                  size=min(10000, len(all_post_texts)),
                                  replace=False)
    random_pool = all_post_texts[random_pool_idx]

    results = []
    t0 = time.time()
    for i, (_, row) in enumerate(sample.iterrows()):
        comment = str(row['content'])
        post = str(row['post_text'])

        rand_posts = list(rng.choice(random_pool, n_random, replace=False))

        # Jaccard-based (primary)
        c_tok = content_tokens(comment)
        p_tok = content_tokens(post)
        jacc_actual = jaccard(c_tok, p_tok)
        jacc_randoms = [jaccard(c_tok, content_tokens(str(rp)))
                        for rp in rand_posts]
        jacc_random_mean = np.nanmean(jacc_randoms)
        spec_lexical = specificity_score_lexical(comment, post, rand_posts)

        # NCD-based (secondary / comparison)
        ncd_actual = ncd(comment, post)
        ncd_randoms = [ncd(comment, str(rp)) for rp in rand_posts]
        ncd_random_mean = np.nanmean(ncd_randoms)
        spec_ncd = specificity_score(comment, post, rand_posts)

        results.append({
            'jaccard_actual': jacc_actual,
            'jaccard_random_mean': jacc_random_mean,
            'specificity_lexical': spec_lexical,
            'ncd_actual_post': ncd_actual,
            'ncd_random_post_mean': ncd_random_mean,
            'specificity_ncd': spec_ncd,
            'comment_len': len(tokenize(comment)),
            'content_word_count': len(c_tok),
            'post_len': len(tokenize(post)),
        })

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1:,}/{len(sample):,} ({rate:.0f} pairs/sec)")

    df = pd.DataFrame(results)
    elapsed = time.time() - t0
    print(f"  Computed relevance for {len(df):,} pairs in {elapsed:.1f}s")

    # --- Lexical specificity summary ---
    print(f"\n  LEXICAL SPECIFICITY (Jaccard, content words):")
    print(f"  Jaccard to actual post:  mean={df['jaccard_actual'].mean():.4f}  "
          f"median={df['jaccard_actual'].median():.4f}")
    print(f"  Jaccard to random posts: mean={df['jaccard_random_mean'].mean():.4f}  "
          f"median={df['jaccard_random_mean'].median():.4f}")
    print(f"  Specificity (lexical):   mean={df['specificity_lexical'].mean():.4f}  "
          f"median={df['specificity_lexical'].median():.4f}  "
          f"std={df['specificity_lexical'].std():.4f}")

    relevant = (df['specificity_lexical'] > 0.02).sum()
    generic = ((df['specificity_lexical'] >= -0.005) &
               (df['specificity_lexical'] <= 0.02)).sum()
    irrelevant = (df['specificity_lexical'] < -0.005).sum()
    print(f"\n  Classification (lexical):")
    print(f"    Specific to post (spec > 0.02):  {relevant:,} "
          f"({relevant/len(df)*100:.1f}%)")
    print(f"    Generic (spec ~ 0):              {generic:,} "
          f"({generic/len(df)*100:.1f}%)")
    print(f"    Off-topic (spec < -0.005):       {irrelevant:,} "
          f"({irrelevant/len(df)*100:.1f}%)")

    # --- NCD specificity summary (comparison) ---
    print(f"\n  NCD SPECIFICITY (for comparison, unreliable for short texts):")
    print(f"  NCD to actual post:  mean={df['ncd_actual_post'].mean():.4f}  "
          f"median={df['ncd_actual_post'].median():.4f}")
    print(f"  NCD to random posts: mean={df['ncd_random_post_mean'].mean():.4f}  "
          f"median={df['ncd_random_post_mean'].median():.4f}")
    print(f"  Specificity (NCD):   mean={df['specificity_ncd'].mean():.4f}  "
          f"median={df['specificity_ncd'].median():.4f}")

    # --- Breakdown by comment length ---
    print(f"\n  Lexical specificity by comment length:")
    for lo, hi in [(1, 5), (5, 10), (10, 25), (25, 50), (50, 100),
                   (100, 200), (200, 500)]:
        mask = (df['comment_len'] >= lo) & (df['comment_len'] < hi)
        sub = df.loc[mask, 'specificity_lexical']
        if len(sub) > 0:
            print(f"    [{lo:3d},{hi:3d}) words: n={len(sub):5d}  "
                  f"mean={sub.mean():.4f}  median={sub.median():.4f}")

    df.to_parquet(f"{CACHE}/relevance_scores.parquet", index=False)
    return df


def plot_relevance(df):
    """Generate relevance figures using Jaccard-based specificity."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Jaccard distributions: actual vs random
    ax = axes[0]
    sns.kdeplot(df['jaccard_actual'].dropna(), ax=ax, color=PAL[0],
                label='Actual post', fill=True, alpha=0.3)
    sns.kdeplot(df['jaccard_random_mean'].dropna(), ax=ax, color=PAL[1],
                label='Random posts', fill=True, alpha=0.3)
    ax.set_xlabel('Content-Word Jaccard Similarity')
    ax.set_ylabel('Density')
    ax.set_title('(a) Comment–Post Similarity')
    ax.legend(fontsize=9)

    # (b) Specificity distribution
    ax = axes[1]
    spec = df['specificity_lexical'].dropna()
    sns.histplot(spec, bins=80, ax=ax, color=PAL[2], edgecolor='white',
                 stat='density')
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.axvline(spec.median(), color='crimson', linestyle='--',
               label=f'median = {spec.median():.4f}')
    ax.set_xlabel('Lexical Specificity')
    ax.set_ylabel('Density')
    ax.set_title('(b) Post-Comment Specificity')
    ax.legend(fontsize=9)

    # (c) Specificity by comment length
    ax = axes[2]
    df_clean = df.dropna(subset=['specificity_lexical', 'comment_len'])
    df_clean = df_clean[df_clean['comment_len'] > 0].copy()
    bins = [0, 10, 25, 50, 100, 200, 500, 10000]
    labels = ['1-10', '11-25', '26-50', '51-100', '101-200', '201-500',
              '500+']
    df_clean['len_bin'] = pd.cut(df_clean['comment_len'], bins=bins,
                                  labels=labels)
    sns.barplot(x='len_bin', y='specificity_lexical', data=df_clean,
                ax=ax, color=PAL[0], errorbar='sd', capsize=0.15)
    ax.set_xlabel('Comment Length (tokens)')
    ax.set_ylabel('Mean Lexical Specificity')
    ax.set_title('(c) Specificity by Comment Length')
    ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/relevance.pdf")
    plt.close()
    print(f"  Saved {FIG_DIR}/relevance.pdf")


# ═══════════════════════════════════════════════════════════════════════
# 4. SEMANTIC RELEVANCE (Embedding-based)
# ═══════════════════════════════════════════════════════════════════════

def analyze_semantic_relevance(posts, comments, n_sample=50000, n_random=10):
    """Compute embedding-based semantic specificity for post-comment pairs.

    Uses the same sample as analyze_relevance for direct comparison.
    Embeds comments and posts with text-embedding-3-small, then computes
    cosine-similarity-based specificity.
    """
    print("\n" + "=" * 70)
    print("4. SEMANTIC RELEVANCE (Embedding-based)")
    print("=" * 70)

    posts_dedup = posts.drop_duplicates('id', keep='first')
    post_texts = posts_dedup.set_index('id')['full_text']

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(comments), min(n_sample, len(comments)),
                            replace=False)
    sample = comments.iloc[sample_idx].copy()

    sample['post_text'] = sample['post_id'].map(post_texts)
    sample = sample.dropna(subset=['post_text', 'content'])
    print(f"  Sampled {len(sample):,} (post, comment) pairs")

    # Collect unique texts to embed (avoid re-embedding duplicates)
    comment_texts = sample['content'].astype(str).tolist()
    post_texts_list = sample['post_text'].astype(str).tolist()

    # Random post pool for baseline
    all_post_texts = post_texts.dropna().values
    random_pool_idx = rng.choice(len(all_post_texts),
                                  size=min(5000, len(all_post_texts)),
                                  replace=False)
    random_pool_texts = [str(t) for t in all_post_texts[random_pool_idx]]

    # Embed all texts
    print("  Embedding comments...")
    comment_embs = embed_texts_batch(comment_texts, cache_name="sem_comments")

    print("  Embedding posts...")
    post_embs = embed_texts_batch(post_texts_list, cache_name="sem_posts")

    print("  Embedding random post pool...")
    random_embs = embed_texts_batch(random_pool_texts, cache_name="sem_random_pool")

    # Compute semantic specificity for each pair
    print("  Computing semantic specificity...")
    t0 = time.time()

    # Select a fixed set of random embeddings for each pair
    n_rand = min(n_random, len(random_embs))
    results = []

    for i in range(len(comment_embs)):
        rand_idx = rng.choice(len(random_embs), n_rand, replace=False)
        rand_emb_subset = random_embs[rand_idx]

        sim_actual = cosine_similarity(comment_embs[i], post_embs[i])
        sims_random = cosine_similarity_batch(comment_embs[i], rand_emb_subset)
        sim_random_mean = float(np.mean(sims_random))

        spec = sim_actual - sim_random_mean

        results.append({
            'sem_sim_actual': sim_actual,
            'sem_sim_random_mean': sim_random_mean,
            'specificity_semantic': spec,
        })

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{len(comment_embs):,} ({elapsed:.1f}s)")

    df = pd.DataFrame(results)
    elapsed = time.time() - t0
    print(f"  Computed semantic specificity for {len(df):,} pairs in {elapsed:.1f}s")

    # Also compute Jaccard for the same sample to allow direct comparison
    print("  Computing Jaccard for comparison...")
    jacc_results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        c_tok = content_tokens(str(row['content']))
        p_tok = content_tokens(str(row['post_text']))
        jacc_actual = jaccard(c_tok, p_tok)

        rand_posts = list(rng.choice(random_pool_texts, n_random, replace=False))
        jacc_randoms = [jaccard(c_tok, content_tokens(rp)) for rp in rand_posts]
        spec_lex = jacc_actual - np.nanmean(jacc_randoms)

        jacc_results.append({
            'jaccard_actual': jacc_actual,
            'specificity_lexical': spec_lex,
            'comment_len': len(tokenize(str(row['content']))),
        })

    jacc_df = pd.DataFrame(jacc_results)
    df = pd.concat([df, jacc_df], axis=1)

    # Summary
    print(f"\n  SEMANTIC SPECIFICITY:")
    print(f"  Cosine sim to actual post:  mean={df['sem_sim_actual'].mean():.4f}  "
          f"median={df['sem_sim_actual'].median():.4f}")
    print(f"  Cosine sim to random posts: mean={df['sem_sim_random_mean'].mean():.4f}  "
          f"median={df['sem_sim_random_mean'].median():.4f}")
    print(f"  Semantic specificity:       mean={df['specificity_semantic'].mean():.4f}  "
          f"median={df['specificity_semantic'].median():.4f}")

    # Cross-metric comparison
    mask = df['specificity_lexical'].notna() & df['specificity_semantic'].notna()
    corr = np.corrcoef(df.loc[mask, 'specificity_lexical'],
                       df.loc[mask, 'specificity_semantic'])[0, 1]
    print(f"\n  Correlation (Jaccard vs Semantic specificity): {corr:.3f}")

    # How many of the "lexically generic" are semantically relevant?
    lex_generic = df[
        (df['specificity_lexical'] >= -0.005) &
        (df['specificity_lexical'] <= 0.02)
    ]
    if len(lex_generic) > 0:
        sem_relevant = (lex_generic['specificity_semantic'] > 0.05).sum()
        print(f"\n  Of {len(lex_generic):,} lexically generic comments:")
        print(f"    {sem_relevant:,} ({sem_relevant/len(lex_generic)*100:.1f}%) "
              f"are semantically relevant (spec_semantic > 0.05)")

    # Breakdown by comment length
    print(f"\n  Semantic specificity by comment length:")
    for lo, hi in [(1, 10), (10, 25), (25, 50), (50, 100), (100, 200),
                   (200, 500)]:
        mask = (df['comment_len'] >= lo) & (df['comment_len'] < hi)
        sub = df.loc[mask, 'specificity_semantic']
        if len(sub) > 0:
            print(f"    [{lo:3d},{hi:3d}) words: n={len(sub):5d}  "
                  f"mean={sub.mean():.4f}  median={sub.median():.4f}")

    df.to_parquet(f"{CACHE}/semantic_relevance.parquet", index=False)

    # Also store texts for LLM judge sampling
    sample_for_judge = sample[['content', 'post_text']].copy()
    sample_for_judge.columns = ['comment_text', 'post_text']
    sample_for_judge = sample_for_judge.reset_index(drop=True)
    sample_for_judge = pd.concat([sample_for_judge, df.reset_index(drop=True)],
                                  axis=1)
    sample_for_judge.to_parquet(f"{CACHE}/semantic_relevance_with_texts.parquet",
                                 index=False)

    return df


def plot_semantic_relevance(df):
    """Generate semantic relevance figures (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Similarity distributions
    # (a) Jaccard: actual vs random
    ax = axes[0, 0]
    sns.kdeplot(df['jaccard_actual'].dropna(), ax=ax, color=PAL[0],
                label='Actual post', fill=True, alpha=0.3)
    ax.set_xlabel('Content-Word Jaccard')
    ax.set_ylabel('Density')
    ax.set_title('(a) Lexical Similarity')
    ax.legend(fontsize=8)

    # (b) Semantic: actual vs random
    ax = axes[0, 1]
    sns.kdeplot(df['sem_sim_actual'].dropna(), ax=ax, color=PAL[0],
                label='Actual post', fill=True, alpha=0.3)
    sns.kdeplot(df['sem_sim_random_mean'].dropna(), ax=ax, color=PAL[1],
                label='Random posts', fill=True, alpha=0.3)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('(b) Semantic Similarity')
    ax.legend(fontsize=8)

    # (c) Specificity distributions side by side
    ax = axes[0, 2]
    sns.kdeplot(df['specificity_lexical'].dropna(), ax=ax, color=PAL[0],
                label='Lexical', fill=True, alpha=0.3)
    sns.kdeplot(df['specificity_semantic'].dropna(), ax=ax, color=PAL[2],
                label='Semantic', fill=True, alpha=0.3)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Density')
    ax.set_title('(c) Specificity Distributions')
    ax.legend(fontsize=8)

    # Row 2: Deeper analysis
    # (d) Jaccard vs Semantic scatter
    ax = axes[1, 0]
    mask = df['specificity_lexical'].notna() & df['specificity_semantic'].notna()
    sub = df[mask].sample(min(5000, mask.sum()), random_state=42)
    ax.scatter(sub['specificity_lexical'], sub['specificity_semantic'],
               alpha=0.15, s=5, color=PAL[0], linewidth=0)
    corr = np.corrcoef(sub['specificity_lexical'],
                       sub['specificity_semantic'])[0, 1]
    ax.set_xlabel('Lexical Specificity (Jaccard)')
    ax.set_ylabel('Semantic Specificity (Embedding)')
    ax.set_title(f'(d) Lexical vs Semantic (r={corr:.2f})')
    ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.3)

    # (e) Semantic specificity by comment length
    ax = axes[1, 1]
    df_clean = df.dropna(subset=['specificity_semantic', 'comment_len'])
    df_clean = df_clean[df_clean['comment_len'] > 0].copy()
    bins = [0, 10, 25, 50, 100, 200, 500, 10000]
    labels = ['1-10', '11-25', '26-50', '51-100', '101-200', '201-500',
              '500+']
    df_clean['len_bin'] = pd.cut(df_clean['comment_len'], bins=bins,
                                  labels=labels)
    sns.barplot(x='len_bin', y='specificity_semantic', data=df_clean,
                ax=ax, color=PAL[2], errorbar='sd', capsize=0.15)
    ax.set_xlabel('Comment Length (tokens)')
    ax.set_ylabel('Mean Semantic Specificity')
    ax.set_title('(e) Semantic Specificity by Length')
    ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # (f) What are the "lexically generic" comments semantically?
    ax = axes[1, 2]
    lex_generic = df[
        (df['specificity_lexical'] >= -0.005) &
        (df['specificity_lexical'] <= 0.02)
    ]['specificity_semantic'].dropna()
    lex_specific = df[df['specificity_lexical'] > 0.02
                      ]['specificity_semantic'].dropna()
    if len(lex_generic) > 0:
        sns.kdeplot(lex_generic, ax=ax, color=PAL[1],
                    label='Lex. generic', fill=True, alpha=0.3)
    if len(lex_specific) > 0:
        sns.kdeplot(lex_specific, ax=ax, color=PAL[0],
                    label='Lex. specific', fill=True, alpha=0.3)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Semantic Specificity')
    ax.set_ylabel('Density')
    ax.set_title('(f) Semantic View of Lexically Generic')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/semantic_relevance.pdf")
    plt.close()
    print(f"  Saved {FIG_DIR}/semantic_relevance.pdf")


# ═══════════════════════════════════════════════════════════════════════
# 5. LLM-AS-JUDGE VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def analyze_judge_validation(
    primary_model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    calibration_model="bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
    n_high=500, n_zero=1000, n_negative=500,
    n_calibration=200,
):
    """Run LLM judge on stratified sample to validate automated metrics.

    Requires cache/semantic_relevance_with_texts.parquet from Phase 1.
    """
    print("\n" + "=" * 70)
    print("5. LLM-AS-JUDGE VALIDATION")
    print("=" * 70)

    # Load the sample with texts and scores
    data_path = f"{CACHE}/semantic_relevance_with_texts.parquet"
    if not os.path.exists(data_path):
        print(f"  ERROR: {data_path} not found. Run semantic relevance first.")
        return None

    full_df = pd.read_parquet(data_path)
    print(f"  Loaded {len(full_df):,} pairs with texts and scores")

    # Stratified sample by Jaccard specificity
    high = full_df[full_df['specificity_lexical'] > 0.02]
    zero = full_df[
        (full_df['specificity_lexical'] >= -0.005) &
        (full_df['specificity_lexical'] <= 0.02)
    ]
    negative = full_df[full_df['specificity_lexical'] < -0.005]

    rng = np.random.RandomState(42)
    samples = []
    for subset, n, label in [
        (high, n_high, "high"),
        (zero, n_zero, "zero"),
        (negative, n_negative, "negative"),
    ]:
        n_actual = min(n, len(subset))
        if n_actual > 0:
            s = subset.sample(n_actual, random_state=rng)
            s = s.copy()
            s['stratum'] = label
            samples.append(s)

    judge_sample = pd.concat(samples, ignore_index=True)
    print(f"  Stratified sample: {len(judge_sample):,} pairs "
          f"(high={len(judge_sample[judge_sample['stratum']=='high'])}, "
          f"zero={len(judge_sample[judge_sample['stratum']=='zero'])}, "
          f"negative={len(judge_sample[judge_sample['stratum']=='negative'])})")

    # Run primary judge
    print(f"\n  Running primary judge ({primary_model})...")
    primary_cache = f"{CACHE}/judge_primary.parquet"
    primary_results = run_judge_batch(
        judge_sample, model=primary_model, cache_path=primary_cache, delay=0.05
    )

    # Combine with sample data
    judge_df = pd.concat([
        judge_sample.reset_index(drop=True),
        primary_results.reset_index(drop=True),
    ], axis=1)

    # Run calibration judge on subset
    print(f"\n  Running calibration judge ({calibration_model}) "
          f"on {n_calibration} pairs...")
    calibration_sample = judge_sample.head(n_calibration)
    calibration_cache = f"{CACHE}/judge_calibration.parquet"
    calibration_results = run_judge_batch(
        calibration_sample, model=calibration_model,
        cache_path=calibration_cache, delay=0.1
    )

    # Inter-rater reliability
    try:
        irr = compute_inter_rater_reliability(
            primary_results.head(n_calibration), calibration_results
        )
        print(f"\n  INTER-RATER RELIABILITY ({primary_model} vs {calibration_model}):")
        print(f"    Category agreement: {irr.get('category_agreement', 0):.2%}")
        print(f"    Category kappa:     {irr.get('category_kappa', 0):.3f}")
        print(f"    Responsiveness corr: {irr.get('responsiveness_correlation', 0):.3f}")
        print(f"    Information corr:    {irr.get('information_correlation', 0):.3f}")
    except Exception as e:
        print(f"  WARNING: Could not compute inter-rater reliability: {e}")
        irr = {}

    # Summary statistics
    valid = judge_df[judge_df['category'] != 'error']
    print(f"\n  JUDGE RESULTS (n={len(valid):,} valid judgments):")

    print(f"\n  Category distribution:")
    cat_counts = valid['category'].value_counts()
    for cat, count in cat_counts.items():
        print(f"    {cat:<25s} {count:5d} ({count/len(valid)*100:.1f}%)")

    print(f"\n  Responsiveness: mean={valid['responsiveness'].mean():.2f}  "
          f"median={valid['responsiveness'].median():.1f}")
    print(f"  Information:    mean={valid['information'].mean():.2f}  "
          f"median={valid['information'].median():.1f}")

    # Correlation with automated metrics
    for metric_name, metric_col in [('Jaccard specificity', 'specificity_lexical'),
                                     ('Semantic specificity', 'specificity_semantic')]:
        mask = valid[metric_col].notna() & valid['responsiveness'].notna()
        if mask.sum() > 2:
            corr = np.corrcoef(
                pd.to_numeric(valid.loc[mask, metric_col]),
                pd.to_numeric(valid.loc[mask, 'responsiveness'])
            )[0, 1]
            print(f"  Correlation ({metric_name} vs responsiveness): {corr:.3f}")

    # Category breakdown by stratum
    print(f"\n  Category breakdown by Jaccard stratum:")
    for stratum in ['high', 'zero', 'negative']:
        sub = valid[valid['stratum'] == stratum]
        if len(sub) == 0:
            continue
        print(f"    {stratum}: ", end="")
        cats = sub['category'].value_counts(normalize=True)
        parts = [f"{cat}={pct:.0%}" for cat, pct in cats.head(3).items()]
        print(", ".join(parts))

    judge_df.to_parquet(f"{CACHE}/judge_results.parquet", index=False)
    return judge_df


def plot_judge_results(judge_df):
    """Generate LLM judge validation figures."""
    valid = judge_df[judge_df['category'] != 'error'].copy()
    if len(valid) == 0:
        print("  No valid judge results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Category distribution
    ax = axes[0, 0]
    cat_order = ['generic_affirmation', 'self_promotion', 'spam',
                 'on_topic', 'substantive', 'off_topic']
    cat_counts = valid['category'].value_counts()
    # Reindex to standard order, keeping only present categories
    cats_present = [c for c in cat_order if c in cat_counts.index]
    cats_present += [c for c in cat_counts.index if c not in cat_order]
    cat_vals = [cat_counts.get(c, 0) for c in cats_present]
    colors = [PAL[i % len(PAL)] for i in range(len(cats_present))]
    ax.barh(range(len(cats_present)), cat_vals, color=colors)
    ax.set_yticks(range(len(cats_present)))
    ax.set_yticklabels([c.replace('_', ' ') for c in cats_present], fontsize=9)
    ax.set_xlabel('Count')
    ax.set_title('(a) Comment Categories (LLM Judge)')
    ax.invert_yaxis()

    # (b) Responsiveness vs Jaccard specificity
    ax = axes[0, 1]
    mask = valid['specificity_lexical'].notna() & valid['responsiveness'].notna()
    sub = valid[mask]
    if len(sub) > 0:
        sns.boxplot(x='responsiveness', y='specificity_lexical', data=sub,
                    ax=ax, color=PAL[0], width=0.5)
        ax.set_xlabel('Responsiveness (LLM Judge)')
        ax.set_ylabel('Lexical Specificity (Jaccard)')
        ax.set_title('(b) Judge Score vs Lexical Specificity')
        ax.axhline(0, color='grey', linestyle='--', alpha=0.3)

    # (c) Responsiveness vs Semantic specificity
    ax = axes[1, 0]
    mask = valid['specificity_semantic'].notna() & valid['responsiveness'].notna()
    sub = valid[mask]
    if len(sub) > 0:
        sns.boxplot(x='responsiveness', y='specificity_semantic', data=sub,
                    ax=ax, color=PAL[2], width=0.5)
        ax.set_xlabel('Responsiveness (LLM Judge)')
        ax.set_ylabel('Semantic Specificity (Embedding)')
        ax.set_title('(c) Judge Score vs Semantic Specificity')
        ax.axhline(0, color='grey', linestyle='--', alpha=0.3)

    # (d) Category breakdown by Jaccard stratum
    ax = axes[1, 1]
    strata = ['high', 'zero', 'negative']
    cat_matrix = []
    for stratum in strata:
        sub = valid[valid['stratum'] == stratum]
        if len(sub) > 0:
            cats = sub['category'].value_counts(normalize=True)
            cat_matrix.append(cats)
        else:
            cat_matrix.append(pd.Series(dtype=float))

    if cat_matrix:
        cat_df = pd.DataFrame(cat_matrix, index=strata).fillna(0)
        cat_df.plot(kind='bar', stacked=True, ax=ax, colormap='Set2',
                    width=0.7, legend=True)
        ax.set_xlabel('Jaccard Specificity Stratum')
        ax.set_ylabel('Fraction')
        ax.set_title('(d) Categories by Specificity Stratum')
        ax.legend(fontsize=7, loc='upper right',
                  labels=[c.replace('_', ' ') for c in cat_df.columns])
        ax.set_xticklabels(strata, rotation=0)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/judge_validation.pdf")
    plt.close()
    print(f"  Saved {FIG_DIR}/judge_validation.pdf")


# ═══════════════════════════════════════════════════════════════════════
# 6. NESTED REPLY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_nested_replies(posts, comments, n_sample=5000, n_random=10):
    """Compare top-level comment engagement with nested reply engagement.

    Computes Jaccard and embedding specificity for nested replies
    (reply → parent comment) and compares with top-level (comment → post).
    """
    print("\n" + "=" * 70)
    print("6. NESTED REPLY ANALYSIS")
    print("=" * 70)

    # Identify nested replies (depth > 0 or parent_id refers to a comment)
    # Build comment ID lookup
    comment_ids = set(comments['id'].values)
    nested = comments[comments['parent_id'].isin(comment_ids)].copy()
    top_level = comments[~comments['parent_id'].isin(comment_ids)].copy()

    print(f"  Top-level comments: {len(top_level):,}")
    print(f"  Nested replies:     {len(nested):,} "
          f"({len(nested)/len(comments)*100:.1f}%)")

    # Build parent text lookup for nested replies (drop duplicates to ensure unique index)
    comment_text_map = comments.drop_duplicates('id', keep='first').set_index('id')['content']

    rng = np.random.RandomState(42)

    # === Nested reply pairs ===
    n_nested = min(n_sample, len(nested))
    nested_sample = nested.sample(n_nested, random_state=rng).copy()
    nested_sample['parent_text'] = nested_sample['parent_id'].map(comment_text_map)
    nested_sample = nested_sample.dropna(subset=['parent_text', 'content'])
    print(f"  Nested reply sample: {len(nested_sample):,}")

    # Random comment pool for nested baseline
    all_comment_texts = comment_text_map.dropna().values
    random_comment_idx = rng.choice(len(all_comment_texts),
                                     size=min(5000, len(all_comment_texts)),
                                     replace=False)
    random_comment_pool = [str(t) for t in all_comment_texts[random_comment_idx]]

    # Compute Jaccard for nested replies
    print("  Computing Jaccard specificity for nested replies...")
    t0 = time.time()
    nested_results = []
    for i, (_, row) in enumerate(nested_sample.iterrows()):
        reply = str(row['content'])
        parent = str(row['parent_text'])

        c_tok = content_tokens(reply)
        p_tok = content_tokens(parent)
        jacc_actual = jaccard(c_tok, p_tok)

        rand_comments = list(rng.choice(random_comment_pool, n_random, replace=False))
        jacc_randoms = [jaccard(c_tok, content_tokens(rc)) for rc in rand_comments]
        spec_lex = jacc_actual - np.nanmean(jacc_randoms)

        nested_results.append({
            'pair_type': 'nested_reply',
            'jaccard_actual': jacc_actual,
            'specificity_lexical': spec_lex,
            'comment_len': len(tokenize(reply)),
        })

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{len(nested_sample):,} ({elapsed:.1f}s)")

    nested_df = pd.DataFrame(nested_results)
    elapsed = time.time() - t0
    print(f"  Nested Jaccard: {len(nested_df):,} pairs in {elapsed:.1f}s")

    # === Top-level pairs (re-sample for comparison) ===
    posts_dedup = posts.drop_duplicates('id', keep='first')
    post_texts = posts_dedup.set_index('id')['full_text']

    n_top = min(n_sample, len(top_level))
    top_sample = top_level.sample(n_top, random_state=rng).copy()
    top_sample['post_text'] = top_sample['post_id'].map(post_texts)
    top_sample = top_sample.dropna(subset=['post_text', 'content'])

    all_post_texts = post_texts.dropna().values
    random_post_idx = rng.choice(len(all_post_texts),
                                  size=min(5000, len(all_post_texts)),
                                  replace=False)
    random_post_pool = [str(t) for t in all_post_texts[random_post_idx]]

    print("  Computing Jaccard specificity for top-level comments...")
    t0 = time.time()
    top_results = []
    for i, (_, row) in enumerate(top_sample.iterrows()):
        comment = str(row['content'])
        post = str(row['post_text'])

        c_tok = content_tokens(comment)
        p_tok = content_tokens(post)
        jacc_actual = jaccard(c_tok, p_tok)

        rand_posts = list(rng.choice(random_post_pool, n_random, replace=False))
        jacc_randoms = [jaccard(c_tok, content_tokens(rp)) for rp in rand_posts]
        spec_lex = jacc_actual - np.nanmean(jacc_randoms)

        top_results.append({
            'pair_type': 'top_level',
            'jaccard_actual': jacc_actual,
            'specificity_lexical': spec_lex,
            'comment_len': len(tokenize(comment)),
        })

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{len(top_sample):,} ({elapsed:.1f}s)")

    top_df = pd.DataFrame(top_results)
    elapsed = time.time() - t0
    print(f"  Top-level Jaccard: {len(top_df):,} pairs in {elapsed:.1f}s")

    # === Embedding specificity for nested replies ===
    print("  Computing embedding specificity for nested replies...")
    nested_reply_texts = nested_sample['content'].astype(str).tolist()
    nested_parent_texts = nested_sample['parent_text'].astype(str).tolist()

    nested_reply_embs = embed_texts_batch(
        nested_reply_texts, cache_name="nested_replies"
    )
    nested_parent_embs = embed_texts_batch(
        nested_parent_texts, cache_name="nested_parents"
    )
    random_comment_embs = embed_texts_batch(
        random_comment_pool, cache_name="nested_random_pool"
    )

    n_rand = min(n_random, len(random_comment_embs))
    sem_results = []
    for i in range(len(nested_reply_embs)):
        rand_idx = rng.choice(len(random_comment_embs), n_rand, replace=False)
        rand_emb_subset = random_comment_embs[rand_idx]

        sim_actual = cosine_similarity(nested_reply_embs[i], nested_parent_embs[i])
        sims_random = cosine_similarity_batch(nested_reply_embs[i], rand_emb_subset)
        sim_random_mean = float(np.mean(sims_random))
        spec = sim_actual - sim_random_mean

        sem_results.append({
            'sem_sim_actual': sim_actual,
            'specificity_semantic': spec,
        })

    nested_sem_df = pd.DataFrame(sem_results)
    nested_df = pd.concat([nested_df.reset_index(drop=True),
                           nested_sem_df.reset_index(drop=True)], axis=1)

    # === Summary comparison ===
    combined = pd.concat([top_df, nested_df], ignore_index=True)

    print(f"\n  COMPARISON: Top-level vs Nested Replies")
    print(f"  {'Metric':<30s} {'Top-level':>12s} {'Nested':>12s}")
    print(f"  {'─' * 54}")

    for col, label in [
        ('jaccard_actual', 'Mean Jaccard'),
        ('specificity_lexical', 'Mean Lexical Specificity'),
    ]:
        top_val = top_df[col].mean()
        nested_val = nested_df[col].mean()
        print(f"  {label:<30s} {top_val:12.4f} {nested_val:12.4f}")

    top_zero = (top_df['jaccard_actual'] == 0).mean()
    nested_zero = (nested_df['jaccard_actual'] == 0).mean()
    print(f"  {'% Zero Jaccard':<30s} {top_zero:11.1%} {nested_zero:11.1%}")

    top_above = (top_df['jaccard_actual'] > 0.05).mean()
    nested_above = (nested_df['jaccard_actual'] > 0.05).mean()
    print(f"  {'% Jaccard > 0.05':<30s} {top_above:11.1%} {nested_above:11.1%}")

    if 'specificity_semantic' in nested_df.columns:
        nested_sem_mean = nested_df['specificity_semantic'].mean()
        print(f"  {'Mean Semantic Spec (nested)':<30s} {'':>12s} {nested_sem_mean:12.4f}")

    combined.to_parquet(f"{CACHE}/nested_reply_analysis.parquet", index=False)
    return combined


def plot_nested_comparison(combined_df):
    """Generate nested reply comparison figures."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    top = combined_df[combined_df['pair_type'] == 'top_level']
    nested = combined_df[combined_df['pair_type'] == 'nested_reply']

    # (a) Jaccard distributions
    ax = axes[0]
    sns.kdeplot(top['jaccard_actual'].dropna(), ax=ax, color=PAL[0],
                label=f'Top-level (n={len(top):,})', fill=True, alpha=0.3)
    sns.kdeplot(nested['jaccard_actual'].dropna(), ax=ax, color=PAL[2],
                label=f'Nested (n={len(nested):,})', fill=True, alpha=0.3)
    ax.set_xlabel('Content-Word Jaccard Similarity')
    ax.set_ylabel('Density')
    ax.set_title('(a) Lexical Engagement')
    ax.legend(fontsize=8)
    ax.set_xlim(-0.01, 0.3)

    # (b) Specificity distributions
    ax = axes[1]
    sns.kdeplot(top['specificity_lexical'].dropna(), ax=ax, color=PAL[0],
                label='Top-level', fill=True, alpha=0.3)
    sns.kdeplot(nested['specificity_lexical'].dropna(), ax=ax, color=PAL[2],
                label='Nested', fill=True, alpha=0.3)
    ax.axvline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lexical Specificity')
    ax.set_ylabel('Density')
    ax.set_title('(b) Specificity: Top-level vs Nested')
    ax.legend(fontsize=8)

    # (c) Bar comparison of key metrics
    ax = axes[2]
    metrics = {
        'Mean\nJaccard': (top['jaccard_actual'].mean(),
                          nested['jaccard_actual'].mean()),
        '% Zero\nJaccard': ((top['jaccard_actual'] == 0).mean(),
                             (nested['jaccard_actual'] == 0).mean()),
        'Mean\nSpecificity': (top['specificity_lexical'].mean(),
                              nested['specificity_lexical'].mean()),
    }
    x = np.arange(len(metrics))
    width = 0.35
    top_vals = [v[0] for v in metrics.values()]
    nested_vals = [v[1] for v in metrics.values()]
    ax.bar(x - width/2, top_vals, width, label='Top-level', color=PAL[0])
    ax.bar(x + width/2, nested_vals, width, label='Nested', color=PAL[2])
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=9)
    ax.set_title('(c) Engagement Comparison')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/nested_comparison.pdf")
    plt.close()
    print(f"  Saved {FIG_DIR}/nested_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════
# DATASET OVERVIEW FIGURE
# ═══════════════════════════════════════════════════════════════════════

def plot_dataset_overview(posts, comments):
    """Generate dataset overview figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Comments per post
    ax = axes[0]
    cpp = comments.groupby('post_id').size()
    sns.histplot(cpp.clip(upper=50), bins=50, ax=ax, color=PAL[0],
                 edgecolor='white')
    ax.set_xlabel('Comments per Post')
    ax.set_ylabel('Number of Posts')
    ax.set_title(f'(a) Comments per Post (median={cpp.median():.0f})')
    ax.set_xlim(0, 50)

    # (b) Comments per agent
    ax = axes[1]
    cpa = comments.groupby('author_id').size()
    sns.histplot(cpa.clip(upper=100), bins=50, ax=ax, color=PAL[1],
                 edgecolor='white')
    ax.set_xlabel('Comments per Agent')
    ax.set_ylabel('Number of Agents')
    ax.set_title(f'(b) Comments per Agent (median={cpa.median():.0f})')
    ax.set_xlim(0, 100)

    # (c) Top submolts
    ax = axes[2]
    submolt_counts = posts['submolt'].value_counts().head(15)
    sns.barplot(x=submolt_counts.values, y=submolt_counts.index, ax=ax,
                color=PAL[2], orient='h')
    ax.set_xlabel('Number of Posts')
    ax.set_title('(c) Top 15 Submolts')

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/dataset_overview.pdf")
    plt.close()
    print(f"  Saved {FIG_DIR}/dataset_overview.pdf")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Moltbook analyses")
    parser.add_argument("--analyses", nargs="*", default=None,
                        help="Which analyses to run (1-6, or 'all'). "
                             "Default: all")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding-based analyses (4, 6 embeddings)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge analysis (5)")
    args = parser.parse_args()

    # Determine which analyses to run
    if args.analyses is None or 'all' in (args.analyses or []):
        run_set = {1, 2, 3, 4, 5, 6}
    else:
        run_set = {int(x) for x in args.analyses}
    if args.skip_embeddings:
        run_set -= {4}
    if args.skip_judge:
        run_set -= {5}

    t_start = time.time()

    print("Loading combined dataset...")
    posts, comments = load_combined()
    print(f"  Posts: {len(posts):,}   Comments: {len(comments):,}")

    # Dataset overview
    print("\nGenerating dataset overview...")
    plot_dataset_overview(posts, comments)

    # 1. Agent entropy — re-use cached computation, just re-plot
    if 1 in run_set:
        entropy_path = f"{CACHE}/agent_entropy.parquet"
        if os.path.exists(entropy_path):
            print("\nLoading cached agent entropy...")
            agent_df = pd.read_parquet(entropy_path)
            print(f"  Loaded {len(agent_df):,} agents from cache")
        else:
            agent_df = analyze_agent_entropy(comments)
        plot_agent_entropy(agent_df)

    # 2. Saturation — re-use cached computation, just re-plot
    if 2 in run_set:
        sat_path = f"{CACHE}/saturation_curves.parquet"
        if os.path.exists(sat_path):
            print("\nLoading cached saturation curves...")
            sat_df = pd.read_parquet(sat_path)
            print(f"  Loaded {len(sat_df)} positions from cache")
        else:
            sat_df = analyze_saturation(posts, comments)
        plot_saturation(sat_df)

    # 3. Lexical relevance
    if 3 in run_set:
        rel_path = f"{CACHE}/relevance_scores.parquet"
        if os.path.exists(rel_path):
            print("\nLoading cached relevance scores...")
            rel_df = pd.read_parquet(rel_path)
            print(f"  Loaded {len(rel_df):,} pairs from cache")
        else:
            rel_df = analyze_relevance(posts, comments)
        plot_relevance(rel_df)

    # 4. Semantic relevance (embeddings)
    if 4 in run_set:
        sem_path = f"{CACHE}/semantic_relevance.parquet"
        if os.path.exists(sem_path):
            print("\nLoading cached semantic relevance...")
            sem_df = pd.read_parquet(sem_path)
            print(f"  Loaded {len(sem_df):,} pairs from cache")
        else:
            sem_df = analyze_semantic_relevance(posts, comments)
        plot_semantic_relevance(sem_df)

    # 5. LLM judge validation
    if 5 in run_set:
        judge_path = f"{CACHE}/judge_results.parquet"
        if os.path.exists(judge_path):
            print("\nLoading cached judge results...")
            judge_df = pd.read_parquet(judge_path)
            print(f"  Loaded {len(judge_df):,} judgments from cache")
        else:
            judge_df = analyze_judge_validation()
        if judge_df is not None:
            plot_judge_results(judge_df)

    # 6. Nested reply analysis
    if 6 in run_set:
        nested_path = f"{CACHE}/nested_reply_analysis.parquet"
        if os.path.exists(nested_path):
            print("\nLoading cached nested reply analysis...")
            nested_df = pd.read_parquet(nested_path)
            print(f"  Loaded {len(nested_df):,} pairs from cache")
        else:
            nested_df = analyze_nested_replies(posts, comments)
        plot_nested_comparison(nested_df)

    total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ALL ANALYSES COMPLETE in {total:.0f}s")
    print(f"{'=' * 70}")
    print(f"  Results: {CACHE}/agent_entropy.parquet")
    print(f"           {CACHE}/saturation_curves.parquet")
    print(f"           {CACHE}/relevance_scores.parquet")
    print(f"           {CACHE}/semantic_relevance.parquet")
    print(f"           {CACHE}/judge_results.parquet")
    print(f"           {CACHE}/nested_reply_analysis.parquet")
    print(f"  Figures: {FIG_DIR}/dataset_overview.pdf")
    print(f"           {FIG_DIR}/agent_entropy.pdf")
    print(f"           {FIG_DIR}/saturation_curve.pdf")
    print(f"           {FIG_DIR}/relevance.pdf")
    print(f"           {FIG_DIR}/semantic_relevance.pdf")
    print(f"           {FIG_DIR}/judge_validation.pdf")
    print(f"           {FIG_DIR}/nested_comparison.pdf")
