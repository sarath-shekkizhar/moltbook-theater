"""
Run three focused analyses on the combined Moltbook dataset.

1. Agent Behavioral Entropy — per-agent diversity
2. Information Saturation — marginal info gain per comment position
3. Post-Comment Relevance — specificity of comments to their posts

Outputs:
  cache/agent_entropy.parquet
  cache/saturation_curves.parquet
  cache/relevance_scores.parquet
  paper-tex/figures/*.pdf
"""

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
    t_start = time.time()

    print("Loading combined dataset...")
    posts, comments = load_combined()
    print(f"  Posts: {len(posts):,}   Comments: {len(comments):,}")

    # Dataset overview
    print("\nGenerating dataset overview...")
    plot_dataset_overview(posts, comments)

    # 1. Agent entropy  — re-use cached computation, just re-plot
    entropy_path = f"{CACHE}/agent_entropy.parquet"
    if os.path.exists(entropy_path):
        print("\nLoading cached agent entropy...")
        agent_df = pd.read_parquet(entropy_path)
        print(f"  Loaded {len(agent_df):,} agents from cache")
    else:
        agent_df = analyze_agent_entropy(comments)
    plot_agent_entropy(agent_df)

    # 2. Saturation — re-use cached computation, just re-plot
    sat_path = f"{CACHE}/saturation_curves.parquet"
    if os.path.exists(sat_path):
        print("\nLoading cached saturation curves...")
        sat_df = pd.read_parquet(sat_path)
        print(f"  Loaded {len(sat_df)} positions from cache")
    else:
        sat_df = analyze_saturation(posts, comments)
    plot_saturation(sat_df)

    # 3. Relevance — always re-run (new metric)
    rel_df = analyze_relevance(posts, comments)
    plot_relevance(rel_df)

    total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ALL ANALYSES COMPLETE in {total:.0f}s")
    print(f"{'=' * 70}")
    print(f"  Results: {CACHE}/agent_entropy.parquet")
    print(f"           {CACHE}/saturation_curves.parquet")
    print(f"           {CACHE}/relevance_scores.parquet")
    print(f"  Figures: {FIG_DIR}/dataset_overview.pdf")
    print(f"           {FIG_DIR}/agent_entropy.pdf")
    print(f"           {FIG_DIR}/saturation_curve.pdf")
    print(f"           {FIG_DIR}/relevance.pdf")
