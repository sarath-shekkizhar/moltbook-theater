"""
Run metrics on Moltbook (parent, reply) pairs.
Uses lnajt/moltbook as the largest dataset.

Two analysis units:
  A) Post → top-level comment (2.67M pairs)
  B) Comment → nested reply   (167K pairs)

Runs on a stratified sample first for quick verification,
then can scale to full dataset.
"""

import pandas as pd
import numpy as np
from collections import Counter
from metrics import compute_pair_metrics, tokenize
import time
import os

CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)


def load_lnajt():
    """Load the lnajt/moltbook dataset (largest available)."""
    posts_path = f"{CACHE}/lnajt_posts.parquet"
    comments_path = f"{CACHE}/lnajt_comments.parquet"

    if os.path.exists(posts_path) and os.path.exists(comments_path):
        print("Loading lnajt from cache...")
        posts = pd.read_parquet(posts_path)
        comments = pd.read_parquet(comments_path)
    else:
        print("Loading lnajt from HuggingFace...")
        posts = pd.read_parquet("hf://datasets/lnajt/moltbook/posts.parquet")
        comments = pd.read_parquet("hf://datasets/lnajt/moltbook/comments.parquet")
        posts.to_parquet(posts_path)
        comments.to_parquet(comments_path)
        print("Cached.")

    return posts, comments


def build_pairs(posts, comments):
    """
    Build (parent_text, reply_text) pairs for both analysis units.

    Returns:
        pairs_A: DataFrame — post→comment pairs (top-level)
        pairs_B: DataFrame — comment→reply pairs (nested)
    """
    print("\nBuilding (parent, reply) pairs...")

    # === Unit A: Post → top-level comment ===
    top_level = comments[comments['parent_id'].isna()].copy()

    # Combine post title + body as parent text
    posts['full_text'] = posts['title'].fillna('') + '\n' + posts['body'].fillna('')
    post_text = posts[['id', 'full_text', 'submolt', 'author_id']].rename(
        columns={'id': 'post_id', 'full_text': 'parent_text',
                 'author_id': 'post_author_id'}
    )

    pairs_A = top_level.merge(post_text, on='post_id', how='inner')
    pairs_A = pairs_A.rename(columns={'body': 'reply_text',
                                       'author_id': 'reply_author_id'})
    pairs_A['pair_type'] = 'post_comment'

    print(f"  Unit A (post→comment): {len(pairs_A):,} pairs")

    # === Unit B: Comment → nested reply ===
    nested = comments[comments['parent_id'].notna()].copy()

    # Get parent comment text
    comment_text = comments[['id', 'body', 'author_id']].rename(
        columns={'id': 'parent_id', 'body': 'parent_text',
                 'author_id': 'parent_author_id'}
    )

    pairs_B = nested.merge(comment_text, on='parent_id', how='inner')
    pairs_B = pairs_B.rename(columns={'body': 'reply_text',
                                       'author_id': 'reply_author_id'})
    pairs_B['pair_type'] = 'comment_reply'

    print(f"  Unit B (comment→reply): {len(pairs_B):,} pairs")

    return pairs_A, pairs_B


def sample_pairs(pairs_A, pairs_B, n_per_type=5000, seed=42):
    """Stratified sample for quick metric computation."""
    rng = np.random.RandomState(seed)

    sample_A = pairs_A.sample(n=min(n_per_type, len(pairs_A)),
                               random_state=rng)
    sample_B = pairs_B.sample(n=min(n_per_type, len(pairs_B)),
                               random_state=rng)

    return pd.concat([sample_A, sample_B], ignore_index=True)


def compute_metrics_batch(df, max_rows=None):
    """Compute all metrics for a DataFrame of (parent_text, reply_text) pairs."""
    if max_rows:
        df = df.head(max_rows)

    print(f"\nComputing metrics on {len(df):,} pairs...")
    t0 = time.time()

    results = []
    for i, row in df.iterrows():
        parent = str(row.get('parent_text', '') or '')
        reply = str(row.get('reply_text', '') or '')
        m = compute_pair_metrics(parent, reply)
        m['pair_type'] = row.get('pair_type', 'unknown')
        m['post_id'] = row.get('post_id', '')
        m['submolt'] = row.get('submolt', '')
        results.append(m)

        if (len(results) % 2000) == 0:
            elapsed = time.time() - t0
            rate = len(results) / elapsed
            print(f"  {len(results):,} done ({rate:.0f} pairs/sec)")

    elapsed = time.time() - t0
    print(f"  Finished {len(results):,} pairs in {elapsed:.1f}s "
          f"({len(results)/elapsed:.0f} pairs/sec)")

    return pd.DataFrame(results)


def summarize_metrics(metrics_df):
    """Print summary statistics for all metrics."""
    print("\n" + "=" * 80)
    print("METRIC SUMMARY")
    print("=" * 80)

    metric_cols = [c for c in metrics_df.columns
                   if c not in ('pair_type', 'post_id', 'submolt')]

    for pair_type in ['post_comment', 'comment_reply']:
        sub = metrics_df[metrics_df['pair_type'] == pair_type]
        if len(sub) == 0:
            continue
        print(f"\n{'─' * 80}")
        print(f"  {pair_type.upper()} pairs (n={len(sub):,})")
        print(f"{'─' * 80}")
        print(f"  {'Metric':<25s} {'mean':>8s} {'median':>8s} "
              f"{'std':>8s} {'p5':>8s} {'p95':>8s}")
        print(f"  {'─' * 73}")

        for col in metric_cols:
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            print(f"  {col:<25s} {vals.mean():8.3f} {vals.median():8.3f} "
                  f"{vals.std():8.3f} {vals.quantile(0.05):8.3f} "
                  f"{vals.quantile(0.95):8.3f}")

    # Also show joint stats
    print(f"\n{'─' * 80}")
    print(f"  ALL pairs (n={len(metrics_df):,})")
    print(f"{'─' * 80}")
    for col in metric_cols:
        vals = metrics_df[col].dropna()
        if len(vals) == 0:
            continue
        print(f"  {col:<25s} {vals.mean():8.3f} {vals.median():8.3f} "
              f"{vals.std():8.3f} {vals.quantile(0.05):8.3f} "
              f"{vals.quantile(0.95):8.3f}")


def show_examples(df, metrics_df, n=3):
    """Show high-theater and low-theater examples side by side."""
    print("\n" + "=" * 80)
    print("EXAMPLE PAIRS — sorted by NCD (low=redundant, high=independent)")
    print("=" * 80)

    # Merge metrics back to get parent/reply text
    merged = df.reset_index(drop=True)
    merged = pd.concat([merged, metrics_df.reset_index(drop=True)], axis=1)
    merged = merged.dropna(subset=['ncd'])

    # Lowest NCD (most redundant replies)
    print("\n── MOST REDUNDANT (lowest NCD) ──")
    for _, row in merged.nsmallest(n, 'ncd').iterrows():
        print(f"\n  NCD={row['ncd']:.3f} | jaccard={row['jaccard']:.3f} | "
              f"bigram_novelty={row['bigram_novelty']:.3f}")
        print(f"  PARENT: {str(row.get('parent_text', ''))[:200]}")
        print(f"  REPLY:  {str(row.get('reply_text', ''))[:200]}")

    # Highest NCD (most independent replies)
    print("\n── MOST INDEPENDENT (highest NCD) ──")
    for _, row in merged.nlargest(n, 'ncd').iterrows():
        print(f"\n  NCD={row['ncd']:.3f} | jaccard={row['jaccard']:.3f} | "
              f"bigram_novelty={row['bigram_novelty']:.3f}")
        print(f"  PARENT: {str(row.get('parent_text', ''))[:200]}")
        print(f"  REPLY:  {str(row.get('reply_text', ''))[:200]}")

    # Mid-range NCD
    print("\n── MIDDLE RANGE (NCD near median) ──")
    median_ncd = merged['ncd'].median()
    near_median = merged.iloc[(merged['ncd'] - median_ncd).abs().argsort()[:n]]
    for _, row in near_median.iterrows():
        print(f"\n  NCD={row['ncd']:.3f} | jaccard={row['jaccard']:.3f} | "
              f"bigram_novelty={row['bigram_novelty']:.3f}")
        print(f"  PARENT: {str(row.get('parent_text', ''))[:200]}")
        print(f"  REPLY:  {str(row.get('reply_text', ''))[:200]}")


if __name__ == "__main__":
    posts, comments = load_lnajt()
    pairs_A, pairs_B = build_pairs(posts, comments)

    # Sample for quick verification
    sample = sample_pairs(pairs_A, pairs_B, n_per_type=5000)
    print(f"\nSampled {len(sample):,} pairs for metric computation")

    # Compute metrics
    metrics_df = compute_metrics_batch(sample)

    # Save
    metrics_df.to_parquet(f"{CACHE}/metrics_sample.parquet")
    print(f"\nSaved metrics to {CACHE}/metrics_sample.parquet")

    # Summarize
    summarize_metrics(metrics_df)

    # Show examples
    show_examples(sample, metrics_df, n=3)
