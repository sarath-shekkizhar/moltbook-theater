"""
Step 0: Load Moltbook data and compute basic statistics.
Verify the dataset structure and understand what we're working with
before building any metrics.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter, defaultdict
import json
import os

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def load_moltbook():
    """Load posts and comments from HuggingFace, cache as parquet."""
    posts_path = f"{CACHE_DIR}/posts.parquet"
    comments_path = f"{CACHE_DIR}/comments.parquet"

    if os.path.exists(posts_path) and os.path.exists(comments_path):
        print("Loading from cache...")
        posts = pd.read_parquet(posts_path)
        comments = pd.read_parquet(comments_path)
    else:
        print("Downloading from HuggingFace...")
        posts_ds = load_dataset("AIcell/moltbook-data", "posts", split="train")
        comments_ds = load_dataset("AIcell/moltbook-data", "comments", split="train")

        posts = posts_ds.to_pandas()
        comments = comments_ds.to_pandas()

        posts.to_parquet(posts_path)
        comments.to_parquet(comments_path)
        print(f"Cached to {CACHE_DIR}/")

    return posts, comments


def basic_stats(posts, comments):
    """Print basic dataset statistics."""
    print("=" * 70)
    print("MOLTBOOK DATASET OVERVIEW")
    print("=" * 70)

    print(f"\n--- Posts ---")
    print(f"  Total posts:           {len(posts):,}")
    print(f"  Unique authors:        {posts['author_id'].nunique():,}")
    print(f"  Unique submolts:       {posts['submolt_name'].nunique():,}")
    print(f"  Date range:            {posts['created_at'].min()} → {posts['created_at'].max()}")
    print(f"  Posts with content:    {posts['content'].notna().sum():,}")
    print(f"  Posts with title only: {(posts['content'].isna() | (posts['content'] == '')).sum():,}")

    print(f"\n--- Comments ---")
    print(f"  Total comments:        {len(comments):,}")
    print(f"  Unique authors:        {comments['author_id'].nunique():,}")
    print(f"  Top-level (depth=0):   {(comments['depth'] == 0).sum():,}")
    print(f"  Nested (depth>0):      {(comments['depth'] > 0).sum():,}")
    print(f"  Max depth:             {comments['depth'].max()}")

    print(f"\n--- Depth distribution ---")
    depth_counts = comments['depth'].value_counts().sort_index()
    for d in range(min(15, depth_counts.index.max() + 1)):
        if d in depth_counts.index:
            pct = depth_counts[d] / len(comments) * 100
            print(f"  depth={d:2d}: {depth_counts[d]:>10,} ({pct:5.1f}%)")
    if depth_counts.index.max() > 14:
        remaining = depth_counts[depth_counts.index > 14].sum()
        print(f"  depth>14: {remaining:>10,} ({remaining / len(comments) * 100:5.1f}%)")

    print(f"\n--- Engagement ---")
    print(f"  Post upvotes:    mean={posts['upvotes'].mean():.1f}, median={posts['upvotes'].median():.0f}, max={posts['upvotes'].max()}")
    print(f"  Post downvotes:  mean={posts['downvotes'].mean():.1f}, median={posts['downvotes'].median():.0f}, max={posts['downvotes'].max()}")
    print(f"  Comment upvotes: mean={comments['upvotes'].mean():.1f}, median={comments['upvotes'].median():.0f}, max={comments['upvotes'].max()}")
    print(f"  Comments/post:   mean={posts['comment_count'].mean():.1f}, median={posts['comment_count'].median():.0f}, max={posts['comment_count'].max()}")

    print(f"\n--- Author overlap ---")
    post_authors = set(posts['author_id'].dropna().unique())
    comment_authors = set(comments['author_id'].dropna().unique())
    both = post_authors & comment_authors
    print(f"  Post-only authors:    {len(post_authors - comment_authors):,}")
    print(f"  Comment-only authors: {len(comment_authors - post_authors):,}")
    print(f"  Both post & comment:  {len(both):,}")

    print(f"\n--- Top 10 submolts by post count ---")
    top_submolts = posts['submolt_display_name'].value_counts().head(10)
    for name, count in top_submolts.items():
        print(f"  {name:30s} {count:>8,}")

    return depth_counts


def build_threads(posts, comments):
    """
    Build thread trees from parent_id.
    Returns a dict: post_id -> list of comment chains.
    Also returns thread-level statistics.
    """
    print("\n" + "=" * 70)
    print("THREAD RECONSTRUCTION")
    print("=" * 70)

    # Index comments by post_id for fast lookup
    comments_by_post = comments.groupby('post_id')

    # For nested comments, build parent->children mapping
    children_of = defaultdict(list)  # parent_id -> [comment rows]
    for _, row in comments.iterrows():
        if pd.notna(row['parent_id']):
            children_of[row['parent_id']].append(row)

    # Thread statistics
    thread_stats = {
        'post_id': [],
        'total_comments': [],
        'max_depth': [],
        'unique_agents': [],
        'has_nested': [],   # depth > 0
        'num_dyadic_chains': [],  # back-and-forth between 2 agents
    }

    # Count dyadic exchanges: post_author <-> commenter back-and-forth
    dyadic_threads = []

    n_posts_with_comments = 0
    for post_id, group in comments_by_post:
        n_posts_with_comments += 1
        post_row = posts[posts['id'] == post_id]
        if len(post_row) == 0:
            continue
        post_author = post_row.iloc[0]['author_id']

        group_sorted = group.sort_values('created_at')
        agents_in_thread = set(group_sorted['author_id'].dropna().unique())
        if pd.notna(post_author):
            agents_in_thread.add(post_author)

        max_depth = group_sorted['depth'].max()

        thread_stats['post_id'].append(post_id)
        thread_stats['total_comments'].append(len(group))
        thread_stats['max_depth'].append(max_depth)
        thread_stats['unique_agents'].append(len(agents_in_thread))
        thread_stats['has_nested'].append(max_depth > 0)

        # Count dyadic chains: look for parent-child comment pairs
        # where the author alternates (A replies to B replies to A...)
        dyadic_count = 0
        if max_depth >= 1:
            for _, c in group_sorted[group_sorted['depth'] >= 1].iterrows():
                if pd.notna(c['parent_id']) and pd.notna(c['author_id']):
                    # Find parent comment
                    parent_mask = group_sorted['id'] == c['parent_id']
                    if parent_mask.any():
                        parent = group_sorted[parent_mask].iloc[0]
                        if (pd.notna(parent['author_id']) and
                                c['author_id'] != parent['author_id']):
                            dyadic_count += 1

        thread_stats['num_dyadic_chains'].append(dyadic_count)

        # If there are extended back-and-forth exchanges, record them
        if dyadic_count >= 3:
            dyadic_threads.append({
                'post_id': post_id,
                'dyadic_exchanges': dyadic_count,
                'unique_agents': len(agents_in_thread),
                'max_depth': max_depth,
            })

        # Progress
        if n_posts_with_comments % 50000 == 0:
            print(f"  Processed {n_posts_with_comments:,} posts with comments...")

    thread_df = pd.DataFrame(thread_stats)

    print(f"\n--- Thread Statistics ---")
    print(f"  Posts with comments: {len(thread_df):,}")
    print(f"  Posts with nested replies (depth>0): {thread_df['has_nested'].sum():,}")
    print(f"\n  Comments per thread:")
    print(f"    mean={thread_df['total_comments'].mean():.1f}, "
          f"median={thread_df['total_comments'].median():.0f}, "
          f"max={thread_df['total_comments'].max()}")
    print(f"\n  Unique agents per thread:")
    print(f"    mean={thread_df['unique_agents'].mean():.1f}, "
          f"median={thread_df['unique_agents'].median():.0f}, "
          f"max={thread_df['unique_agents'].max()}")
    print(f"\n  Max depth per thread:")
    for d in range(8):
        count = (thread_df['max_depth'] == d).sum()
        if count > 0:
            print(f"    max_depth={d}: {count:,} threads ({count/len(thread_df)*100:.1f}%)")
    deep = (thread_df['max_depth'] >= 8).sum()
    print(f"    max_depth>=8: {deep:,} threads ({deep/len(thread_df)*100:.1f}%)")

    print(f"\n  Dyadic exchanges per thread (back-and-forth between different agents):")
    print(f"    mean={thread_df['num_dyadic_chains'].mean():.1f}, "
          f"median={thread_df['num_dyadic_chains'].median():.0f}")
    print(f"    Threads with 3+ dyadic exchanges: {len(dyadic_threads):,}")
    print(f"    Threads with 5+ dyadic exchanges: "
          f"{sum(1 for t in dyadic_threads if t['dyadic_exchanges'] >= 5):,}")

    return thread_df, dyadic_threads


def sample_conversations(posts, comments, n=5):
    """Print a few example conversations to see what we're working with."""
    print("\n" + "=" * 70)
    print(f"SAMPLE CONVERSATIONS (n={n})")
    print("=" * 70)

    # Find posts with nested comments (depth >= 2) for interesting threads
    nested_posts = comments[comments['depth'] >= 2]['post_id'].unique()
    if len(nested_posts) == 0:
        nested_posts = comments[comments['depth'] >= 1]['post_id'].unique()

    rng = np.random.RandomState(42)
    sample_post_ids = rng.choice(nested_posts, size=min(n, len(nested_posts)),
                                  replace=False)

    for i, post_id in enumerate(sample_post_ids):
        post = posts[posts['id'] == post_id]
        if len(post) == 0:
            continue
        post = post.iloc[0]

        print(f"\n{'─' * 70}")
        print(f"Thread {i+1} | submolt: {post.get('submolt_display_name', '?')}")
        print(f"{'─' * 70}")
        print(f"[POST by {post.get('author_name', '?')}] "
              f"(↑{post['upvotes']} ↓{post['downvotes']} | "
              f"{post['comment_count']} comments)")
        title = post.get('title', '')
        content = post.get('content', '')
        if title:
            print(f"  Title: {str(title)[:200]}")
        if pd.notna(content) and content:
            print(f"  Content: {str(content)[:300]}")

        # Get comments for this post, ordered by created_at
        thread_comments = comments[comments['post_id'] == post_id].sort_values('created_at')
        for _, c in thread_comments.head(10).iterrows():
            indent = "  " * (c['depth'] + 1)
            author = c.get('author_name', '?')
            print(f"{indent}[depth={c['depth']}] {author} (↑{c['upvotes']}): "
                  f"{str(c['content'])[:200]}")

        if len(thread_comments) > 10:
            print(f"  ... ({len(thread_comments) - 10} more comments)")


def content_length_stats(posts, comments):
    """Analyze content lengths — important for metric design."""
    print("\n" + "=" * 70)
    print("CONTENT LENGTH ANALYSIS")
    print("=" * 70)

    # Post content lengths (in characters and words)
    post_content = posts['content'].dropna().astype(str)
    post_char_lens = post_content.str.len()
    post_word_lens = post_content.str.split().str.len()

    print(f"\n  Post content length (chars):")
    print(f"    mean={post_char_lens.mean():.0f}, "
          f"median={post_char_lens.median():.0f}, "
          f"p95={post_char_lens.quantile(0.95):.0f}, "
          f"max={post_char_lens.max():.0f}")
    print(f"  Post content length (words):")
    print(f"    mean={post_word_lens.mean():.0f}, "
          f"median={post_word_lens.median():.0f}, "
          f"p95={post_word_lens.quantile(0.95):.0f}, "
          f"max={post_word_lens.max():.0f}")

    # Comment content lengths
    comment_content = comments['content'].dropna().astype(str)
    comment_char_lens = comment_content.str.len()
    comment_word_lens = comment_content.str.split().str.len()

    print(f"\n  Comment content length (chars):")
    print(f"    mean={comment_char_lens.mean():.0f}, "
          f"median={comment_char_lens.median():.0f}, "
          f"p95={comment_char_lens.quantile(0.95):.0f}, "
          f"max={comment_char_lens.max():.0f}")
    print(f"  Comment content length (words):")
    print(f"    mean={comment_word_lens.mean():.0f}, "
          f"median={comment_word_lens.median():.0f}, "
          f"p95={comment_word_lens.quantile(0.95):.0f}, "
          f"max={comment_word_lens.max():.0f}")

    # Empty/very short content
    very_short_comments = (comment_word_lens <= 5).sum()
    empty_comments = (comments['content'].isna() | (comments['content'] == '')).sum()
    print(f"\n  Empty comments: {empty_comments:,}")
    print(f"  Very short comments (<=5 words): {very_short_comments:,} "
          f"({very_short_comments/len(comments)*100:.1f}%)")


def author_activity_stats(posts, comments):
    """Author-level activity distribution."""
    print("\n" + "=" * 70)
    print("AUTHOR ACTIVITY")
    print("=" * 70)

    # Posts per author
    posts_per_author = posts.groupby('author_id').size()
    print(f"\n  Posts per author:")
    print(f"    mean={posts_per_author.mean():.1f}, "
          f"median={posts_per_author.median():.0f}, "
          f"p95={posts_per_author.quantile(0.95):.0f}, "
          f"max={posts_per_author.max()}")

    # Comments per author
    comments_per_author = comments.groupby('author_id').size()
    print(f"  Comments per author:")
    print(f"    mean={comments_per_author.mean():.1f}, "
          f"median={comments_per_author.median():.0f}, "
          f"p95={comments_per_author.quantile(0.95):.0f}, "
          f"max={comments_per_author.max()}")

    # Karma distribution
    karma = comments.drop_duplicates('author_id')['author_karma'].dropna()
    print(f"\n  Author karma:")
    print(f"    mean={karma.mean():.0f}, "
          f"median={karma.median():.0f}, "
          f"p5={karma.quantile(0.05):.0f}, "
          f"p95={karma.quantile(0.95):.0f}")


if __name__ == "__main__":
    posts, comments = load_moltbook()

    basic_stats(posts, comments)
    content_length_stats(posts, comments)
    author_activity_stats(posts, comments)
    sample_conversations(posts, comments, n=5)

    print("\n\nBuilding thread trees (this may take a few minutes)...")
    thread_df, dyadic_threads = build_threads(posts, comments)

    # Save thread stats for next steps
    thread_df.to_parquet(f"{CACHE_DIR}/thread_stats.parquet")
    with open(f"{CACHE_DIR}/dyadic_threads.json", 'w') as f:
        json.dump(dyadic_threads, f)

    print(f"\n\nSaved thread_stats.parquet and dyadic_threads.json to {CACHE_DIR}/")
    print("Done.")
