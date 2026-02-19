"""
Combine all three Moltbook HuggingFace datasets into a single
deduplicated corpus with unified schema.

Sources:
  1. lnajt/moltbook        — 668K posts, 2.84M comments (largest)
  2. AIcell/moltbook-data   — 290K posts, 1.84M comments (has depth field)
  3. SimulaMet/...archive   — 214K posts, 882K comments + 78K agent profiles

Strategy:
  - Start with lnajt (largest base)
  - Union in rows from AIcell and SimulaMet whose IDs are NOT already present
  - Carry forward the depth field from AIcell where available
  - Merge agent descriptions from SimulaMet onto the combined comment set
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import os, time

CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)

# ── Load all three sources ────────────────────────────────────────────

def load_lnajt():
    p = f"{CACHE}/lnajt_posts.parquet"
    c = f"{CACHE}/lnajt_comments.parquet"
    if os.path.exists(p) and os.path.exists(c):
        return pd.read_parquet(p), pd.read_parquet(c)
    posts = pd.read_parquet("hf://datasets/lnajt/moltbook/posts.parquet")
    comments = pd.read_parquet("hf://datasets/lnajt/moltbook/comments.parquet")
    posts.to_parquet(p); comments.to_parquet(c)
    return posts, comments


def load_aicell():
    p = f"{CACHE}/posts.parquet"
    c = f"{CACHE}/comments.parquet"
    if os.path.exists(p) and os.path.exists(c):
        return pd.read_parquet(p), pd.read_parquet(c)
    ds_p = load_dataset("AIcell/moltbook-data", "posts", split="train")
    ds_c = load_dataset("AIcell/moltbook-data", "comments", split="train")
    posts = ds_p.to_pandas(); comments = ds_c.to_pandas()
    posts.to_parquet(p); comments.to_parquet(c)
    return posts, comments


def load_simulamet():
    prefix = f"{CACHE}/simulamet"
    cp = f"{prefix}_comments.parquet"
    pp = f"{prefix}_posts.parquet"
    ap = f"{prefix}_agents.parquet"
    if os.path.exists(cp) and os.path.exists(pp) and os.path.exists(ap):
        return (pd.read_parquet(pp), pd.read_parquet(cp),
                pd.read_parquet(ap))
    posts = load_dataset("SimulaMet/moltbook-observatory-archive",
                         "posts", split="archive").to_pandas()
    comments = load_dataset("SimulaMet/moltbook-observatory-archive",
                            "comments", split="archive").to_pandas()
    agents = load_dataset("SimulaMet/moltbook-observatory-archive",
                          "agents", split="archive").to_pandas()
    posts.to_parquet(pp); comments.to_parquet(cp); agents.to_parquet(ap)
    return posts, comments, agents


# ── Normalize schemas ─────────────────────────────────────────────────

def normalize_comments(df, source):
    """Map each dataset's comment columns to a unified schema."""
    out = pd.DataFrame()
    out['id'] = df['id']
    out['post_id'] = df['post_id']
    out['parent_id'] = df.get('parent_id')
    out['created_at'] = df['created_at']

    # Content field
    if 'body' in df.columns:
        out['content'] = df['body']
    elif 'content' in df.columns:
        out['content'] = df['content']

    # Author
    if 'author_id' in df.columns:
        out['author_id'] = df['author_id']
    elif 'agent_id' in df.columns:
        out['author_id'] = df['agent_id']

    if 'author' in df.columns:
        out['author_name'] = df['author']
    elif 'author_name' in df.columns:
        out['author_name'] = df['author_name']
    elif 'agent_name' in df.columns:
        out['author_name'] = df['agent_name']

    # Votes
    if 'upvotes' in df.columns:
        out['upvotes'] = df['upvotes']
        out['downvotes'] = df.get('downvotes', 0)
    elif 'score' in df.columns:
        out['upvotes'] = df['score']
        out['downvotes'] = 0

    # Depth (only AIcell has this natively)
    if 'depth' in df.columns:
        out['depth'] = df['depth']

    out['source'] = source
    return out


def normalize_posts(df, source):
    """Map each dataset's post columns to a unified schema."""
    out = pd.DataFrame()
    out['id'] = df['id']
    out['created_at'] = df['created_at']

    out['title'] = df.get('title')

    if 'body' in df.columns:
        out['body'] = df['body']
    elif 'content' in df.columns:
        out['body'] = df['content']

    if 'author_id' in df.columns:
        out['author_id'] = df['author_id']
    elif 'agent_id' in df.columns:
        out['author_id'] = df['agent_id']

    if 'author' in df.columns:
        out['author_name'] = df['author']
    elif 'author_name' in df.columns:
        out['author_name'] = df['author_name']
    elif 'agent_name' in df.columns:
        out['author_name'] = df['agent_name']

    if 'submolt' in df.columns:
        out['submolt'] = df['submolt']
    elif 'submolt_name' in df.columns:
        out['submolt'] = df['submolt_name']

    if 'comment_count' in df.columns:
        out['comment_count'] = df['comment_count']

    if 'upvotes' in df.columns:
        out['upvotes'] = df['upvotes']
    elif 'score' in df.columns:
        out['upvotes'] = df['score']

    out['source'] = source
    return out


# ── Combine & dedup ───────────────────────────────────────────────────

def combine():
    t0 = time.time()

    print("Loading lnajt/moltbook...")
    ln_posts, ln_comments = load_lnajt()
    print(f"  posts={len(ln_posts):,}  comments={len(ln_comments):,}")

    print("Loading AIcell/moltbook-data...")
    ai_posts, ai_comments = load_aicell()
    print(f"  posts={len(ai_posts):,}  comments={len(ai_comments):,}")

    print("Loading SimulaMet/moltbook-observatory-archive...")
    sm_posts, sm_comments, sm_agents = load_simulamet()
    print(f"  posts={len(sm_posts):,}  comments={len(sm_comments):,}  "
          f"agents={len(sm_agents):,}")

    # Normalize
    print("\nNormalizing schemas...")
    c1 = normalize_comments(ln_comments, 'lnajt')
    c2 = normalize_comments(ai_comments, 'aicell')
    c3 = normalize_comments(sm_comments, 'simulamet')

    p1 = normalize_posts(ln_posts, 'lnajt')
    p2 = normalize_posts(ai_posts, 'aicell')
    p3 = normalize_posts(sm_posts, 'simulamet')

    # Combine comments: start with lnajt, add new IDs from others
    print("\nCombining comments...")
    all_comments = c1.copy()
    existing_ids = set(all_comments['id'])
    print(f"  Base (lnajt): {len(all_comments):,}")

    new_from_aicell = c2[~c2['id'].isin(existing_ids)]
    all_comments = pd.concat([all_comments, new_from_aicell], ignore_index=True)
    existing_ids.update(new_from_aicell['id'])
    print(f"  + AIcell new IDs: {len(new_from_aicell):,} → total {len(all_comments):,}")

    new_from_sm = c3[~c3['id'].isin(existing_ids)]
    all_comments = pd.concat([all_comments, new_from_sm], ignore_index=True)
    print(f"  + SimulaMet new IDs: {len(new_from_sm):,} → total {len(all_comments):,}")

    # Compute depth from parent_id (vectorized, iterative BFS)
    print("  Computing depth from parent_id structure...")
    if 'depth' not in all_comments.columns:
        all_comments['depth'] = np.nan

    # Carry depth from AIcell where available
    depth_map = c2[['id', 'depth']].dropna(subset=['depth']).set_index('id')['depth']
    mask = all_comments['id'].isin(depth_map.index) & all_comments['depth'].isna()
    all_comments.loc[mask, 'depth'] = all_comments.loc[mask, 'id'].map(depth_map)

    # Top-level (no parent) = depth 0
    is_top_level = all_comments['parent_id'].isna()
    all_comments.loc[is_top_level & all_comments['depth'].isna(), 'depth'] = 0

    # Iterative vectorized BFS for remaining
    # Build a Series: id -> depth (using index for fast lookup)
    id_to_idx = pd.Series(all_comments.index, index=all_comments['id'])
    id_to_idx = id_to_idx[~id_to_idx.index.duplicated(keep='first')]
    parent_ids = all_comments['parent_id']
    depths = all_comments['depth'].copy()

    for iteration in range(35):
        unknown = depths.isna()
        if not unknown.any():
            break
        # For unknown-depth comments, look up parent's depth
        parent_of_unknown = parent_ids[unknown]
        # Map parent_id -> index in our dataframe
        parent_idx = parent_of_unknown.map(id_to_idx)
        # Get parent depths (only where parent exists in our data)
        valid = parent_idx.notna()
        parent_idx_valid = parent_idx[valid].astype(int)
        parent_depths = depths.iloc[parent_idx_valid.values]
        # Where parent depth is known, set child = parent + 1
        resolved = parent_depths.notna()
        if not resolved.any():
            break
        # Map back to the unknown indices
        unknown_indices = unknown[unknown].index
        valid_indices = valid[valid].index
        resolved_indices = resolved[resolved].index
        # The actual indices in all_comments to update
        update_idx = valid_indices[resolved.values]
        update_vals = parent_depths[resolved].values + 1
        depths.iloc[depths.index.get_indexer(update_idx)] = update_vals
        print(f"    Iteration {iteration}: resolved {resolved.sum():,} depths")

    all_comments['depth'] = depths
    has_depth = all_comments['depth'].notna().sum()
    print(f"  Depth resolved: {has_depth:,} / {len(all_comments):,}")

    # Combine posts
    print("\nCombining posts...")
    all_posts = p1.copy()
    existing_pids = set(all_posts['id'])
    print(f"  Base (lnajt): {len(all_posts):,}")

    new_p_ai = p2[~p2['id'].isin(existing_pids)]
    all_posts = pd.concat([all_posts, new_p_ai], ignore_index=True)
    existing_pids.update(new_p_ai['id'])
    print(f"  + AIcell new IDs: {len(new_p_ai):,} → total {len(all_posts):,}")

    new_p_sm = p3[~p3['id'].isin(existing_pids)]
    all_posts = pd.concat([all_posts, new_p_sm], ignore_index=True)
    print(f"  + SimulaMet new IDs: {len(new_p_sm):,} → total {len(all_posts):,}")

    # Merge agent descriptions from SimulaMet
    print("\nMerging agent descriptions from SimulaMet...")
    # Dedup agents by ID (take first occurrence)
    agents = sm_agents.drop_duplicates('id', keep='first')
    agent_desc = agents[['id', 'name', 'description', 'karma',
                          'follower_count', 'following_count',
                          'is_claimed']].rename(
        columns={'id': 'agent_id', 'name': 'agent_name',
                 'description': 'agent_description',
                 'karma': 'agent_karma',
                 'follower_count': 'agent_followers',
                 'following_count': 'agent_following',
                 'is_claimed': 'agent_is_claimed'}
    )

    # Match on author_id
    all_comments = all_comments.merge(
        agent_desc, left_on='author_id', right_on='agent_id', how='left'
    )
    matched = all_comments['agent_description'].notna().sum()
    print(f"  Comments with agent description: {matched:,} / {len(all_comments):,}")

    # Save
    print("\nSaving combined dataset...")
    all_posts.to_parquet(f"{CACHE}/combined_posts.parquet", index=False)
    all_comments.to_parquet(f"{CACHE}/combined_comments.parquet", index=False)
    agent_desc.to_parquet(f"{CACHE}/agents.parquet", index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    # Final summary
    print("\n" + "=" * 70)
    print("COMBINED DATASET SUMMARY")
    print("=" * 70)
    print(f"  Posts:                 {len(all_posts):,}")
    print(f"  Comments:              {len(all_comments):,}")
    print(f"  Agent profiles:        {len(agent_desc):,}")

    top = (all_comments['depth'] == 0).sum()
    nested = (all_comments['depth'] > 0).sum()
    unknown = all_comments['depth'].isna().sum()
    print(f"\n  Top-level (depth=0):   {top:,} ({top/len(all_comments)*100:.1f}%)")
    print(f"  Nested (depth>0):      {nested:,} ({nested/len(all_comments)*100:.1f}%)")
    print(f"  Depth unknown:         {unknown:,}")

    print(f"\n  Date range (comments): "
          f"{all_comments['created_at'].min()} → {all_comments['created_at'].max()}")
    print(f"  Date range (posts):    "
          f"{all_posts['created_at'].min()} → {all_posts['created_at'].max()}")

    # Depth distribution
    print(f"\n  Depth distribution:")
    depth_counts = all_comments['depth'].dropna().astype(int).value_counts().sort_index()
    for d in range(min(10, depth_counts.index.max() + 1)):
        if d in depth_counts.index:
            print(f"    depth={d}: {depth_counts[d]:>12,}")
    if depth_counts.index.max() > 9:
        deep = depth_counts[depth_counts.index > 9].sum()
        print(f"    depth>9: {deep:>12,}")

    # Source breakdown
    print(f"\n  Comment sources:")
    for src, cnt in all_comments['source'].value_counts().items():
        print(f"    {src}: {cnt:,}")

    return all_posts, all_comments, agent_desc


if __name__ == "__main__":
    combine()
