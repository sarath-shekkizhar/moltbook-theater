"""
Quick check: compare thread depth across different Moltbook datasets.
Is the flat structure (93.6% depth-0) an artifact of AIcell's dataset,
or is it the platform reality?
"""

import pandas as pd

print("=" * 70)
print("DATASET 1: lnajt/moltbook (3.51M rows, largest)")
print("=" * 70)

# lnajt stores posts and comments as separate parquet files
# within the same dataset
try:
    comments_l = pd.read_parquet(
        "hf://datasets/lnajt/moltbook/comments.parquet"
    )
    posts_l = pd.read_parquet(
        "hf://datasets/lnajt/moltbook/posts.parquet"
    )
    print(f"Posts:    {len(posts_l):,}")
    print(f"Comments: {len(comments_l):,}")
    print(f"\nComment columns: {list(comments_l.columns)}")
    print(f"Post columns:    {list(posts_l.columns)}")

    # Check parent_id distribution
    top_level = comments_l['parent_id'].isna().sum()
    nested = comments_l['parent_id'].notna().sum()
    print(f"\nTop-level comments (parent_id=null): {top_level:,} "
          f"({top_level/len(comments_l)*100:.1f}%)")
    print(f"Nested comments (parent_id!=null):   {nested:,} "
          f"({nested/len(comments_l)*100:.1f}%)")

    # Date range
    print(f"\nComment date range: {comments_l['created_at'].min()} → "
          f"{comments_l['created_at'].max()}")
    print(f"Post date range:    {posts_l['created_at'].min()} → "
          f"{posts_l['created_at'].max()}")

    # Sample a nested comment chain
    nested_comments = comments_l[comments_l['parent_id'].notna()]
    if len(nested_comments) > 0:
        print(f"\nSample nested comment:")
        sample = nested_comments.iloc[0]
        print(f"  id: {sample['id'][:20]}...")
        print(f"  parent_id: {sample['parent_id'][:20]}...")
        print(f"  post_id: {sample['post_id'][:20]}...")
        print(f"  body: {str(sample.get('body', ''))[:200]}")

except Exception as e:
    print(f"Error loading lnajt/moltbook: {e}")

print("\n" + "=" * 70)
print("DATASET 2: SimulaMet/moltbook-observatory-archive")
print("=" * 70)

try:
    from datasets import load_dataset

    comments_s = load_dataset(
        "SimulaMet/moltbook-observatory-archive", "comments",
        split="archive"
    ).to_pandas()
    agents_s = load_dataset(
        "SimulaMet/moltbook-observatory-archive", "agents",
        split="archive"
    ).to_pandas()

    print(f"Comments: {len(comments_s):,}")
    print(f"Agents:   {len(agents_s):,}")
    print(f"\nComment columns: {list(comments_s.columns)}")
    print(f"Agent columns:   {list(agents_s.columns)}")

    # Check parent_id distribution
    top_level = comments_s['parent_id'].isna().sum()
    nested = comments_s['parent_id'].notna().sum()
    print(f"\nTop-level comments (parent_id=null): {top_level:,} "
          f"({top_level/len(comments_s)*100:.1f}%)")
    print(f"Nested comments (parent_id!=null):   {nested:,} "
          f"({nested/len(comments_s)*100:.1f}%)")

    print(f"\nComment date range: {comments_s['created_at'].min()} → "
          f"{comments_s['created_at'].max()}")

    # Agent descriptions (useful for understanding agent personas)
    has_desc = agents_s['description'].notna().sum()
    print(f"\nAgents with descriptions: {has_desc:,} / {len(agents_s):,}")
    if has_desc > 0:
        sample_desc = agents_s[agents_s['description'].notna()].iloc[0]
        print(f"Sample agent: {sample_desc['name']}")
        print(f"  Description: {str(sample_desc['description'])[:300]}")

except Exception as e:
    print(f"Error loading SimulaMet: {e}")

print("\n" + "=" * 70)
print("COMPARISON WITH AIcell/moltbook-data")
print("=" * 70)

try:
    comments_a = pd.read_parquet("cache/comments.parquet")
    print(f"AIcell comments: {len(comments_a):,}")
    top_a = (comments_a['depth'] == 0).sum()
    nested_a = (comments_a['depth'] > 0).sum()
    print(f"  Top-level (depth=0): {top_a:,} ({top_a/len(comments_a)*100:.1f}%)")
    print(f"  Nested (depth>0):    {nested_a:,} ({nested_a/len(comments_a)*100:.1f}%)")
except Exception as e:
    print(f"AIcell cache not available: {e}")
