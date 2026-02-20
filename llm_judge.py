"""
LLM-as-Judge validation for Moltbook comment quality.

Uses litellm for provider-agnostic LLM calls and Pydantic for structured output.
Judges each (post, comment) pair on responsiveness, information contribution,
and category.

Primary model: bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0 (fast)
Calibration model: bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0 (200-pair subset for inter-rater reliability)
"""

import json
import os
import time

from dotenv import load_dotenv
load_dotenv()

import litellm
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

CACHE = "cache"

JUDGE_PROMPT = """You are evaluating a comment on a social media post made by an AI agent.

POST:
{post}

COMMENT:
{comment}

Rate the comment on two dimensions:

**Responsiveness (1-5):**
1 = Completely generic. Could be a response to any post.
2 = Loosely related to the general topic but doesn't engage with specific content.
3 = References the topic and 1-2 specific points from the post.
4 = Directly addresses the main points and adds relevant content.
5 = Directly engages with specific claims/questions in the post, building on them.

**Information (1-5):**
1 = No new information. Generic filler ("Great post!", "I agree!").
2 = Restates what the post already said in different words.
3 = Adds 1-2 new points tangentially related to the post.
4 = Adds substantive new information or a distinct perspective.
5 = Adds multiple new claims, evidence, or analysis.

**Category:** Classify as one of:
- generic_affirmation: Vague praise or agreement without substance
- self_promotion: Advertising the agent's own services/links
- spam: Repeated/template content, scams, manipulation
- on_topic: Related to the post topic but not deeply engaged
- substantive: Meaningful engagement with the post's specific content
- off_topic: Unrelated to the post

Respond in JSON format with fields: reasoning, responsiveness, information, category."""


class CommentJudgment(BaseModel):
    reasoning: str = Field(description="Brief reasoning for the judgment")
    responsiveness: int = Field(
        description="1-5: how specifically does the comment address the post content"
    )
    information: int = Field(
        description="1-5: how much new information does the comment add"
    )
    category: str = Field(
        description="One of: generic_affirmation, self_promotion, spam, on_topic, substantive, off_topic"
    )


VALID_CATEGORIES = {
    "generic_affirmation",
    "self_promotion",
    "spam",
    "on_topic",
    "substantive",
    "off_topic",
}


def _parse_judgment(raw) -> CommentJudgment:
    """Parse LLM response into CommentJudgment, handling various response formats."""
    if isinstance(raw, CommentJudgment):
        return raw

    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        # Try to extract JSON from the response
        text = raw.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
    else:
        raise ValueError(f"Unexpected response type: {type(raw)}")

    # Normalize category
    category = str(data.get("category", "off_topic")).strip().lower()
    if category not in VALID_CATEGORIES:
        # Try fuzzy matching
        for valid in VALID_CATEGORIES:
            if valid in category or category in valid:
                category = valid
                break
        else:
            category = "off_topic"

    # Clamp scores to valid range
    responsiveness = max(1, min(5, int(data.get("responsiveness", 1))))
    information = max(1, min(5, int(data.get("information", 1))))

    return CommentJudgment(
        reasoning=str(data.get("reasoning", "")),
        responsiveness=responsiveness,
        information=information,
        category=category,
    )


def judge_pair(
    post: str,
    comment: str,
    model: str = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
) -> CommentJudgment:
    """Judge a single (post, comment) pair using an LLM.

    Args:
        post: The post text.
        comment: The comment text.
        model: litellm model identifier.

    Returns:
        CommentJudgment with scores and category.
    """
    prompt = JUDGE_PROMPT.format(
        post=post[:2000],  # Truncate very long texts
        comment=comment[:2000],
    )

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content
    return _parse_judgment(raw_content)


def run_judge_batch(
    pairs: pd.DataFrame,
    model: str = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    cache_path: str | None = None,
    delay: float = 0.1,
) -> pd.DataFrame:
    """Run LLM judge on a batch of (post, comment) pairs.

    Args:
        pairs: DataFrame with 'post_text' and 'comment_text' columns.
        model: litellm model identifier.
        cache_path: If provided, save/resume from this parquet file.
        delay: Seconds between API calls to avoid rate limits.

    Returns:
        DataFrame with columns: responsiveness, information, category, reasoning.
    """
    # Load existing cache if available
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        start_idx = len(cached)
        print(f"  Resuming from cached {start_idx} judgments at {cache_path}")
        results = cached.to_dict("records")
    else:
        start_idx = 0
        results = []

    t0 = time.time()
    n_errors = 0

    for i in range(start_idx, len(pairs)):
        row = pairs.iloc[i]
        post = str(row.get("post_text", "") or "")
        comment = str(row.get("comment_text", row.get("content", "")) or "")

        try:
            judgment = judge_pair(post, comment, model=model)
            results.append(
                {
                    "responsiveness": judgment.responsiveness,
                    "information": judgment.information,
                    "category": judgment.category,
                    "reasoning": judgment.reasoning,
                }
            )
        except Exception as e:
            n_errors += 1
            results.append(
                {
                    "responsiveness": np.nan,
                    "information": np.nan,
                    "category": "error",
                    "reasoning": str(e),
                }
            )

        # Progress and periodic caching
        n_done = i + 1
        if n_done % 100 == 0 or n_done == len(pairs):
            elapsed = time.time() - t0
            rate = (n_done - start_idx) / elapsed if elapsed > 0 else 0
            print(
                f"    Judged {n_done:,}/{len(pairs):,} "
                f"({rate:.1f} pairs/sec, {n_errors} errors)"
            )
            # Save checkpoint
            if cache_path:
                pd.DataFrame(results).to_parquet(cache_path, index=False)

        if delay > 0:
            time.sleep(delay)

    df = pd.DataFrame(results)

    # Final save
    if cache_path:
        df.to_parquet(cache_path, index=False)
        print(f"  Saved {len(df)} judgments to {cache_path}")

    return df


def sample_for_judging(
    relevance_df: pd.DataFrame,
    n_high: int = 500,
    n_zero: int = 1000,
    n_negative: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample (post, comment) pairs stratified by Jaccard specificity.

    Args:
        relevance_df: DataFrame with 'specificity_lexical' from analyze_relevance.
        n_high: Number of high-specificity pairs (>0.02).
        n_zero: Number of zero-specificity pairs (~0).
        n_negative: Number of negative-specificity pairs (<-0.005).
        seed: Random seed.

    Returns:
        DataFrame with 'post_text', 'comment_text', 'specificity_lexical', 'stratum'.
    """
    rng = np.random.RandomState(seed)

    high = relevance_df[relevance_df["specificity_lexical"] > 0.02]
    zero = relevance_df[
        (relevance_df["specificity_lexical"] >= -0.005)
        & (relevance_df["specificity_lexical"] <= 0.02)
    ]
    negative = relevance_df[relevance_df["specificity_lexical"] < -0.005]

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
            s["stratum"] = label
            samples.append(s)

    sampled = pd.concat(samples, ignore_index=True)

    # Attach original texts — this requires the index alignment between
    # relevance_df and comments_df to be preserved. We stored these in
    # the relevance analysis, so we need post_text and content columns.
    # The caller is responsible for ensuring these columns exist.

    return sampled


def compute_inter_rater_reliability(
    primary_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> dict:
    """Compute agreement between primary and calibration judge models.

    Args:
        primary_df: Judgments from the primary (cheap) model.
        calibration_df: Judgments from the calibration (expensive) model.

    Returns:
        dict with Cohen's kappa for category and correlation for scores.
    """
    # Align on common indices
    n = min(len(primary_df), len(calibration_df))
    p = primary_df.iloc[:n]
    c = calibration_df.iloc[:n]

    # Drop error rows
    mask = (p["category"] != "error") & (c["category"] != "error")
    p = p[mask]
    c = c[mask]

    results = {}

    # Category agreement (Cohen's kappa)
    from sklearn.metrics import cohen_kappa_score

    try:
        results["category_kappa"] = cohen_kappa_score(
            p["category"].values, c["category"].values
        )
        results["category_agreement"] = (
            p["category"].values == c["category"].values
        ).mean()
    except Exception:
        results["category_kappa"] = np.nan
        results["category_agreement"] = np.nan

    # Score correlations
    for col in ["responsiveness", "information"]:
        p_vals = pd.to_numeric(p[col], errors="coerce")
        c_vals = pd.to_numeric(c[col], errors="coerce")
        mask_valid = p_vals.notna() & c_vals.notna()
        if mask_valid.sum() > 2:
            results[f"{col}_correlation"] = float(
                np.corrcoef(p_vals[mask_valid], c_vals[mask_valid])[0, 1]
            )
            results[f"{col}_mean_diff"] = float(
                (p_vals[mask_valid] - c_vals[mask_valid]).mean()
            )
        else:
            results[f"{col}_correlation"] = np.nan
            results[f"{col}_mean_diff"] = np.nan

    return results
