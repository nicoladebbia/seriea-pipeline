"""Head-to-head features computed from all prior meetings between two teams."""

from __future__ import annotations

import math

import pandas as pd


def add_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head features to the matches table.

    For each match, aggregates all prior meetings between home_team and away_team:
      h2h_matches_played, h2h_home_wins, h2h_away_wins, h2h_draws,
      h2h_home_goals_avg, h2h_away_goals_avg, h2h_home_win_rate,
      h2h_last_result (1 = home win, 0 = draw, -1 = away win)
    """
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)

    h2h_cols = {
        "h2h_matches_played": [],
        "h2h_home_wins": [],
        "h2h_away_wins": [],
        "h2h_draws": [],
        "h2h_draw_rate": [],
        "h2h_goals_avg": [],
        "h2h_home_goals_avg": [],
        "h2h_away_goals_avg": [],
        "h2h_home_win_rate": [],
        "h2h_last_result": [],
        # Recency-weighted stats (exponential decay: weight = exp(-0.3 * years_ago))
        "h2h_weighted_home_win_rate": [],
        # Recent-5 meetings only
        "h2h_recent_5_win_rate": [],
    }

    # Decay constant for recency weighting
    DECAY_RATE = 0.3  # exp(-0.3 * years) → last year=0.74, 3yr=0.41, 5yr=0.22

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["match_date"]

        # All prior meetings (either direction)
        prior = df[
            (df["match_date"] < date)
            & (
                ((df["home_team"] == home) & (df["away_team"] == away))
                | ((df["home_team"] == away) & (df["away_team"] == home))
            )
        ]

        n = len(prior)
        h2h_cols["h2h_matches_played"].append(n)

        if n == 0:
            for key in list(h2h_cols.keys())[1:]:
                h2h_cols[key].append(None)
            continue

        # Normalize: from the perspective of current home vs current away
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals_total = 0
        away_goals_total = 0
        last_result = None

        # For recency weighting
        weighted_home_wins = 0.0
        weighted_total = 0.0

        # For recent-5: collect results in chronological order
        recent_results = []

        for _, prev in prior.iterrows():
            hs = prev.get("home_score")
            as_ = prev.get("away_score")
            if pd.isna(hs) or pd.isna(as_):
                continue

            hs, as_ = int(hs), int(as_)

            # Compute recency weight
            years_ago = (date - prev["match_date"]).days / 365.25
            weight = math.exp(-DECAY_RATE * years_ago)

            if prev["home_team"] == home:
                # Same direction
                home_goals_total += hs
                away_goals_total += as_
                if hs > as_:
                    home_wins += 1
                    last_result = 1
                    weighted_home_wins += weight
                    recent_results.append(1)
                elif hs < as_:
                    away_wins += 1
                    last_result = -1
                    recent_results.append(0)
                else:
                    draws += 1
                    last_result = 0
                    recent_results.append(0)
            else:
                # Reversed direction
                home_goals_total += as_
                away_goals_total += hs
                if as_ > hs:
                    home_wins += 1
                    last_result = 1
                    weighted_home_wins += weight
                    recent_results.append(1)
                elif as_ < hs:
                    away_wins += 1
                    last_result = -1
                    recent_results.append(0)
                else:
                    draws += 1
                    last_result = 0
                    recent_results.append(0)

            weighted_total += weight

        played = home_wins + away_wins + draws
        h2h_cols["h2h_home_wins"].append(home_wins)
        h2h_cols["h2h_away_wins"].append(away_wins)
        h2h_cols["h2h_draws"].append(draws)
        h2h_cols["h2h_draw_rate"].append(round(draws / played, 4) if played > 0 else None)
        h2h_home_avg = home_goals_total / played if played > 0 else None
        h2h_away_avg = away_goals_total / played if played > 0 else None
        h2h_cols["h2h_goals_avg"].append(
            round(h2h_home_avg + h2h_away_avg, 4) if h2h_home_avg is not None else None
        )
        h2h_cols["h2h_home_goals_avg"].append(h2h_home_avg)
        h2h_cols["h2h_away_goals_avg"].append(h2h_away_avg)
        h2h_cols["h2h_home_win_rate"].append(home_wins / played if played > 0 else None)
        h2h_cols["h2h_last_result"].append(last_result)

        # Recency-weighted win rate
        h2h_cols["h2h_weighted_home_win_rate"].append(
            round(weighted_home_wins / weighted_total, 4) if weighted_total > 0 else None
        )

        # Recent-5 win rate (last 5 meetings only)
        last_5 = recent_results[-5:]
        h2h_cols["h2h_recent_5_win_rate"].append(
            round(sum(last_5) / len(last_5), 4) if last_5 else None
        )

    for key, values in h2h_cols.items():
        df[key] = values

    return df
