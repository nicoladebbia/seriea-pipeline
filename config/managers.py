"""
Module: config/managers.py
Purpose: Historical Serie A manager tenure data for manager-change features and form discontinuity detection
Inputs:  None (static data from Wikipedia and Transfermarkt)
Outputs: MANAGER_TENURES list of tuples (team, manager, start_date, end_date)
Called by: features/manager_changes.py (manager change detection), features/build.py (feature engineering)
Depends on: None (pure data module)
"""

from __future__ import annotations

import pandas as pd

# (team, manager, start_date_inclusive, end_date_inclusive_or_None)
# Dates are approximate to the week of appointment/dismissal.
MANAGER_TENURES: list[tuple[str, str, str, str | None]] = [
    # ── Atalanta ──
    ("Atalanta", "Gian Piero Gasperini", "2016-06-14", "2025-06-30"),
    ("Atalanta", "Ivan Juric", "2025-07-01", "2025-11-10"),
    ("Atalanta", "Raffaele Palladino", "2025-11-11", None),

    # ── Benevento ──
    ("Benevento", "Marco Baroni", "2017-06-01", "2018-05-20"),
    ("Benevento", "Filippo Inzaghi", "2020-06-01", "2021-05-23"),

    # ── Bologna ──
    ("Bologna", "Roberto Donadoni", "2015-06-11", "2018-01-17"),
    ("Bologna", "Filippo Inzaghi", "2018-01-17", "2019-06-30"),
    ("Bologna", "Sinisa Mihajlovic", "2019-01-28", "2022-09-06"),
    ("Bologna", "Thiago Motta", "2022-09-07", "2024-06-30"),
    ("Bologna", "Vincenzo Italiano", "2024-06-01", None),  # Continuing 2025-2026

    # ── Brescia ──
    ("Brescia", "Eugenio Corini", "2019-06-01", "2019-11-03"),
    ("Brescia", "Fabio Grosso", "2019-11-04", "2019-12-02"),
    ("Brescia", "Eugenio Corini", "2019-12-03", "2020-02-22"),
    ("Brescia", "Diego Lopez", "2020-02-23", "2020-08-02"),

    # ── Cagliari ──
    ("Cagliari", "Diego Lopez", "2017-03-13", "2018-03-18"),
    ("Cagliari", "Rolando Maran", "2018-03-19", "2020-03-03"),
    ("Cagliari", "Walter Zenga", "2020-03-04", "2020-08-02"),
    ("Cagliari", "Eusebio Di Francesco", "2020-08-03", "2021-02-22"),
    ("Cagliari", "Leonardo Semplici", "2021-02-23", "2021-10-25"),
    ("Cagliari", "Walter Mazzarri", "2021-10-26", "2022-05-22"),
    ("Cagliari", "Claudio Ranieri", "2023-06-01", "2024-05-26"),
    ("Cagliari", "Davide Nicola", "2024-06-01", "2025-06-10"),
    ("Cagliari", "Fabio Pisacane", "2025-06-11", None),

    # ── Chievo ──
    ("Chievo", "Rolando Maran", "2017-06-01", "2018-05-20"),
    ("Chievo", "Lorenzo D'Anna", "2018-06-01", "2018-11-26"),
    ("Chievo", "Mimmo Di Carlo", "2018-11-27", "2019-05-26"),

    # ── Como ──
    ("Como", "Cesc Fabregas", "2024-06-01", None),  # Continuing 2025-2026

    # ── Cremonese ── (promoted 2025-2026)
    ("Cremonese", "Davide Nicola", "2025-06-01", None),

    # ── Cremonese ──
    ("Cremonese", "Massimiliano Alvini", "2022-06-01", "2023-02-08"),
    ("Cremonese", "Davide Ballardini", "2023-02-09", "2023-05-28"),

    # ── Crotone ──
    ("Crotone", "Davide Nicola", "2017-06-01", "2018-05-20"),
    ("Crotone", "Giovanni Stroppa", "2020-06-01", "2021-03-02"),
    ("Crotone", "Serse Cosmi", "2021-03-03", "2021-05-23"),

    # ── Empoli ──
    ("Empoli", "Aurelio Andreazzoli", "2018-06-01", "2019-05-26"),
    ("Empoli", "Aurelio Andreazzoli", "2021-06-01", "2022-01-03"),
    ("Empoli", "Paolo Zanetti", "2022-06-01", "2023-06-30"),
    ("Empoli", "Davide Nicola", "2023-06-01", "2024-06-30"),
    ("Empoli", "Roberto D'Aversa", "2024-06-01", "2025-05-25"),  # Relegated

    # ── Fiorentina ──
    ("Fiorentina", "Stefano Pioli", "2017-06-01", "2019-04-09"),
    ("Fiorentina", "Vincenzo Montella", "2019-04-10", "2019-12-21"),
    ("Fiorentina", "Giuseppe Iachini", "2019-12-22", "2020-11-09"),
    ("Fiorentina", "Cesare Prandelli", "2020-11-10", "2021-03-23"),
    ("Fiorentina", "Giuseppe Iachini", "2021-03-24", "2021-06-30"),
    ("Fiorentina", "Vincenzo Italiano", "2021-06-22", "2024-06-30"),
    ("Fiorentina", "Raffaele Palladino", "2024-06-01", "2025-06-30"),
    ("Fiorentina", "Stefano Pioli", "2025-07-01", "2025-11-04"),
    ("Fiorentina", "Paolo Vanoli", "2025-11-07", None),

    # ── Frosinone ──
    ("Frosinone", "Moreno Longo", "2018-06-01", "2019-04-14"),
    ("Frosinone", "Marco Baroni", "2019-04-15", "2019-05-26"),
    ("Frosinone", "Eusebio Di Francesco", "2023-06-01", "2024-05-26"),

    # ── Genoa ──
    ("Genoa", "Ivan Juric", "2017-06-01", "2017-10-07"),
    ("Genoa", "Davide Ballardini", "2017-11-07", "2018-06-30"),
    ("Genoa", "Cesare Prandelli", "2018-06-20", "2018-12-27"),
    ("Genoa", "Davide Nicola", "2018-12-28", "2019-06-30"),
    ("Genoa", "Aurelio Andreazzoli", "2019-06-01", "2019-10-06"),
    ("Genoa", "Thiago Motta", "2019-10-07", "2019-12-22"),
    ("Genoa", "Davide Nicola", "2019-12-23", "2020-08-02"),
    ("Genoa", "Rolando Maran", "2020-08-03", "2020-12-22"),
    ("Genoa", "Davide Ballardini", "2020-12-23", "2021-11-07"),
    ("Genoa", "Andriy Shevchenko", "2021-11-08", "2022-01-15"),
    ("Genoa", "Alexander Blessin", "2022-01-16", "2022-05-22"),
    ("Genoa", "Alberto Gilardino", "2023-06-01", "2024-11-20"),
    ("Genoa", "Patrick Vieira", "2024-11-21", "2025-10-31"),
    ("Genoa", "Daniele De Rossi", "2025-11-06", None),

    # ── Inter ──
    ("Inter", "Luciano Spalletti", "2017-06-09", "2019-05-30"),
    ("Inter", "Antonio Conte", "2019-05-31", "2021-05-26"),
    ("Inter", "Simone Inzaghi", "2021-06-03", "2025-06-30"),
    ("Inter", "Cristian Chivu", "2025-07-01", None),

    # ── Juventus ──
    ("Juventus", "Massimiliano Allegri", "2017-06-01", "2019-05-26"),
    ("Juventus", "Maurizio Sarri", "2019-06-16", "2020-08-08"),
    ("Juventus", "Andrea Pirlo", "2020-08-09", "2021-05-23"),
    ("Juventus", "Massimiliano Allegri", "2021-05-28", "2024-05-17"),
    ("Juventus", "Thiago Motta", "2024-06-01", "2025-06-30"),
    ("Juventus", "Igor Tudor", "2025-07-01", "2025-10-28"),
    ("Juventus", "Luciano Spalletti", "2025-10-30", None),

    # ── Lazio ──
    ("Lazio", "Simone Inzaghi", "2016-04-03", "2021-05-23"),
    ("Lazio", "Maurizio Sarri", "2021-06-09", "2023-03-14"),
    ("Lazio", "Igor Tudor", "2023-03-15", "2023-06-30"),
    ("Lazio", "Maurizio Sarri", "2023-06-01", "2024-03-12"),
    ("Lazio", "Igor Tudor", "2024-03-13", "2024-06-30"),
    ("Lazio", "Marco Baroni", "2024-06-01", "2025-06-30"),
    ("Lazio", "Maurizio Sarri", "2025-06-02", None),

    # ── Lecce ──
    ("Lecce", "Fabio Liverani", "2019-06-01", "2020-08-02"),
    ("Lecce", "Marco Baroni", "2022-06-01", "2023-06-30"),
    ("Lecce", "Roberto D'Aversa", "2023-06-01", "2024-02-25"),
    ("Lecce", "Luca Gotti", "2024-02-26", "2024-11-18"),
    ("Lecce", "Marco Giampaolo", "2024-11-19", "2025-06-25"),
    ("Lecce", "Eusebio Di Francesco", "2025-06-26", None),

    # ── Milan ──
    ("Milan", "Vincenzo Montella", "2016-06-28", "2017-11-27"),
    ("Milan", "Gennaro Gattuso", "2017-11-28", "2019-05-28"),
    ("Milan", "Marco Giampaolo", "2019-06-20", "2019-10-08"),
    ("Milan", "Stefano Pioli", "2019-10-09", "2024-06-30"),
    ("Milan", "Paulo Fonseca", "2024-06-01", "2024-12-30"),
    ("Milan", "Sergio Conceicao", "2024-12-31", "2025-05-29"),
    ("Milan", "Massimiliano Allegri", "2025-05-30", None),

    # ── Monza ──
    ("Monza", "Giovanni Stroppa", "2022-06-01", "2022-09-13"),
    ("Monza", "Raffaele Palladino", "2022-09-14", "2024-06-30"),
    ("Monza", "Alessandro Nesta", "2024-06-01", "2024-11-04"),
    ("Monza", "Salvatore Bocchetti", "2024-11-05", "2025-05-25"),  # Relegated

    # ── Napoli ──
    ("Napoli", "Maurizio Sarri", "2015-06-12", "2018-05-23"),
    ("Napoli", "Carlo Ancelotti", "2018-05-24", "2019-12-10"),
    ("Napoli", "Gennaro Gattuso", "2019-12-11", "2021-05-23"),
    ("Napoli", "Luciano Spalletti", "2021-05-24", "2023-06-30"),
    ("Napoli", "Rudi Garcia", "2023-06-14", "2023-11-14"),
    ("Napoli", "Walter Mazzarri", "2023-11-15", "2024-02-19"),
    ("Napoli", "Francesco Calzona", "2024-02-20", "2024-06-30"),
    ("Napoli", "Antonio Conte", "2024-06-01", None),  # Continuing 2025-2026

    # ── Parma ──
    ("Parma", "Roberto D'Aversa", "2018-06-01", "2020-08-02"),
    ("Parma", "Fabio Liverani", "2020-08-03", "2021-01-11"),
    ("Parma", "Roberto D'Aversa", "2021-01-12", "2021-05-23"),
    ("Parma", "Fabio Pecchia", "2024-06-01", "2025-06-18"),
    ("Parma", "Carlos Cuesta", "2025-06-19", None),

    # ── Pisa ── (promoted 2025-2026)
    ("Pisa", "Alberto Gilardino", "2025-06-01", None),

    # ── Roma ──
    ("Roma", "Eusebio Di Francesco", "2017-06-13", "2019-03-07"),
    ("Roma", "Claudio Ranieri", "2019-03-08", "2019-05-26"),
    ("Roma", "Paulo Fonseca", "2019-06-11", "2021-05-23"),
    ("Roma", "Jose Mourinho", "2021-05-25", "2024-01-16"),
    ("Roma", "Daniele De Rossi", "2024-01-17", "2024-09-18"),
    ("Roma", "Ivan Juric", "2024-09-19", "2024-11-10"),
    ("Roma", "Claudio Ranieri", "2024-11-11", "2025-06-05"),
    ("Roma", "Gian Piero Gasperini", "2025-06-06", None),

    # ── Salernitana ──
    ("Salernitana", "Fabrizio Castori", "2021-06-01", "2021-10-16"),
    ("Salernitana", "Stefano Colantuono", "2021-10-17", "2022-02-15"),
    ("Salernitana", "Davide Nicola", "2022-02-16", "2022-05-22"),
    ("Salernitana", "Paulo Sousa", "2022-06-01", "2022-11-10"),
    ("Salernitana", "Davide Nicola", "2022-11-11", "2023-06-30"),
    ("Salernitana", "Paulo Sousa", "2023-06-01", "2023-11-07"),
    ("Salernitana", "Stefano Colantuono", "2023-11-08", "2024-02-09"),
    ("Salernitana", "Roberto Breda", "2024-02-10", "2024-05-26"),

    # ── Sampdoria ──
    ("Sampdoria", "Marco Giampaolo", "2017-01-10", "2019-06-30"),
    ("Sampdoria", "Eusebio Di Francesco", "2019-06-18", "2019-10-07"),
    ("Sampdoria", "Claudio Ranieri", "2019-10-08", "2021-05-23"),
    ("Sampdoria", "Roberto D'Aversa", "2021-06-01", "2022-02-06"),
    ("Sampdoria", "Marco Giampaolo", "2022-02-07", "2022-09-05"),
    ("Sampdoria", "Dejan Stankovic", "2022-10-04", "2023-05-28"),

    # ── Sassuolo ── (promoted back 2025-2026)
    ("Sassuolo", "Cristian Bucchi", "2017-06-01", "2017-10-02"),
    ("Sassuolo", "Giuseppe Iachini", "2017-10-03", "2018-01-09"),
    ("Sassuolo", "Roberto De Zerbi", "2018-01-10", "2021-05-23"),
    ("Sassuolo", "Alessio Dionisi", "2021-06-01", "2024-06-30"),
    ("Sassuolo", "Fabio Grosso", "2024-07-01", None),  # Promoted to Serie A 2025-2026

    # ── Spezia ──
    ("Spezia", "Vincenzo Italiano", "2020-06-01", "2021-06-21"),
    ("Spezia", "Thiago Motta", "2021-06-22", "2022-02-13"),
    ("Spezia", "Luca Gotti", "2022-06-01", "2023-05-28"),

    # ── SPAL ──
    ("SPAL", "Leonardo Semplici", "2017-06-01", "2020-02-10"),
    ("SPAL", "Luigi Di Biagio", "2020-02-11", "2020-08-02"),

    # ── Torino ──
    ("Torino", "Sinisa Mihajlovic", "2016-01-06", "2018-01-04"),
    ("Torino", "Walter Mazzarri", "2018-01-05", "2020-02-03"),
    ("Torino", "Moreno Longo", "2020-02-04", "2020-08-02"),
    ("Torino", "Marco Giampaolo", "2020-08-03", "2021-01-17"),
    ("Torino", "Davide Nicola", "2021-01-18", "2021-05-23"),
    ("Torino", "Ivan Juric", "2021-06-01", "2024-09-18"),
    ("Torino", "Paolo Vanoli", "2024-06-01", "2025-06-05"),
    ("Torino", "Marco Baroni", "2025-06-06", None),

    # ── Udinese ──
    ("Udinese", "Luigi Delneri", "2017-06-01", "2017-11-14"),
    ("Udinese", "Massimo Oddo", "2017-11-15", "2018-02-13"),
    ("Udinese", "Igor Tudor", "2018-02-14", "2018-06-30"),
    ("Udinese", "Julio Velazquez", "2018-06-01", "2018-11-05"),
    ("Udinese", "Davide Nicola", "2018-11-06", "2019-06-30"),
    ("Udinese", "Igor Tudor", "2019-06-01", "2019-11-06"),
    ("Udinese", "Luca Gotti", "2019-11-07", "2021-11-28"),
    ("Udinese", "Gabriele Cioffi", "2021-11-29", "2022-06-30"),
    ("Udinese", "Andrea Sottil", "2022-06-01", "2023-10-09"),
    ("Udinese", "Fabio Cannavaro", "2023-10-10", "2024-02-02"),
    ("Udinese", "Gabriele Cioffi", "2024-02-03", "2024-06-30"),
    ("Udinese", "Kosta Runjaic", "2024-06-01", None),  # Continuing 2025-2026

    # ── Venezia ──
    ("Venezia", "Paolo Zanetti", "2021-06-01", "2022-05-22"),
    ("Venezia", "Eusebio Di Francesco", "2024-06-01", "2025-05-25"),  # Relegated

    # ── Verona ──
    ("Verona", "Fabio Pecchia", "2017-06-01", "2018-05-20"),
    ("Verona", "Ivan Juric", "2019-06-01", "2021-05-23"),
    ("Verona", "Eusebio Di Francesco", "2021-06-01", "2021-09-21"),
    ("Verona", "Igor Tudor", "2021-09-22", "2022-06-30"),
    ("Verona", "Gabriele Cioffi", "2022-06-01", "2022-10-10"),
    ("Verona", "Salvatore Bocchetti", "2022-10-11", "2023-01-09"),
    ("Verona", "Marco Zaffaroni", "2023-01-10", "2023-05-28"),
    ("Verona", "Marco Baroni", "2023-06-01", "2024-06-30"),
    ("Verona", "Paolo Zanetti", "2024-06-01", "2026-02-02"),
    ("Verona", "Paolo Sammarco", "2026-02-03", None),  # Caretaker
]


def resolve_manager(team: str, match_date: str) -> str | None:
    """Look up who managed a team on a given date.

    Returns the manager name, or None if not found.
    """
    dt = pd.Timestamp(match_date)
    for t, mgr, start, end in MANAGER_TENURES:
        if t != team:
            continue
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else pd.Timestamp("2099-12-31")
        if s <= dt <= e:
            return mgr
    return None


def backfill_managers(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill in home_manager/away_manager from the static lookup.

    Only fills rows where the column is NaN/None (preserves FBref data).
    """
    df = matches.copy()

    for prefix in ("home", "away"):
        mgr_col = f"{prefix}_manager"
        team_col = f"{prefix}_team"

        if mgr_col not in df.columns:
            df[mgr_col] = None

        mask = df[mgr_col].isna() | (df[mgr_col].astype(str).str.strip() == "")
        if not mask.any():
            continue

        df.loc[mask, mgr_col] = df.loc[mask].apply(
            lambda row: resolve_manager(row[team_col], row["match_date"]),
            axis=1,
        )

    return df
