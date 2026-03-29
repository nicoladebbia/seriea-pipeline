"""
Module: config/managers.py
Purpose: Historical manager tenure data (Serie A + EPL) for manager-change features and form discontinuity detection
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

    # ═══════════════════════════════════════════════════════════════════════
    # PREMIER LEAGUE (England) — 2017-present
    # ═══════════════════════════════════════════════════════════════════════

    # ── Arsenal ──
    ("Arsenal", "Arsene Wenger", "2017-06-01", "2018-05-13"),
    ("Arsenal", "Unai Emery", "2018-05-23", "2019-11-29"),
    ("Arsenal", "Freddie Ljungberg", "2019-11-29", "2019-12-20"),
    ("Arsenal", "Mikel Arteta", "2019-12-20", None),

    # ── Aston Villa ──
    ("Aston Villa", "Steve Bruce", "2017-06-01", "2018-10-03"),
    ("Aston Villa", "Dean Smith", "2018-10-10", "2021-11-07"),
    ("Aston Villa", "Steven Gerrard", "2021-11-11", "2022-10-20"),
    ("Aston Villa", "Unai Emery", "2022-10-24", None),

    # ── Bournemouth ──
    ("Bournemouth", "Eddie Howe", "2017-06-01", "2020-08-01"),
    ("Bournemouth", "Jason Tindall", "2020-08-01", "2021-02-03"),
    ("Bournemouth", "Jonathan Woodgate", "2021-02-03", "2021-06-30"),
    ("Bournemouth", "Scott Parker", "2021-06-29", "2022-08-30"),
    ("Bournemouth", "Gary O'Neil", "2022-08-30", "2023-06-30"),
    ("Bournemouth", "Andoni Iraola", "2023-06-29", None),

    # ── Brentford ──
    ("Brentford", "Thomas Frank", "2018-10-16", None),

    # ── Brighton ──
    ("Brighton", "Chris Hughton", "2017-06-01", "2019-05-13"),
    ("Brighton", "Graham Potter", "2019-05-20", "2022-09-08"),
    ("Brighton", "Roberto De Zerbi", "2022-09-18", "2023-06-30"),
    ("Brighton", "Fabian Hurzeler", "2024-06-08", None),

    # ── Burnley ──
    ("Burnley", "Sean Dyche", "2017-06-01", "2022-04-15"),
    ("Burnley", "Mike Jackson", "2022-04-15", "2022-05-22"),
    ("Burnley", "Vincent Kompany", "2022-06-14", "2024-05-29"),
    ("Burnley", "Scott Parker", "2024-06-01", "2025-02-20"),
    ("Burnley", "Craig Bellamy", "2025-02-24", None),

    # ── Chelsea ──
    ("Chelsea", "Antonio Conte", "2017-06-01", "2018-07-13"),
    ("Chelsea", "Maurizio Sarri", "2018-07-14", "2019-06-16"),
    ("Chelsea", "Frank Lampard", "2019-07-04", "2021-01-25"),
    ("Chelsea", "Thomas Tuchel", "2021-01-26", "2022-09-07"),
    ("Chelsea", "Graham Potter", "2022-09-08", "2023-04-02"),
    ("Chelsea", "Frank Lampard", "2023-04-06", "2023-06-30"),
    ("Chelsea", "Mauricio Pochettino", "2023-07-01", "2024-05-21"),
    ("Chelsea", "Enzo Maresca", "2024-06-03", None),

    # ── Crystal Palace ──
    ("Crystal Palace", "Roy Hodgson", "2017-09-12", "2021-05-23"),
    ("Crystal Palace", "Patrick Vieira", "2021-07-04", "2023-03-17"),
    ("Crystal Palace", "Roy Hodgson", "2023-03-21", "2024-03-19"),
    ("Crystal Palace", "Oliver Glasner", "2024-03-20", None),

    # ── Everton ──
    ("Everton", "Sam Allardyce", "2017-11-30", "2018-05-16"),
    ("Everton", "Marco Silva", "2018-05-31", "2019-12-05"),
    ("Everton", "Duncan Ferguson", "2019-12-05", "2019-12-21"),
    ("Everton", "Carlo Ancelotti", "2019-12-21", "2021-06-01"),
    ("Everton", "Rafael Benitez", "2021-06-30", "2022-01-16"),
    ("Everton", "Frank Lampard", "2022-01-31", "2023-01-23"),
    ("Everton", "Sean Dyche", "2023-01-30", "2025-01-09"),
    ("Everton", "David Moyes", "2025-01-12", None),

    # ── Fulham ──
    ("Fulham", "Slavisa Jokanovic", "2017-06-01", "2018-11-14"),
    ("Fulham", "Claudio Ranieri", "2018-11-14", "2019-02-28"),
    ("Fulham", "Scott Parker", "2019-02-28", "2021-06-28"),
    ("Fulham", "Marco Silva", "2021-07-01", None),

    # ── Ipswich ──
    ("Ipswich", "Kieran McKenna", "2021-12-16", None),

    # ── Leeds ──
    ("Leeds", "Thomas Christiansen", "2017-06-14", "2018-02-04"),
    ("Leeds", "Paul Heckingbottom", "2018-02-05", "2018-06-01"),
    ("Leeds", "Marcelo Bielsa", "2018-06-15", "2022-02-27"),
    ("Leeds", "Jesse Marsch", "2022-02-28", "2023-02-06"),
    ("Leeds", "Javi Gracia", "2023-02-13", "2023-05-04"),
    ("Leeds", "Sam Allardyce", "2023-05-05", "2023-05-28"),
    ("Leeds", "Daniel Farke", "2023-07-04", None),

    # ── Leicester ──
    ("Leicester", "Craig Shakespeare", "2017-06-01", "2017-10-17"),
    ("Leicester", "Claude Puel", "2017-10-25", "2019-02-24"),
    ("Leicester", "Brendan Rodgers", "2019-02-26", "2023-04-02"),
    ("Leicester", "Dean Smith", "2023-04-03", "2023-05-28"),
    ("Leicester", "Enzo Maresca", "2023-06-20", "2024-06-01"),
    ("Leicester", "Steve Cooper", "2024-06-19", "2025-04-06"),
    ("Leicester", "Ruud van Nistelrooy", "2025-04-07", None),

    # ── Liverpool ──
    ("Liverpool", "Jurgen Klopp", "2017-06-01", "2024-05-19"),
    ("Liverpool", "Arne Slot", "2024-06-01", None),

    # ── Luton ──
    ("Luton", "Nathan Jones", "2017-06-01", "2019-01-09"),
    ("Luton", "Mick Harford", "2019-01-09", "2019-05-04"),
    ("Luton", "Graeme Jones", "2019-05-14", "2020-04-24"),
    ("Luton", "Nathan Jones", "2020-05-01", "2023-06-30"),
    ("Luton", "Rob Edwards", "2023-06-21", None),

    # ── Man City ──
    ("Man City", "Pep Guardiola", "2017-06-01", None),

    # ── Man United ──
    ("Man United", "Jose Mourinho", "2017-06-01", "2018-12-18"),
    ("Man United", "Ole Gunnar Solskjaer", "2018-12-19", "2021-11-21"),
    ("Man United", "Michael Carrick", "2021-11-21", "2021-11-29"),
    ("Man United", "Ralf Rangnick", "2021-11-29", "2022-05-22"),
    ("Man United", "Erik ten Hag", "2022-05-23", "2024-10-28"),
    ("Man United", "Ruud van Nistelrooy", "2024-10-28", "2024-11-10"),
    ("Man United", "Ruben Amorim", "2024-11-11", None),

    # ── Newcastle ──
    ("Newcastle", "Rafael Benitez", "2017-06-01", "2019-06-30"),
    ("Newcastle", "Steve Bruce", "2019-07-17", "2021-10-20"),
    ("Newcastle", "Eddie Howe", "2021-11-08", None),

    # ── Norwich ──
    ("Norwich", "Daniel Farke", "2017-05-25", "2021-11-06"),
    ("Norwich", "Dean Smith", "2021-11-15", "2023-05-28"),

    # ── Nottingham Forest ──
    ("Nottingham Forest", "Mark Warburton", "2017-03-14", "2017-12-31"),
    ("Nottingham Forest", "Aitor Karanka", "2018-01-15", "2019-01-11"),
    ("Nottingham Forest", "Martin O'Neill", "2019-01-15", "2019-06-28"),
    ("Nottingham Forest", "Sabri Lamouchi", "2019-06-28", "2020-10-06"),
    ("Nottingham Forest", "Chris Hughton", "2020-10-06", "2021-09-16"),
    ("Nottingham Forest", "Steve Cooper", "2021-09-21", "2023-12-20"),
    ("Nottingham Forest", "Nuno Espirito Santo", "2023-12-20", None),

    # ── Sheffield United ──
    ("Sheffield United", "Chris Wilder", "2017-05-12", "2021-03-13"),
    ("Sheffield United", "Paul Heckingbottom", "2021-03-22", "2023-12-14"),
    ("Sheffield United", "Chris Wilder", "2023-12-14", "2024-05-19"),
    ("Sheffield United", "Chris Wilder", "2025-07-01", None),

    # ── Southampton ──
    ("Southampton", "Mauricio Pellegrino", "2017-06-14", "2018-03-12"),
    ("Southampton", "Mark Hughes", "2018-03-12", "2018-12-03"),
    ("Southampton", "Ralph Hasenhuttl", "2018-12-05", "2022-11-07"),
    ("Southampton", "Nathan Jones", "2023-02-08", "2023-02-12"),
    ("Southampton", "Ruben Selles", "2023-02-12", "2023-06-22"),
    ("Southampton", "Russell Martin", "2023-06-22", "2024-11-11"),
    ("Southampton", "Ivan Juric", "2025-01-06", None),

    # ── Tottenham ──
    ("Tottenham", "Mauricio Pochettino", "2017-06-01", "2019-11-19"),
    ("Tottenham", "Jose Mourinho", "2019-11-20", "2021-04-19"),
    ("Tottenham", "Ryan Mason", "2021-04-20", "2021-06-30"),
    ("Tottenham", "Nuno Espirito Santo", "2021-06-30", "2021-11-01"),
    ("Tottenham", "Antonio Conte", "2021-11-02", "2023-03-26"),
    ("Tottenham", "Cristian Stellini", "2023-03-27", "2023-04-23"),
    ("Tottenham", "Ryan Mason", "2023-04-24", "2023-06-11"),
    ("Tottenham", "Ange Postecoglou", "2023-06-12", None),

    # ── Watford ──
    ("Watford", "Marco Silva", "2017-05-27", "2018-01-21"),
    ("Watford", "Javi Gracia", "2018-01-21", "2019-09-07"),
    ("Watford", "Quique Sanchez Flores", "2019-09-07", "2019-12-01"),
    ("Watford", "Nigel Pearson", "2019-12-06", "2020-07-19"),
    ("Watford", "Hayden Mullins", "2020-07-19", "2020-08-02"),
    ("Watford", "Vladimir Ivic", "2020-08-12", "2020-12-19"),
    ("Watford", "Xisco Munoz", "2020-12-20", "2021-10-03"),
    ("Watford", "Claudio Ranieri", "2021-10-04", "2022-01-24"),
    ("Watford", "Roy Hodgson", "2022-01-25", "2022-05-22"),

    # ── West Brom ──
    ("West Brom", "Tony Pulis", "2017-06-01", "2017-11-20"),
    ("West Brom", "Alan Pardew", "2017-11-29", "2018-04-02"),
    ("West Brom", "Darren Moore", "2018-04-02", "2019-03-09"),
    ("West Brom", "Slaven Bilic", "2019-06-24", "2020-12-16"),
    ("West Brom", "Sam Allardyce", "2020-12-16", "2021-05-23"),

    # ── West Ham ──
    ("West Ham", "Slaven Bilic", "2017-06-01", "2017-11-06"),
    ("West Ham", "David Moyes", "2017-11-07", "2018-05-13"),
    ("West Ham", "Manuel Pellegrini", "2018-05-22", "2019-12-28"),
    ("West Ham", "David Moyes", "2019-12-29", "2024-05-19"),
    ("West Ham", "Julen Lopetegui", "2024-06-17", "2025-01-08"),
    ("West Ham", "Graham Potter", "2025-01-09", None),

    # ── Wolves ──
    ("Wolves", "Nuno Espirito Santo", "2017-05-31", "2021-05-23"),
    ("Wolves", "Bruno Lage", "2021-06-09", "2022-10-02"),
    ("Wolves", "Julen Lopetegui", "2022-11-05", "2023-08-08"),
    ("Wolves", "Gary O'Neil", "2023-08-09", "2024-12-15"),
    ("Wolves", "Vitor Pereira", "2024-12-16", None),

    # ── Huddersfield ──
    ("Huddersfield", "David Wagner", "2017-06-01", "2019-01-14"),
    ("Huddersfield", "Jan Siewert", "2019-01-21", "2019-08-12"),

    # ── Cardiff ──
    ("Cardiff", "Neil Warnock", "2017-06-01", "2019-11-06"),

    # ── Swansea ──
    ("Swansea", "Paul Clement", "2017-06-01", "2017-12-20"),
    ("Swansea", "Carlos Carvalhal", "2017-12-28", "2018-05-13"),

    # ── Stoke ──
    ("Stoke", "Mark Hughes", "2017-06-01", "2018-01-06"),
    ("Stoke", "Paul Lambert", "2018-01-15", "2018-05-13"),

    # ── Middlesbrough ── (relegated 2017)
    ("Middlesbrough", "Aitor Karanka", "2017-06-01", "2017-03-16"),

    # ── Hull ── (relegated 2017)
    ("Hull", "Marco Silva", "2017-01-05", "2017-05-21"),

    # ── Sunderland ── (relegated 2017)
    ("Sunderland", "David Moyes", "2017-06-01", "2017-05-28"),
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
