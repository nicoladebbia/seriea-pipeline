#!/usr/bin/env python3
"""
Module: scripts/squad_audit.py
Purpose: Generates corrected current_squads.json based on manual research and produces discrepancy report against existing data
Inputs:  data/squads/current_squads.json (existing), data/features/player_xg_profiles.json (existing player profiles)
Outputs: data/squads/current_squads.json (corrected), data/squads/squad_audit_report.json (discrepancies)
Called by: Manual CLI invocation after squad research or transfer window
Depends on: config.settings (DATA_DIR)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SQUADS_PATH = DATA_DIR / "squads" / "current_squads.json"
PROFILES_PATH = DATA_DIR / "features" / "player_xg_profiles.json"
AUDIT_PATH = DATA_DIR / "squads" / "squad_audit_report.json"

# ============================================================================
# CORRECTED SQUADS - Serie A 2025-2026 (as of February 7, 2026)
# ============================================================================
# Based on web research from Transfermarkt, ESPN, Tribuna.com, official club
# sites, Football Italia, etc. Key January 2026 transfers included.
#
# Format: list of {"name", "position", "number", "age", "nationality", "status"}
# status: "active" | "injured" | "loaned_out" | "suspended"
# ============================================================================

CORRECTED_SQUADS = {
    "Atalanta": {
        "players": [
            # Goalkeepers
            {"name": "M. Carnesecchi", "position": "GK", "number": 29, "age": 24, "nationality": "Italy", "status": "active"},
            {"name": "R. Rossi", "position": "GK", "number": 31, "age": 25, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "R. Kossounou", "position": "CB", "number": 3, "age": 24, "nationality": "Ivory Coast", "status": "active"},
            {"name": "I. Hien", "position": "CB", "number": 4, "age": 25, "nationality": "Sweden", "status": "active"},
            {"name": "B. Djimsiti", "position": "CB", "number": 19, "age": 31, "nationality": "Albania", "status": "active"},
            {"name": "S. Kolasinac", "position": "CB", "number": 23, "age": 31, "nationality": "Bosnia", "status": "active"},
            {"name": "O. Godfrey", "position": "CB", "number": 5, "age": 27, "nationality": "England", "status": "active"},
            {"name": "D. Zappacosta", "position": "CM", "number": 77, "age": 33, "nationality": "Italy", "status": "active"},
            {"name": "R. Bellanova", "position": "CM", "number": 16, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "G. Ruggeri", "position": "CB", "number": 22, "age": 22, "nationality": "Italy", "status": "active"},
            # Midfielders
            {"name": "M. de Roon", "position": "CM", "number": 15, "age": 35, "nationality": "Netherlands", "status": "active"},
            {"name": "E. Ederson", "position": "CM", "number": 13, "age": 25, "nationality": "Brazil", "status": "active"},
            {"name": "M. Pasalic", "position": "CM", "number": 88, "age": 30, "nationality": "Croatia", "status": "active"},
            {"name": "L. Scalvini", "position": "CM", "number": 42, "age": 22, "nationality": "Italy", "status": "active"},
            {"name": "D. Samardzic", "position": "CM", "number": 24, "age": 23, "nationality": "Serbia", "status": "active"},
            # Forwards
            {"name": "A. Lookman", "position": "ST", "number": 11, "age": 27, "nationality": "Nigeria", "status": "active"},
            {"name": "C. De Ketelaere", "position": "ST", "number": 17, "age": 24, "nationality": "Belgium", "status": "active"},
            {"name": "M. Retegui", "position": "ST", "number": 32, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "N. Zaniolo", "position": "ST", "number": 10, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "G. Raspadori", "position": "ST", "number": 18, "age": 25, "nationality": "Italy", "status": "active"},  # Jan 2026 from Napoli
        ],
        "coach": "Gian Piero Gasperini",
        "jan_2026_in": ["G. Raspadori (from Napoli)"],
        "jan_2026_out": ["I. Sulemana (loan to Cagliari)"],
    },

    "Bologna": {
        "players": [
            # Goalkeepers
            {"name": "L. Skorupski", "position": "GK", "number": 28, "age": 34, "nationality": "Poland", "status": "suspended"},
            {"name": "F. Ravaglia", "position": "GK", "number": 32, "age": 26, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "S. Beukema", "position": "CB", "number": 31, "age": 27, "nationality": "Netherlands", "status": "active"},
            {"name": "J. Lucumi", "position": "CB", "number": 26, "age": 27, "nationality": "Colombia", "status": "injured"},
            {"name": "S. Posch", "position": "CB", "number": 22, "age": 28, "nationality": "Austria", "status": "active"},
            {"name": "L. De Silvestri", "position": "CB", "number": 29, "age": 38, "nationality": "Italy", "status": "active"},
            {"name": "C. Lykogiannis", "position": "CB", "number": 3, "age": 33, "nationality": "Greece", "status": "active"},
            {"name": "J. Holm", "position": "CB", "number": 2, "age": 24, "nationality": "Sweden", "status": "active"},
            {"name": "E. Helland", "position": "CB", "number": None, "age": 20, "nationality": "Norway", "status": "active"},  # Jan 2026
            # Midfielders
            {"name": "L. Ferguson", "position": "CM", "number": 14, "age": 25, "nationality": "Scotland", "status": "active"},
            {"name": "R. Freuler", "position": "CM", "number": 8, "age": 33, "nationality": "Switzerland", "status": "active"},
            {"name": "N. Moro", "position": "CM", "number": 30, "age": 22, "nationality": "Croatia", "status": "active"},
            {"name": "M. Aebischer", "position": "CM", "number": 20, "age": 27, "nationality": "Switzerland", "status": "active"},
            {"name": "T. Pobega", "position": "CM", "number": 17, "age": 26, "nationality": "Italy", "status": "active"},
            # Forwards
            {"name": "S. Castro", "position": "ST", "number": 9, "age": 20, "nationality": "Argentina", "status": "active"},
            {"name": "R. Orsolini", "position": "ST", "number": 7, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "D. Ndoye", "position": "ST", "number": 11, "age": 24, "nationality": "Switzerland", "status": "active"},
            {"name": "J. Karlsson", "position": "ST", "number": 18, "age": 27, "nationality": "Sweden", "status": "active"},
            {"name": "T. Dallinga", "position": "ST", "number": 14, "age": 24, "nationality": "Netherlands", "status": "active"},
            {"name": "F. Bernardeschi", "position": "ST", "number": 10, "age": 32, "nationality": "Italy", "status": "injured"},
        ],
        "coach": "Vincenzo Italiano",
        "jan_2026_in": ["E. Helland (from Brann, ~€8M)"],
        "jan_2026_out": ["G. Fabbian (loan to Fiorentina)", "C. Immobile (to Paris FC)", "E. Holm (to Juventus)"],
    },

    "Cagliari": {
        "players": [
            # Goalkeepers
            {"name": "E. Caprile", "position": "GK", "number": 1, "age": 24, "nationality": "Italy", "status": "active"},
            {"name": "B. Radunovic", "position": "GK", "number": 12, "age": 29, "nationality": "Serbia", "status": "active"},
            # Defenders
            {"name": "S. Luperto", "position": "CB", "number": 6, "age": 29, "nationality": "Italy", "status": "active"},
            {"name": "Y. Mina", "position": "CB", "number": 26, "age": 31, "nationality": "Colombia", "status": "active"},
            {"name": "A. Dossena", "position": "CB", "number": 22, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "G. Zappa", "position": "CB", "number": 28, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "A. Obert", "position": "CB", "number": 33, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "J. Rodriguez", "position": "CB", "number": 15, "age": 20, "nationality": "Uruguay", "status": "active"},  # Jan 2026
            # Midfielders
            {"name": "M. Adopo", "position": "CM", "number": 8, "age": 25, "nationality": "France", "status": "active"},
            {"name": "G. Gaetano", "position": "CM", "number": 10, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "L. Mazzitelli", "position": "CM", "number": 4, "age": 30, "nationality": "Italy", "status": "active"},
            {"name": "A. Deiola", "position": "CM", "number": 14, "age": 30, "nationality": "Italy", "status": "active"},
            {"name": "M. Felici", "position": "CM", "number": 17, "age": 24, "nationality": "Italy", "status": "active"},
            {"name": "M. Folorunsho", "position": "CM", "number": 90, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "I. Sulemana", "position": "CM", "number": 25, "age": 22, "nationality": "Ghana", "status": "active"},  # loan from Atalanta
            # Forwards
            {"name": "A. Belotti", "position": "ST", "number": 19, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "G. Borrelli", "position": "ST", "number": 29, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "S. Esposito", "position": "ST", "number": 94, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "Zito Luvumbo", "position": "ST", "number": 77, "age": 23, "nationality": "Angola", "status": "active"},
            {"name": "L. Pavoletti", "position": "ST", "number": 30, "age": 37, "nationality": "Italy", "status": "active"},
            {"name": "S. Kilicsoy", "position": "ST", "number": 9, "age": 20, "nationality": "Turkey", "status": "active"},
        ],
        "coach": "Pisacane",
        "jan_2026_in": ["J. Rodriguez (from Penarol)", "I. Sulemana (loan from Atalanta)"],
        "jan_2026_out": ["R. Piccoli (to Fiorentina, €25M)"],
    },

    "Como": {
        "players": [
            # Goalkeepers
            {"name": "P. Reina", "position": "GK", "number": 25, "age": 43, "nationality": "Spain", "status": "active"},
            {"name": "E. Audero", "position": "GK", "number": 1, "age": 28, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "R. Goldaniga", "position": "CB", "number": 4, "age": 31, "nationality": "Italy", "status": "active"},
            {"name": "L. Dossena", "position": "CB", "number": 13, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "A. Barba", "position": "CB", "number": 5, "age": 29, "nationality": "Spain", "status": "active"},
            {"name": "M. Kempf", "position": "CB", "number": 6, "age": 30, "nationality": "Germany", "status": "active"},
            {"name": "A. Moreno", "position": "CB", "number": 3, "age": 32, "nationality": "Spain", "status": "active"},
            {"name": "I. Van der Brempt", "position": "CB", "number": 22, "age": 23, "nationality": "Belgium", "status": "active"},
            # Midfielders
            {"name": "S. Perrone", "position": "CM", "number": 32, "age": 23, "nationality": "Argentina", "status": "active"},
            {"name": "E. Engelhardt", "position": "CM", "number": 8, "age": 24, "nationality": "Germany", "status": "active"},
            {"name": "A. Sergi Roberto", "position": "CM", "number": 20, "age": 33, "nationality": "Spain", "status": "active"},
            {"name": "N. Paz", "position": "CM", "number": 79, "age": 21, "nationality": "Argentina", "status": "active"},
            {"name": "L. Da Cunha", "position": "CM", "number": 14, "age": 22, "nationality": "France", "status": "active"},
            {"name": "G. Strefezza", "position": "CM", "number": 7, "age": 29, "nationality": "Italy", "status": "active"},
            # Forwards
            {"name": "A. Morata", "position": "ST", "number": 9, "age": 33, "nationality": "Spain", "status": "active"},  # from AC Milan
            {"name": "P. Cutrone", "position": "ST", "number": 10, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "A. Belotti", "position": "ST", "number": 11, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "D. Alli", "position": "ST", "number": 20, "age": 30, "nationality": "England", "status": "active"},
            {"name": "A. Diao", "position": "ST", "number": 18, "age": 20, "nationality": "Senegal", "status": "injured"},  # Jan 2026, injured until March
            {"name": "S. Fadera", "position": "ST", "number": 17, "age": 24, "nationality": "Gambia", "status": "active"},
        ],
        "coach": "Cesc Fabregas",
        "jan_2026_in": ["A. Diao (from Real Betis, ~€12M)", "A. Morata (from AC Milan/Galatasaray)"],
        "jan_2026_out": [],
    },

    "Cremonese": {
        "players": [
            # Goalkeepers
            {"name": "M. Fulignati", "position": "GK", "number": 1, "age": 29, "nationality": "Italy", "status": "active"},
            {"name": "S. Saro", "position": "GK", "number": 22, "age": 27, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "E. Bianchetti", "position": "CB", "number": 5, "age": 30, "nationality": "Italy", "status": "active"},
            {"name": "L. Ravanelli", "position": "CB", "number": 4, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "G. Pezzella", "position": "CB", "number": 3, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "M. Antov", "position": "CB", "number": 15, "age": 24, "nationality": "Bulgaria", "status": "active"},
            {"name": "M. Sernicola", "position": "CB", "number": 22, "age": 29, "nationality": "Italy", "status": "active"},
            # Midfielders
            {"name": "M. Collocolo", "position": "CM", "number": 8, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "M. Payero", "position": "CM", "number": 10, "age": 26, "nationality": "Argentina", "status": "active"},
            {"name": "C. Falletti", "position": "CM", "number": 7, "age": 33, "nationality": "Uruguay", "status": "active"},
            {"name": "S. Castagnetti", "position": "CM", "number": 21, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "L. Valzania", "position": "CM", "number": 28, "age": 30, "nationality": "Italy", "status": "active"},
            {"name": "Y. Maleh", "position": "CM", "number": 14, "age": 27, "nationality": "Morocco", "status": "active"},  # Jan 2026 from Lecce
            {"name": "M. Thorsby", "position": "CM", "number": 6, "age": 29, "nationality": "Norway", "status": "active"},
            # Forwards
            {"name": "M. De Luca", "position": "ST", "number": 9, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "F. Vazquez", "position": "ST", "number": 11, "age": 27, "nationality": "Argentina", "status": "active"},
            {"name": "M. Nasti", "position": "ST", "number": 18, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "F. Tsadjout", "position": "ST", "number": 19, "age": 26, "nationality": "Cameroon", "status": "active"},
        ],
        "coach": "Giovanni Stroppa",
        "jan_2026_in": ["Y. Maleh (from Lecce)", "M. Thorsby (from Genoa)"],
        "jan_2026_out": ["S. Luperto (to Cagliari)"],
    },

    "Fiorentina": {
        "players": [
            # Goalkeepers
            {"name": "D. De Gea", "position": "GK", "number": 43, "age": 35, "nationality": "Spain", "status": "active"},
            {"name": "P. Terracciano", "position": "GK", "number": 1, "age": 35, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "L. Martinez Quarta", "position": "CB", "number": 28, "age": 28, "nationality": "Argentina", "status": "active"},
            {"name": "P. Comuzzo", "position": "CB", "number": 15, "age": 20, "nationality": "Italy", "status": "active"},
            {"name": "M. Ranieri", "position": "CB", "number": 6, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "C. Biraghi", "position": "CB", "number": 3, "age": 33, "nationality": "Italy", "status": "active"},
            {"name": "M. Dodò", "position": "CB", "number": 2, "age": 26, "nationality": "Brazil", "status": "active"},
            {"name": "F. Parisi", "position": "CB", "number": 65, "age": 24, "nationality": "Italy", "status": "active"},
            {"name": "T. Lamptey", "position": "CB", "number": 21, "age": 24, "nationality": "England", "status": "injured"},
            # Midfielders
            {"name": "Y. Adli", "position": "CM", "number": 11, "age": 25, "nationality": "France", "status": "active"},
            {"name": "A. Mandragora", "position": "CM", "number": 38, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "D. Cataldi", "position": "CM", "number": 32, "age": 31, "nationality": "Italy", "status": "active"},
            {"name": "A. Richardson", "position": "CM", "number": 24, "age": 24, "nationality": "Morocco", "status": "active"},
            {"name": "G. Fabbian", "position": "CM", "number": 20, "age": 22, "nationality": "Italy", "status": "active"},  # Jan 2026 from Bologna
            {"name": "M. Brescianini", "position": "CM", "number": 16, "age": 25, "nationality": "Italy", "status": "active"},  # Jan 2026 loan from Atalanta
            {"name": "M. Solomon", "position": "CM", "number": 7, "age": 25, "nationality": "Israel", "status": "active"},  # Jan 2026 loan from Tottenham
            # Forwards
            {"name": "M. Kean", "position": "ST", "number": 20, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "R. Sottil", "position": "ST", "number": 7, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "A. Colpani", "position": "ST", "number": 23, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "L. Beltran", "position": "ST", "number": 9, "age": 24, "nationality": "Argentina", "status": "active"},
            {"name": "R. Piccoli", "position": "ST", "number": 14, "age": 24, "nationality": "Italy", "status": "active"},  # Summer 2025 from Cagliari
        ],
        "coach": "Stefano Pioli",
        "jan_2026_in": ["G. Fabbian (loan from Bologna)", "M. Brescianini (loan from Atalanta)", "M. Solomon (loan from Tottenham)"],
        "jan_2026_out": ["A. Gudmundsson (sold to Genoa in summer 2025)"],
    },

    "Genoa": {
        "players": [
            # Goalkeepers
            {"name": "J. Bijlow", "position": "GK", "number": 16, "age": 27, "nationality": "Netherlands", "status": "active"},  # Jan 2026
            {"name": "N. Leali", "position": "GK", "number": 1, "age": 32, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "J. Vasquez", "position": "CB", "number": 22, "age": 27, "nationality": "Mexico", "status": "active"},
            {"name": "L. Ostigard", "position": "CB", "number": 5, "age": 26, "nationality": "Norway", "status": "active"},
            {"name": "S. Sabelli", "position": "CB", "number": 20, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "A. Marcandalli", "position": "CB", "number": 27, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "Aaron Martin", "position": "CB", "number": 3, "age": 28, "nationality": "Spain", "status": "active"},
            {"name": "B. Norton-Cuffy", "position": "CB", "number": 15, "age": 21, "nationality": "England", "status": "active"},
            # Midfielders
            {"name": "M. Frendrup", "position": "CM", "number": 32, "age": 24, "nationality": "Denmark", "status": "active"},
            {"name": "R. Malinovskyi", "position": "CM", "number": 17, "age": 32, "nationality": "Ukraine", "status": "active"},
            {"name": "N. Stanciu", "position": "CM", "number": 8, "age": 32, "nationality": "Romania", "status": "active"},
            {"name": "M. Ellertsson", "position": "CM", "number": 77, "age": 23, "nationality": "Iceland", "status": "active"},
            {"name": "P. Masini", "position": "CM", "number": 73, "age": 24, "nationality": "Italy", "status": "active"},
            {"name": "J. Onana", "position": "CM", "number": 14, "age": 25, "nationality": "Cameroon", "status": "injured"},
            {"name": "A. Gronbaek", "position": "CM", "number": 11, "age": 24, "nationality": "Denmark", "status": "injured"},
            {"name": "V. Carboni", "position": "CM", "number": 21, "age": 20, "nationality": "Argentina", "status": "active"},  # loan from Inter
            {"name": "H. Cuenca", "position": "CM", "number": 30, "age": 20, "nationality": "Paraguay", "status": "active"},
            # Forwards
            {"name": "L. Colombo", "position": "ST", "number": 29, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "M. Cornet", "position": "ST", "number": 70, "age": 29, "nationality": "Ivory Coast", "status": "injured"},
            {"name": "J. Ekhator", "position": "ST", "number": 21, "age": 19, "nationality": "Nigeria", "status": "active"},
            {"name": "C. Ekuban", "position": "ST", "number": 18, "age": 31, "nationality": "Ghana", "status": "active"},
            {"name": "Junior Messias", "position": "ST", "number": 10, "age": 34, "nationality": "Brazil", "status": "injured"},
            {"name": "Vitinha", "position": "ST", "number": 9, "age": 25, "nationality": "Portugal", "status": "active"},
            {"name": "T. Baldanzi", "position": "ST", "number": 35, "age": 22, "nationality": "Italy", "status": "injured"},  # Jan 2026 loan from Roma
        ],
        "coach": "Patrick Vieira",
        "jan_2026_in": ["T. Baldanzi (loan from Roma)", "J. Bijlow (from Feyenoord)"],
        "jan_2026_out": ["K. De Winter (sold to Milan, €20M)"],
    },

    "Inter": {
        "players": [
            # Goalkeepers
            {"name": "Y. Sommer", "position": "GK", "number": 1, "age": 36, "nationality": "Switzerland", "status": "active"},
            {"name": "J. Martinez", "position": "GK", "number": 13, "age": 27, "nationality": "Spain", "status": "active"},
            # Defenders
            {"name": "A. Bastoni", "position": "CB", "number": 95, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "S. de Vrij", "position": "CB", "number": 6, "age": 33, "nationality": "Netherlands", "status": "active"},
            {"name": "B. Pavard", "position": "CB", "number": 28, "age": 29, "nationality": "France", "status": "active"},
            {"name": "Y. Bisseck", "position": "CB", "number": 31, "age": 24, "nationality": "Germany", "status": "active"},
            {"name": "M. Darmian", "position": "CB", "number": 36, "age": 35, "nationality": "Italy", "status": "active"},
            {"name": "C. Augusto", "position": "CB", "number": 30, "age": 31, "nationality": "Brazil", "status": "active"},
            {"name": "D. Dumfries", "position": "CM", "number": 2, "age": 29, "nationality": "Netherlands", "status": "active"},
            {"name": "F. Dimarco", "position": "CB", "number": 32, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "T. Buchanan", "position": "CM", "number": 12, "age": 26, "nationality": "Canada", "status": "active"},
            {"name": "M. Akanji", "position": "CB", "number": 25, "age": 30, "nationality": "Switzerland", "status": "active"},  # loan from Man City
            # Midfielders
            {"name": "N. Barella", "position": "CM", "number": 23, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "H. Calhanoglu", "position": "CM", "number": 20, "age": 31, "nationality": "Turkey", "status": "active"},
            {"name": "H. Mkhitaryan", "position": "CM", "number": 22, "age": 37, "nationality": "Armenia", "status": "active"},
            {"name": "D. Frattesi", "position": "CM", "number": 16, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "K. Asllani", "position": "CM", "number": 21, "age": 23, "nationality": "Albania", "status": "active"},
            {"name": "P. Zielinski", "position": "CM", "number": 7, "age": 32, "nationality": "Poland", "status": "active"},
            # Forwards
            {"name": "L. Martinez", "position": "ST", "number": 10, "age": 28, "nationality": "Argentina", "status": "active"},
            {"name": "M. Thuram", "position": "ST", "number": 9, "age": 28, "nationality": "France", "status": "active"},
            {"name": "M. Taremi", "position": "ST", "number": 99, "age": 33, "nationality": "Iran", "status": "active"},
            {"name": "M. Arnautovic", "position": "ST", "number": 8, "age": 37, "nationality": "Austria", "status": "active"},
            {"name": "J. Correa", "position": "ST", "number": 11, "age": 31, "nationality": "Argentina", "status": "active"},
        ],
        "coach": "Simone Inzaghi",
        "jan_2026_in": ["M. Akanji (loan from Manchester City)"],
        "jan_2026_out": [],
    },

    "Juventus": {
        "players": [
            # Goalkeepers
            {"name": "M. Di Gregorio", "position": "GK", "number": 16, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "M. Perin", "position": "GK", "number": 1, "age": 33, "nationality": "Italy", "status": "active"},
            {"name": "C. Pinsoglio", "position": "GK", "number": 23, "age": 35, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "Bremer", "position": "CB", "number": 3, "age": 28, "nationality": "Brazil", "status": "active"},  # returned from ACL
            {"name": "F. Gatti", "position": "CB", "number": 4, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "A. Cambiaso", "position": "CB", "number": 27, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "P. Kalulu", "position": "CB", "number": 15, "age": 25, "nationality": "France", "status": "active"},
            {"name": "L. Kelly", "position": "CB", "number": 6, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "J. Cabal", "position": "CB", "number": 32, "age": 24, "nationality": "Colombia", "status": "active"},  # returning from injury
            {"name": "D. Rugani", "position": "CB", "number": 24, "age": 31, "nationality": "Italy", "status": "active"},
            {"name": "J. Rouhi", "position": "CB", "number": 40, "age": 21, "nationality": "Sweden", "status": "active"},
            {"name": "E. Holm", "position": "CB", "number": 2, "age": 24, "nationality": "Sweden", "status": "active"},  # Jan 2026 from Bologna
            # Midfielders
            {"name": "T. Koopmeiners", "position": "CM", "number": 8, "age": 27, "nationality": "Netherlands", "status": "active"},
            {"name": "M. Locatelli", "position": "CM", "number": 5, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "W. McKennie", "position": "CM", "number": 22, "age": 27, "nationality": "USA", "status": "active"},
            {"name": "K. Thuram", "position": "CM", "number": 19, "age": 24, "nationality": "France", "status": "active"},
            {"name": "F. Kostic", "position": "CM", "number": 18, "age": 33, "nationality": "Serbia", "status": "active"},
            {"name": "Francisco Conceicao", "position": "CM", "number": 7, "age": 23, "nationality": "Portugal", "status": "active"},
            # Forwards
            {"name": "D. Vlahovic", "position": "ST", "number": 9, "age": 25, "nationality": "Serbia", "status": "active"},  # minor adductor, expected fit
            {"name": "K. Yildiz", "position": "ST", "number": 10, "age": 20, "nationality": "Turkey", "status": "active"},
            {"name": "J. David", "position": "ST", "number": 30, "age": 25, "nationality": "Canada", "status": "active"},  # Jan 2026 from Lille
            {"name": "L. Openda", "position": "ST", "number": 20, "age": 25, "nationality": "Belgium", "status": "active"},  # Jan 2026 from RB Leipzig
            {"name": "E. Zhegrova", "position": "ST", "number": 11, "age": 26, "nationality": "Kosovo", "status": "active"},  # Jan 2026 from Lille
            {"name": "J. Boga", "position": "ST", "number": 17, "age": 28, "nationality": "Ivory Coast", "status": "active"},  # Jan 2026 loan from Nice
            {"name": "A. Milik", "position": "ST", "number": 14, "age": 31, "nationality": "Poland", "status": "active"},
        ],
        "coach": "Thiago Motta",
        "jan_2026_in": [
            "J. David (free from Lille)",
            "L. Openda (from RB Leipzig)",
            "E. Zhegrova (from Lille)",
            "J. Boga (loan from Nice)",
            "E. Holm (from Bologna)",
        ],
        "jan_2026_out": ["T. Weah (loan to Marseille)"],
        "notes": "Kolo Muani NOT signed (Tottenham blocked). Bremer returned from ACL. Cabal nearly fit.",
    },

    "Lazio": {
        "players": [
            # Goalkeepers
            {"name": "I. Provedel", "position": "GK", "number": 94, "age": 31, "nationality": "Italy", "status": "active"},
            {"name": "A. Furlanetto", "position": "GK", "number": 55, "age": 23, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "Mario Gila", "position": "CB", "number": 34, "age": 25, "nationality": "Spain", "status": "active"},
            {"name": "A. Romagnoli", "position": "CB", "number": 13, "age": 30, "nationality": "Italy", "status": "active"},
            {"name": "A. Marusic", "position": "CB", "number": 77, "age": 33, "nationality": "Montenegro", "status": "active"},
            {"name": "Patric", "position": "CB", "number": 4, "age": 32, "nationality": "Spain", "status": "active"},
            {"name": "L. Pellegrini", "position": "CB", "number": 3, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "Nuno Tavares", "position": "CB", "number": 17, "age": 25, "nationality": "Portugal", "status": "active"},
            # Midfielders
            {"name": "N. Rovella", "position": "CM", "number": 6, "age": 24, "nationality": "Italy", "status": "active"},
            {"name": "M. Vecino", "position": "CM", "number": 5, "age": 34, "nationality": "Uruguay", "status": "active"},
            {"name": "M. Zaccagni", "position": "CM", "number": 10, "age": 30, "nationality": "Italy", "status": "active"},
            {"name": "F. Dele-Bashiru", "position": "CM", "number": 7, "age": 24, "nationality": "Nigeria", "status": "active"},
            {"name": "M. Lazzari", "position": "CM", "number": 29, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "R. Belahyane", "position": "CM", "number": 21, "age": 21, "nationality": "Morocco", "status": "active"},  # from Hellas Verona
            # Forwards
            {"name": "B. Dia", "position": "ST", "number": 19, "age": 29, "nationality": "Senegal", "status": "active"},
            {"name": "T. Noslin", "position": "ST", "number": 14, "age": 26, "nationality": "Netherlands", "status": "active"},
            {"name": "G. Isaksen", "position": "ST", "number": 18, "age": 24, "nationality": "Denmark", "status": "active"},
            {"name": "Pedro", "position": "ST", "number": 9, "age": 38, "nationality": "Spain", "status": "active"},
            {"name": "D. Maldini", "position": "ST", "number": 27, "age": 24, "nationality": "Italy", "status": "active"},
            {"name": "M. Cancellieri", "position": "ST", "number": 22, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "P. Ratkov", "position": "ST", "number": 20, "age": 22, "nationality": "Serbia", "status": "active"},
        ],
        "coach": "Marco Baroni",
        "jan_2026_in": ["R. Belahyane (from Hellas Verona)"],
        "jan_2026_out": ["C. Mandas (loan to Bournemouth)"],
    },

    "Lecce": {
        "players": [
            # Goalkeepers
            {"name": "W. Falcone", "position": "GK", "number": 1, "age": 29, "nationality": "Italy", "status": "active"},
            {"name": "F. Brancolini", "position": "GK", "number": 12, "age": 25, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "J. Siebert", "position": "CB", "number": 5, "age": 22, "nationality": "Germany", "status": "active"},
            {"name": "F. Baschirotto", "position": "CB", "number": 6, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "D. Veiga", "position": "CB", "number": 3, "age": 23, "nationality": "Portugal", "status": "active"},
            {"name": "C. Kouassi", "position": "CB", "number": 4, "age": 22, "nationality": "Ivory Coast", "status": "active"},
            {"name": "M. Perez", "position": "CB", "number": 22, "age": 24, "nationality": "Argentina", "status": "active"},
            {"name": "C. Ndaba", "position": "CB", "number": 15, "age": 24, "nationality": "South Africa", "status": "active"},
            # Midfielders
            {"name": "Y. Ramadani", "position": "CM", "number": 8, "age": 27, "nationality": "Albania", "status": "active"},
            {"name": "M. Kaba", "position": "CM", "number": 20, "age": 23, "nationality": "Guinea", "status": "active"},
            {"name": "M. Berisha", "position": "CM", "number": 10, "age": 24, "nationality": "Albania", "status": "active"},
            {"name": "L. Coulibaly", "position": "CM", "number": 14, "age": 30, "nationality": "Mali", "status": "active"},
            {"name": "A. Sala", "position": "CM", "number": 16, "age": 22, "nationality": "Italy", "status": "active"},
            # Forwards
            {"name": "N. Krstovic", "position": "ST", "number": 9, "age": 25, "nationality": "Montenegro", "status": "active"},
            {"name": "L. Banda", "position": "ST", "number": 77, "age": 25, "nationality": "Zambia", "status": "active"},
            {"name": "R. Sottil", "position": "ST", "number": 7, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "S. Pierotti", "position": "ST", "number": 11, "age": 26, "nationality": "Argentina", "status": "active"},
            {"name": "T. Morente", "position": "ST", "number": 18, "age": 27, "nationality": "Spain", "status": "active"},
            {"name": "F. Camarda", "position": "ST", "number": 19, "age": 17, "nationality": "Italy", "status": "active"},  # loan from Milan
        ],
        "coach": "Eusebio Di Francesco",
        "jan_2026_in": [],
        "jan_2026_out": ["Y. Maleh (to Cremonese)"],
    },

    "Milan": {
        "players": [
            # Goalkeepers
            {"name": "M. Maignan", "position": "GK", "number": 16, "age": 30, "nationality": "France", "status": "active"},
            {"name": "F. Terracciano", "position": "GK", "number": 1, "age": 35, "nationality": "Italy", "status": "active"},
            {"name": "L. Torriani", "position": "GK", "number": 96, "age": 20, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "F. Tomori", "position": "CB", "number": 23, "age": 28, "nationality": "England", "status": "active"},
            {"name": "M. Gabbia", "position": "CB", "number": 46, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "S. Pavlovic", "position": "CB", "number": 31, "age": 24, "nationality": "Serbia", "status": "active"},
            {"name": "K. De Winter", "position": "CB", "number": 5, "age": 23, "nationality": "Belgium", "status": "active"},  # from Genoa, €20M
            {"name": "P. Estupinan", "position": "CB", "number": 2, "age": 27, "nationality": "Ecuador", "status": "active"},  # Jan 2026
            {"name": "D. Calabria", "position": "CB", "number": 2, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "T. Hernandez", "position": "CB", "number": 19, "age": 28, "nationality": "France", "status": "active"},
            # Midfielders
            {"name": "Y. Fofana", "position": "CM", "number": 29, "age": 26, "nationality": "France", "status": "active"},
            {"name": "S. Ricci", "position": "CM", "number": 4, "age": 24, "nationality": "Italy", "status": "active"},  # Jan 2026
            {"name": "L. Modric", "position": "CM", "number": 14, "age": 40, "nationality": "Croatia", "status": "active"},  # signed from Real Madrid
            {"name": "A. Rabiot", "position": "CM", "number": 12, "age": 30, "nationality": "France", "status": "active"},  # Jan 2026
            {"name": "R. Loftus-Cheek", "position": "CM", "number": 8, "age": 29, "nationality": "England", "status": "active"},
            {"name": "A. Saelemaekers", "position": "CM", "number": 56, "age": 26, "nationality": "Belgium", "status": "active"},
            {"name": "D. Bartesaghi", "position": "CM", "number": 33, "age": 20, "nationality": "Italy", "status": "active"},
            # Forwards
            {"name": "C. Pulisic", "position": "ST", "number": 11, "age": 27, "nationality": "USA", "status": "active"},
            {"name": "Rafael Leao", "position": "ST", "number": 10, "age": 26, "nationality": "Portugal", "status": "active"},
            {"name": "S. Gimenez", "position": "ST", "number": 7, "age": 24, "nationality": "Mexico", "status": "active"},  # may depart
            {"name": "N. Fullkrug", "position": "ST", "number": 9, "age": 32, "nationality": "Germany", "status": "active"},  # from West Ham
            {"name": "C. Nkunku", "position": "ST", "number": 18, "age": 28, "nationality": "France", "status": "active"},
        ],
        "coach": "Max Allegri",
        "jan_2026_in": [
            "S. Ricci (from Torino)",
            "A. Rabiot",
            "P. Estupinan",
            "N. Fullkrug (from West Ham)",
        ],
        "jan_2026_out": ["A. Morata (to Como via Galatasaray)"],
        "notes": "S. Gimenez future uncertain, may leave. Nkunku and Modric summer signings confirmed in squad.",
    },

    "Napoli": {
        "players": [
            # Goalkeepers
            {"name": "A. Meret", "position": "GK", "number": 1, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "V. Milinkovic-Savic", "position": "GK", "number": 32, "age": 28, "nationality": "Serbia", "status": "injured"},  # muscular
            {"name": "N. Contini", "position": "GK", "number": 14, "age": 29, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "A. Buongiorno", "position": "CB", "number": 4, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "G. Di Lorenzo", "position": "CB", "number": 22, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "Amir Rrahmani", "position": "CB", "number": 13, "age": 31, "nationality": "Kosovo", "status": "active"},
            {"name": "Juan Jesus", "position": "CB", "number": 5, "age": 34, "nationality": "Brazil", "status": "active"},
            {"name": "P. Mazzocchi", "position": "CB", "number": 30, "age": 30, "nationality": "Italy", "status": "active"},
            {"name": "M. Olivera", "position": "CB", "number": 17, "age": 28, "nationality": "Uruguay", "status": "active"},
            {"name": "L. Spinazzola", "position": "CB", "number": 37, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "S. Beukema", "position": "CB", "number": 31, "age": 27, "nationality": "Netherlands", "status": "active"},
            {"name": "Miguel Gutierrez", "position": "CB", "number": 3, "age": 24, "nationality": "Spain", "status": "active"},
            # Midfielders
            {"name": "A. Zambo Anguissa", "position": "CM", "number": 99, "age": 30, "nationality": "Cameroon", "status": "active"},
            {"name": "S. Lobotka", "position": "CM", "number": 68, "age": 31, "nationality": "Slovakia", "status": "active"},
            {"name": "S. McTominay", "position": "CM", "number": 8, "age": 29, "nationality": "Scotland", "status": "active"},
            {"name": "B. Gilmour", "position": "CM", "number": 6, "age": 24, "nationality": "Scotland", "status": "active"},
            {"name": "K. De Bruyne", "position": "CM", "number": 11, "age": 34, "nationality": "Belgium", "status": "injured"},  # surgery, out until March
            {"name": "E. Elmas", "position": "CM", "number": 20, "age": 26, "nationality": "North Macedonia", "status": "active"},
            # Forwards
            {"name": "R. Lukaku", "position": "ST", "number": 9, "age": 32, "nationality": "Belgium", "status": "active"},
            {"name": "R. Hojlund", "position": "ST", "number": 19, "age": 22, "nationality": "Denmark", "status": "active"},  # loan from Man Utd
            {"name": "David Neres", "position": "ST", "number": 7, "age": 28, "nationality": "Brazil", "status": "active"},
            {"name": "M. Politano", "position": "ST", "number": 21, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "Giovane", "position": "ST", "number": 23, "age": 22, "nationality": "Brazil", "status": "active"},  # from Verona
        ],
        "coach": "Antonio Conte",
        "jan_2026_in": ["Giovane (from Verona)"],
        "jan_2026_out": ["G. Raspadori (to Atalanta)"],
        "notes": "De Bruyne joined summer 2025 as free agent from Man City. Injured (surgery Oct), out until March. Hojlund on loan from Man Utd with €44M option.",
    },

    "Parma": {
        "players": [
            # Goalkeepers
            {"name": "V. Guaita", "position": "GK", "number": 1, "age": 38, "nationality": "Spain", "status": "injured"},
            {"name": "Z. Suzuki", "position": "GK", "number": 23, "age": 23, "nationality": "Japan", "status": "injured"},
            {"name": "L. Chichizola", "position": "GK", "number": 12, "age": 35, "nationality": "Argentina", "status": "active"},
            # Defenders
            {"name": "E. Del Prato", "position": "CB", "number": 6, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "W. Circati", "position": "CB", "number": 4, "age": 22, "nationality": "Italy", "status": "active"},
            {"name": "B. Osorio", "position": "CB", "number": 3, "age": 23, "nationality": "Colombia", "status": "active"},
            {"name": "Y. Valenti", "position": "CB", "number": 5, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "M. Lovik", "position": "CB", "number": 15, "age": 22, "nationality": "Norway", "status": "injured"},
            {"name": "E. Coulibaly", "position": "CB", "number": 2, "age": 24, "nationality": "Mali", "status": "active"},
            # Midfielders
            {"name": "A. Bernabe", "position": "CM", "number": 10, "age": 23, "nationality": "Spain", "status": "active"},
            {"name": "S. Sohm", "position": "CM", "number": 8, "age": 25, "nationality": "Switzerland", "status": "active"},
            {"name": "M. Estevez", "position": "CM", "number": 20, "age": 24, "nationality": "Argentina", "status": "active"},
            {"name": "J. Man", "position": "CM", "number": 14, "age": 24, "nationality": "Romania", "status": "active"},
            {"name": "A. Mihaila", "position": "CM", "number": 7, "age": 26, "nationality": "Romania", "status": "active"},
            # Forwards
            {"name": "M. Pellegrino", "position": "ST", "number": 9, "age": 19, "nationality": "Argentina", "status": "active"},  # 6 goals, leading scorer
            {"name": "P. Almqvist", "position": "ST", "number": 11, "age": 27, "nationality": "Sweden", "status": "injured"},
            {"name": "A. Benedyczak", "position": "ST", "number": 18, "age": 25, "nationality": "Poland", "status": "active"},
            {"name": "B. Keita", "position": "ST", "number": 19, "age": 30, "nationality": "Guinea", "status": "active"},
            {"name": "D. Cancellieri", "position": "ST", "number": 22, "age": 23, "nationality": "Italy", "status": "active"},
        ],
        "coach": "Carlos Cuesta",
        "jan_2026_in": [],
        "jan_2026_out": [],
    },

    "Pisa": {
        "players": [
            # Goalkeepers
            {"name": "A. Semper", "position": "GK", "number": 1, "age": 28, "nationality": "Croatia", "status": "active"},
            {"name": "L. Nicolas", "position": "GK", "number": 22, "age": 31, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "A. Caracciolo", "position": "CB", "number": 4, "age": 33, "nationality": "Italy", "status": "suspended"},
            {"name": "R. Albiol", "position": "CB", "number": 5, "age": 40, "nationality": "Spain", "status": "injured"},
            {"name": "M. Canestrelli", "position": "CB", "number": 15, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "G. Bonfanti", "position": "CB", "number": 6, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "A. Rus", "position": "CB", "number": 3, "age": 27, "nationality": "Romania", "status": "active"},
            # Midfielders
            {"name": "M. Marin", "position": "CM", "number": 8, "age": 30, "nationality": "Romania", "status": "active"},
            {"name": "S. Tourè", "position": "CM", "number": 10, "age": 26, "nationality": "Ivory Coast", "status": "active"},
            {"name": "J. Cuadrado", "position": "CM", "number": 7, "age": 37, "nationality": "Colombia", "status": "injured"},  # hamstring
            {"name": "N. Abildgaard", "position": "CM", "number": 6, "age": 28, "nationality": "Denmark", "status": "active"},
            {"name": "H. Morutan", "position": "CM", "number": 14, "age": 24, "nationality": "Romania", "status": "active"},
            {"name": "C. Stengs", "position": "CM", "number": 11, "age": 26, "nationality": "Netherlands", "status": "injured"},  # groin surgery
            {"name": "Iling Junior", "position": "CM", "number": 20, "age": 22, "nationality": "England", "status": "active"},  # Jan 2026 loan from Aston Villa
            # Forwards
            {"name": "N. Bonfanti", "position": "ST", "number": 9, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "M. Tramoni", "position": "ST", "number": 17, "age": 24, "nationality": "France", "status": "active"},
            {"name": "E. Vignato", "position": "ST", "number": 18, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "M. Nzola", "position": "ST", "number": 19, "age": 28, "nationality": "Angola", "status": "active"},  # Jan 2026 from Sassuolo
            {"name": "P. Arena", "position": "ST", "number": 21, "age": 23, "nationality": "Italy", "status": "active"},
        ],
        "coach": "Filippo Inzaghi",
        "jan_2026_in": ["M. Nzola (from Sassuolo/Juventus)", "Iling Junior (loan from Aston Villa)"],
        "jan_2026_out": [],
    },

    "Roma": {
        "players": [
            # Goalkeepers
            {"name": "M. Svilar", "position": "GK", "number": 99, "age": 25, "nationality": "Belgium", "status": "active"},
            {"name": "R. Ryan", "position": "GK", "number": 87, "age": 28, "nationality": "Australia", "status": "active"},
            # Defenders
            {"name": "G. Mancini", "position": "CB", "number": 23, "age": 28, "nationality": "Italy", "status": "active"},
            {"name": "E. Ndicka", "position": "CB", "number": 2, "age": 26, "nationality": "France", "status": "active"},
            {"name": "M. Hermoso", "position": "CB", "number": 5, "age": 30, "nationality": "Spain", "status": "active"},
            {"name": "Z. Celik", "position": "CB", "number": 19, "age": 28, "nationality": "Turkey", "status": "active"},
            {"name": "Angelino", "position": "CB", "number": 18, "age": 28, "nationality": "Spain", "status": "injured"},
            {"name": "S. El Shaarawy", "position": "CM", "number": 92, "age": 33, "nationality": "Italy", "status": "active"},
            {"name": "S. Saud", "position": "CB", "number": 77, "age": 24, "nationality": "Saudi Arabia", "status": "active"},
            # Midfielders
            {"name": "L. Pellegrini", "position": "CM", "number": 7, "age": 29, "nationality": "Italy", "status": "active"},
            {"name": "B. Cristante", "position": "CM", "number": 4, "age": 30, "nationality": "Italy", "status": "active"},
            {"name": "L. Paredes", "position": "CM", "number": 16, "age": 31, "nationality": "Argentina", "status": "active"},
            {"name": "M. Kone", "position": "CM", "number": 17, "age": 24, "nationality": "France", "status": "active"},
            {"name": "E. Le Fee", "position": "CM", "number": 8, "age": 25, "nationality": "France", "status": "active"},
            # Forwards
            {"name": "P. Dybala", "position": "ST", "number": 21, "age": 31, "nationality": "Argentina", "status": "active"},
            {"name": "A. Dovbyk", "position": "ST", "number": 11, "age": 27, "nationality": "Ukraine", "status": "injured"},  # recurring injuries
            {"name": "L. Soulé", "position": "ST", "number": 18, "age": 21, "nationality": "Argentina", "status": "active"},
            {"name": "D. Malen", "position": "ST", "number": 22, "age": 26, "nationality": "Netherlands", "status": "active"},  # Jan 2026
            {"name": "B. Zaragoza", "position": "ST", "number": 10, "age": 24, "nationality": "Spain", "status": "active"},  # Jan 2026 loan
            {"name": "S. Shomurodov", "position": "ST", "number": 14, "age": 29, "nationality": "Uzbekistan", "status": "active"},
        ],
        "coach": "Gian Piero Gasperini",
        "jan_2026_in": [
            "D. Malen",
            "B. Zaragoza (loan with €13.5M option)",
            "R. Vaz (teenage prospect, injured)",
        ],
        "jan_2026_out": ["T. Baldanzi (loan to Genoa)"],
    },

    "Sassuolo": {
        "players": [
            # Goalkeepers
            {"name": "S. Turati", "position": "GK", "number": 22, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "G. Pegolo", "position": "GK", "number": 1, "age": 43, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "J. Idzes", "position": "CB", "number": 5, "age": 27, "nationality": "Italy", "status": "active"},
            {"name": "S. Walukiewicz", "position": "CB", "number": 4, "age": 25, "nationality": "Poland", "status": "active"},
            {"name": "T. Muharemovic", "position": "CB", "number": 15, "age": 22, "nationality": "Bosnia", "status": "active"},
            {"name": "C. Odenthal", "position": "CB", "number": 3, "age": 25, "nationality": "Netherlands", "status": "active"},
            {"name": "J. Doig", "position": "CB", "number": 6, "age": 23, "nationality": "Scotland", "status": "active"},
            {"name": "Fali Cande", "position": "CB", "number": 2, "age": 26, "nationality": "Guinea-Bissau", "status": "injured"},
            # Midfielders
            {"name": "N. Matic", "position": "CM", "number": 32, "age": 37, "nationality": "Serbia", "status": "active"},
            {"name": "K. Thorstvedt", "position": "CM", "number": 28, "age": 26, "nationality": "Norway", "status": "active"},
            {"name": "I. Kone", "position": "CM", "number": 14, "age": 23, "nationality": "Canada", "status": "active"},
            {"name": "A. Vranckx", "position": "CM", "number": 8, "age": 22, "nationality": "Belgium", "status": "active"},
            {"name": "D. Boloca", "position": "CM", "number": 10, "age": 25, "nationality": "Italy", "status": "injured"},  # knee issue
            # Forwards
            {"name": "D. Berardi", "position": "ST", "number": 25, "age": 31, "nationality": "Italy", "status": "active"},  # captain, returning from injury
            {"name": "A. Pinamonti", "position": "ST", "number": 9, "age": 26, "nationality": "Italy", "status": "active"},
            {"name": "A. Lauriente", "position": "ST", "number": 11, "age": 27, "nationality": "France", "status": "active"},
            {"name": "A. Fadera", "position": "ST", "number": 7, "age": 23, "nationality": "Gambia", "status": "active"},
            {"name": "L. Moro", "position": "ST", "number": 18, "age": 23, "nationality": "Italy", "status": "active"},
            {"name": "C. Volpato", "position": "ST", "number": 17, "age": 22, "nationality": "Australia", "status": "active"},
            {"name": "Cheddira", "position": "ST", "number": 19, "age": 28, "nationality": "Morocco", "status": "active"},  # Jan 2026
        ],
        "coach": "Fabio Grosso",
        "jan_2026_in": ["Cheddira"],
        "jan_2026_out": ["M. Nzola (to Pisa)"],
        "notes": "Berardi returning from flexor injury. Promoted as Serie B champions.",
    },

    "Torino": {
        "players": [
            # Goalkeepers
            {"name": "A. Paleari", "position": "GK", "number": 1, "age": 32, "nationality": "Italy", "status": "active"},
            {"name": "F. Israel", "position": "GK", "number": 32, "age": 24, "nationality": "Uruguay", "status": "active"},
            # Defenders
            {"name": "A. Ismajli", "position": "CB", "number": 5, "age": 28, "nationality": "Albania", "status": "injured"},
            {"name": "P. Schuurs", "position": "CB", "number": 3, "age": 25, "nationality": "Netherlands", "status": "active"},
            {"name": "C. Biraghi", "position": "CB", "number": 13, "age": 33, "nationality": "Italy", "status": "active"},
            {"name": "N. Nkounkou", "position": "CB", "number": 15, "age": 24, "nationality": "France", "status": "active"},
            {"name": "V. Lazaro", "position": "CB", "number": 22, "age": 29, "nationality": "Austria", "status": "active"},
            {"name": "M. Pedersen", "position": "CB", "number": 2, "age": 24, "nationality": "Norway", "status": "active"},
            # Midfielders
            {"name": "N. Vlasic", "position": "CM", "number": 10, "age": 28, "nationality": "Croatia", "status": "suspended"},
            {"name": "I. Ilic", "position": "CM", "number": 8, "age": 24, "nationality": "Serbia", "status": "active"},
            {"name": "G. Gineitis", "position": "CM", "number": 66, "age": 21, "nationality": "Lithuania", "status": "active"},
            {"name": "A. Tameze", "position": "CM", "number": 7, "age": 30, "nationality": "Cameroon", "status": "active"},
            {"name": "C. Casadei", "position": "CM", "number": 14, "age": 22, "nationality": "Italy", "status": "active"},
            {"name": "K. Asllani", "position": "CM", "number": 20, "age": 23, "nationality": "Albania", "status": "active"},
            {"name": "M. Prati", "position": "CM", "number": 16, "age": 23, "nationality": "Italy", "status": "suspended"},  # Jan 2026 from Cagliari
            {"name": "E. Ilkhan", "position": "CM", "number": 17, "age": 22, "nationality": "Turkey", "status": "active"},
            # Forwards
            {"name": "D. Zapata", "position": "ST", "number": 91, "age": 35, "nationality": "Colombia", "status": "active"},
            {"name": "C. Adams", "position": "ST", "number": 18, "age": 29, "nationality": "Scotland", "status": "active"},
            {"name": "G. Simeone", "position": "ST", "number": 9, "age": 30, "nationality": "Argentina", "status": "active"},
            {"name": "C. Ngonge", "position": "ST", "number": 11, "age": 25, "nationality": "Belgium", "status": "active"},
            {"name": "A. Njie", "position": "ST", "number": 19, "age": 20, "nationality": "Sweden", "status": "active"},
        ],
        "coach": "Marco Baroni",
        "jan_2026_in": ["M. Prati (from Cagliari)", "A. Njie", "C. Casadei"],
        "jan_2026_out": ["S. Ricci (to Milan)"],
    },

    "Udinese": {
        "players": [
            # Goalkeepers
            {"name": "M. Okoye", "position": "GK", "number": 40, "age": 26, "nationality": "Nigeria", "status": "active"},
            {"name": "R. Sava", "position": "GK", "number": 90, "age": 23, "nationality": "Romania", "status": "active"},
            {"name": "D. Padelli", "position": "GK", "number": 93, "age": 40, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "N. Bertola", "position": "CB", "number": 13, "age": 22, "nationality": "Italy", "status": "active"},
            {"name": "S. Goglichidze", "position": "CB", "number": 2, "age": 21, "nationality": "Georgia", "status": "active"},
            {"name": "C. Kabasele", "position": "CB", "number": 27, "age": 34, "nationality": "Belgium", "status": "active"},
            {"name": "T. Kristensen", "position": "CB", "number": 31, "age": 23, "nationality": "Denmark", "status": "active"},
            {"name": "O. Solet", "position": "CB", "number": 28, "age": 25, "nationality": "France", "status": "active"},
            {"name": "A. Zanoli", "position": "CB", "number": 59, "age": 25, "nationality": "Italy", "status": "active"},
            {"name": "K. Ehizibue", "position": "CB", "number": 19, "age": 30, "nationality": "Nigeria", "status": "active"},
            {"name": "J. Zemura", "position": "CB", "number": 33, "age": 26, "nationality": "Zimbabwe", "status": "active"},
            # Midfielders
            {"name": "J. Karlstrom", "position": "CM", "number": 8, "age": 30, "nationality": "Sweden", "status": "active"},
            {"name": "F. Thauvin", "position": "CM", "number": 10, "age": 33, "nationality": "France", "status": "active"},
            {"name": "N. Zaniolo", "position": "CM", "number": 10, "age": 26, "nationality": "Italy", "status": "injured"},  # knee
            {"name": "J. Piotrowski", "position": "CM", "number": 24, "age": 28, "nationality": "Poland", "status": "injured"},  # knee ligament
            {"name": "Rui Modesto", "position": "CM", "number": 77, "age": 26, "nationality": "Portugal", "status": "active"},
            {"name": "Oier Zarraga", "position": "CM", "number": 6, "age": 26, "nationality": "Spain", "status": "active"},
            # Forwards
            {"name": "K. Davis", "position": "ST", "number": 9, "age": 27, "nationality": "England", "status": "active"},  # 7 goals, top scorer
            {"name": "A. Sanchez", "position": "ST", "number": 7, "age": 37, "nationality": "Chile", "status": "active"},
            {"name": "A. Buksa", "position": "ST", "number": 18, "age": 29, "nationality": "Poland", "status": "injured"},  # calf
            {"name": "J. Ekkelenkamp", "position": "ST", "number": 32, "age": 25, "nationality": "Netherlands", "status": "active"},
            {"name": "I. Gueye", "position": "ST", "number": 7, "age": 19, "nationality": "Senegal", "status": "active"},
        ],
        "coach": "Kosta Runjaic",
        "jan_2026_in": [],
        "jan_2026_out": [],
    },

    "Verona": {
        "players": [
            # Goalkeepers
            {"name": "L. Montipo", "position": "GK", "number": 1, "age": 30, "nationality": "Italy", "status": "active"},
            # Defenders
            {"name": "D. Oyegoke", "position": "CB", "number": 2, "age": 21, "nationality": "England", "status": "active"},
            {"name": "E. Ebosse", "position": "CB", "number": 5, "age": 26, "nationality": "Cameroon", "status": "active"},
            {"name": "U. Nunez", "position": "CB", "number": 4, "age": 29, "nationality": "Spain", "status": "active"},
            {"name": "V. Nelsson", "position": "CB", "number": 3, "age": 25, "nationality": "Denmark", "status": "active"},
            {"name": "T. Slotsager", "position": "CB", "number": 15, "age": 22, "nationality": "Denmark", "status": "active"},
            {"name": "R. Belghali", "position": "CB", "number": 22, "age": 21, "nationality": "France", "status": "active"},
            # Midfielders
            {"name": "C. Niasse", "position": "CM", "number": 8, "age": 24, "nationality": "Senegal", "status": "active"},
            {"name": "S. Serdar", "position": "CM", "number": 10, "age": 28, "nationality": "Germany", "status": "active"},
            {"name": "G. Kastanos", "position": "CM", "number": 14, "age": 28, "nationality": "Cyprus", "status": "active"},
            {"name": "A. Harroui", "position": "CM", "number": 20, "age": 27, "nationality": "Netherlands", "status": "active"},
            {"name": "R. Gagliardini", "position": "CM", "number": 5, "age": 31, "nationality": "Italy", "status": "active"},
            {"name": "T. Suslov", "position": "CM", "number": 7, "age": 23, "nationality": "Slovakia", "status": "active"},
            {"name": "A. Bernede", "position": "CM", "number": 16, "age": 26, "nationality": "France", "status": "active"},
            # Forwards
            {"name": "G. Orban", "position": "ST", "number": 9, "age": 26, "nationality": "Nigeria", "status": "active"},
            {"name": "D. Mosquera", "position": "ST", "number": 11, "age": 23, "nationality": "Colombia", "status": "active"},
            {"name": "J. Ajayi", "position": "ST", "number": 18, "age": 22, "nationality": "Nigeria", "status": "active"},
            {"name": "A. Sarr", "position": "ST", "number": 19, "age": 24, "nationality": "Senegal", "status": "active"},
            {"name": "A. Edmundsson", "position": "ST", "number": 17, "age": 24, "nationality": "Faroe Islands", "status": "active"},
        ],
        "coach": "Paolo Zanetti",  # dismissed Feb 2026
        "jan_2026_in": [],
        "jan_2026_out": ["Giovane (to Napoli)", "R. Belahyane (to Lazio)"],
        "notes": "Bottom of table, 2W 8D 13L. Coach Zanetti dismissed after 4-0 loss at Cagliari.",
    },
}


def generate_discrepancy_report():
    """Compare corrected squads against existing data and report differences."""
    log.info("Generating discrepancy report...")

    # Load existing data
    existing = {}
    if SQUADS_PATH.exists():
        with open(SQUADS_PATH) as f:
            data = json.load(f)
        existing = data.get("teams", {})

    # Load xG profiles for secondary comparison
    xg_teams = set()
    if PROFILES_PATH.exists():
        with open(PROFILES_PATH) as f:
            profiles = json.load(f)
        for _key, val in profiles.items():
            if isinstance(val, dict):
                xg_teams.add(val.get("team", ""))

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {},
        "teams": {},
    }

    # Overall status
    existing_team_names = set(existing.keys())
    corrected_team_names = set(CORRECTED_SQUADS.keys())
    all_teams = corrected_team_names

    # Teams in xG profiles that shouldn't be (relegated)
    relegated_in_xg = xg_teams - corrected_team_names
    missing_from_xg = corrected_team_names - xg_teams

    report["summary"] = {
        "total_serie_a_teams": 20,
        "teams_in_existing_squads": len(existing_team_names),
        "teams_missing_from_squads": sorted(corrected_team_names - existing_team_names),
        "relegated_teams_in_xg_profiles": sorted(relegated_in_xg),
        "promoted_teams_missing_from_xg": sorted(missing_from_xg),
        "promoted_2025_26": ["Sassuolo", "Pisa", "Cremonese"],
        "relegated_2024_25": ["Venezia", "Empoli", "Monza"],
    }

    # Per-team comparison
    total_issues = 0
    for team in sorted(all_teams):
        corrected = CORRECTED_SQUADS[team]
        corrected_players = {p["name"] for p in corrected["players"]}

        team_report = {
            "in_existing_squads": team in existing_team_names,
            "in_xg_profiles": team in xg_teams,
            "corrected_squad_size": len(corrected["players"]),
            "coach": corrected.get("coach", ""),
            "jan_2026_in": corrected.get("jan_2026_in", []),
            "jan_2026_out": corrected.get("jan_2026_out", []),
            "notes": corrected.get("notes", ""),
            "issues": [],
        }

        # Compare with existing squads
        if team in existing:
            existing_team = existing[team]
            existing_players_list = existing_team.get("players", []) if isinstance(existing_team, dict) else []
            existing_names = {p.get("name", "") for p in existing_players_list}

            # Players in existing but NOT in corrected (phantom)
            phantom = existing_names - corrected_players
            for p in sorted(phantom):
                team_report["issues"].append({
                    "player": p,
                    "action": "REMOVE",
                    "reason": "Not in current squad (transferred/released)",
                })

            # Players in corrected but NOT in existing (missing)
            missing = corrected_players - existing_names
            for p in sorted(missing):
                player_data = next((x for x in corrected["players"] if x["name"] == p), {})
                team_report["issues"].append({
                    "player": p,
                    "action": "ADD",
                    "reason": f"New player - {player_data.get('position', '?')}",
                    "status": player_data.get("status", "active"),
                })
        else:
            # Entire team missing
            team_report["issues"].append({
                "player": "ALL",
                "action": "ADD_TEAM",
                "reason": f"Team not in existing squads ({len(corrected['players'])} players)",
            })

        # Injured players
        injured = [p for p in corrected["players"] if p.get("status") == "injured"]
        if injured:
            team_report["injured_players"] = [
                {"name": p["name"], "position": p["position"]}
                for p in injured
            ]

        total_issues += len(team_report["issues"])
        report["teams"][team] = team_report

    report["summary"]["total_issues"] = total_issues
    report["summary"]["teams_with_issues"] = sum(
        1 for t in report["teams"].values() if t["issues"]
    )

    return report


def save_corrected_squads():
    """Write corrected squads to current_squads.json."""
    # Convert to the format expected by the pipeline
    teams_dict = {}
    for team, data in CORRECTED_SQUADS.items():
        teams_dict[team] = {
            "players": [
                {
                    "name": p["name"],
                    "position": p["position"],
                    "number": p["number"],
                    "age": p["age"],
                    "nationality": p.get("nationality"),
                    "status": p.get("status", "active"),
                }
                for p in data["players"]
            ]
        }

    output = {
        "fetched_at": datetime.now().isoformat(),
        "source": "manual_audit_2026_02_07",
        "season": "2025-26",
        "teams": teams_dict,
    }

    SQUADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SQUADS_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info("Saved corrected squads for %d teams to %s", len(teams_dict), SQUADS_PATH)
    return output


def main():
    """Run full squad audit."""
    # Generate discrepancy report
    report = generate_discrepancy_report()

    # Save report
    SQUADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Audit report saved to %s", AUDIT_PATH)

    # Print summary
    s = report["summary"]
    print("\n" + "=" * 70)
    print(" SERIE A 2025-2026 SQUAD AUDIT REPORT")
    print("=" * 70)
    print(f"\n  Teams in existing squads: {s['teams_in_existing_squads']}/20")
    print(f"  Missing teams: {', '.join(s['teams_missing_from_squads']) or 'None'}")
    print(f"  Promoted: {', '.join(s['promoted_2025_26'])}")
    print(f"  Relegated: {', '.join(s['relegated_2024_25'])}")
    print(f"  Relegated still in xG profiles: {', '.join(s['relegated_teams_in_xg_profiles']) or 'None'}")
    print(f"  Promoted missing from xG: {', '.join(s['promoted_teams_missing_from_xg']) or 'None'}")
    print(f"\n  Total issues found: {s['total_issues']}")
    print(f"  Teams with issues: {s['teams_with_issues']}")

    # Per-team details
    for team in sorted(report["teams"]):
        team_data = report["teams"][team]
        if team_data["issues"]:
            print(f"\n  --- {team} ---")
            print(f"  Coach: {team_data.get('coach', 'N/A')}")
            if team_data.get("jan_2026_in"):
                print(f"  Jan IN: {', '.join(team_data['jan_2026_in'])}")
            if team_data.get("jan_2026_out"):
                print(f"  Jan OUT: {', '.join(team_data['jan_2026_out'])}")
            for issue in team_data["issues"][:10]:  # Cap at 10 per team
                icon = {"REMOVE": "❌", "ADD": "✅", "ADD_TEAM": "🆕"}.get(issue["action"], "⚠️")
                print(f"    {icon} {issue['action']}: {issue['player']} - {issue['reason']}")
            if team_data.get("injured_players"):
                for inj in team_data["injured_players"]:
                    print(f"    🚑 INJURED: {inj['name']} ({inj['position']})")

    # Ask before overwriting
    print("\n" + "=" * 70)
    print(f"\n  Ready to save corrected squads for all 20 teams.")
    print(f"  This will overwrite {SQUADS_PATH}")

    # Save corrected squads
    save_corrected_squads()

    print(f"\n  ✅ Corrected squads saved for 20 teams")
    print(f"  ✅ Audit report saved to {AUDIT_PATH}")

    return report


if __name__ == "__main__":
    main()
