import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# KONFIGURACJA STRONY
# ------------------------------
st.set_page_config(page_title="AI Football Analyst Pro", layout="wide")
st.title("🧠 Zaawansowany Analityk Piłkarski + Gemini + Kupony")
st.caption("Football-Data.org (wyniki) + API-Football (składy) | Model Poissona | Raport AI")

# ------------------------------
# 1. MAPY I ŚREDNIE LIGOWE
# ------------------------------
LEAGUE_MAP_FD = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "Ligue 1": "FL1",
}

LEAGUE_MAP_API = {
    "Premier League": 39,
    "La Liga": 140,
    "Bundesliga": 78,
    "Serie A": 135,
    "Ligue 1": 61,
}

LEAGUE_FACTORS = {
    "Premier League": 1.0, "La Liga": 0.95, "Bundesliga": 1.1,
    "Serie A": 0.9, "Ligue 1": 0.98
}

LEAGUE_AVERAGES = {
    "Premier League": {"cards": 3.8, "corners": 10.2},
    "La Liga": {"cards": 4.2, "corners": 9.5},
    "Bundesliga": {"cards": 3.5, "corners": 10.8},
    "Serie A": {"cards": 4.5, "corners": 10.0},
    "Ligue 1": {"cards": 3.9, "corners": 9.8},
}

# Lista kluczowych graczy (przykład)
KEY_PLAYERS = {
    "Paris Saint-Germain": ["Kylian Mbappé", "Ousmane Dembélé", "Marquinhos", "Gianluigi Donnarumma"],
    "Olympique Lyonnais": ["Alexandre Lacazette", "Rayan Cherki", "Maxence Caqueret"],
    "Manchester City": ["Erling Haaland", "Kevin De Bruyne", "Rodri", "Phil Foden"],
    "Arsenal": ["Bukayo Saka", "Martin Ødegaard", "Declan Rice", "William Saliba"],
    "Real Madrid": ["Vinícius Júnior", "Jude Bellingham", "Kylian Mbappé", "Antonio Rüdiger"],
    "FC Barcelona": ["Robert Lewandowski", "Lamine Yamal", "Pedri", "Ronald Araújo"],
    "Bayern Munich": ["Harry Kane", "Jamal Musiala", "Joshua Kimmich"],
    "Borussia Dortmund": ["Serhou Guirassy", "Julian Brandt", "Nico Schlotterbeck"],
    "Juventus": ["Dušan Vlahović", "Kenan Yıldız", "Gleison Bremer"],
    "Inter": ["Lautaro Martínez", "Nicolò Barella", "Alessandro Bastoni"],
    "AC Milan": ["Rafael Leão", "Christian Pulisic", "Theo Hernández"],
}

# ------------------------------
# 2. FUNKCJE POBIERANIA DANYCH
# ------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_team_stats_fd(team_name, league_code, api_key):
    """Pobiera średnie bramki z ostatnich 5 meczów ligowych (Football-Data.org)"""
    if not api_key:
        return None
    headers = {'X-Auth-Token': api_key}
    teams_url = f"https://api.football-data.org/v4/competitions/{league_code}/teams"
    try:
        resp = requests.get(teams_url, headers=headers)
        resp.raise_for_status()
        teams = resp.json()['teams']
        team_id = next((t['id'] for t in teams if team_name.lower() in t['name'].lower()), None)
        if not team_id:
            st.warning(f"Nie znaleziono {team_name} w Football-Data.org")
            return None
    except Exception as e:
        st.error(f"Błąd pobierania listy drużyn: {e}")
        return None

    matches_url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
    params = {'limit': 20, 'status': 'FINISHED'}
    try:
        resp = requests.get(matches_url, headers=headers, params=params)
        resp.raise_for_status()
        matches = resp.json()['matches']
        league_matches = [m for m in matches if m['competition']['code'] == league_code]
        if len(league_matches) < 5:
            st.warning(f"Znaleziono tylko {len(league_matches)} meczów dla {team_name}")
            if not league_matches:
                return None
        recent = league_matches[-5:]
        goals_for, goals_against = [], []
        df_matches = []
        for m in recent:
            is_home = m['homeTeam']['id'] == team_id
            gf = m['score']['fullTime']['home'] if is_home else m['score']['fullTime']['away']
            ga = m['score']['fullTime']['away'] if is_home else m['score']['fullTime']['home']
            if gf is not None and ga is not None:
                goals_for.append(gf)
                goals_against.append(ga)
            df_matches.append({
                'date': m['utcDate'][:10],
                'opponent': m['awayTeam']['name'] if is_home else m['homeTeam']['name'],
                'gf': gf,
                'ga': ga
            })
        return {
            'avg_gf': round(np.mean(goals_for), 2) if goals_for else 1.4,
            'avg_ga': round(np.mean(goals_against), 2) if goals_against else 1.4,
            'recent_form': pd.DataFrame(df_matches)
        }
    except Exception as e:
        st.error(f"Błąd pobierania meczów: {e}")
        return None

def get_upcoming_fixture(team_a, team_b, league_id, api_key):
    """Znajduje nadchodzący mecz między dwiema drużynami (API-Football)"""
    if not api_key:
        return None
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {'x-apisports-key': api_key}
    today = datetime.now().strftime('%Y-%m-%d')
    next_week = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    params = {
        'league': league_id,
        'season': datetime.now().year,
        'from': today,
        'to': next_week
    }
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        for fix in data.get('response', []):
            home = fix['teams']['home']['name']
            away = fix['teams']['away']['name']
            if (team_a.lower() in home.lower() or team_a.lower() in away.lower()) and \
               (team_b.lower() in home.lower() or team_b.lower() in away.lower()):
                return fix['fixture']['id']
        return None
    except Exception as e:
        st.warning(f"Nie udało się pobrać ID meczu: {e}")
        return None

def fetch_lineup(fixture_id, team_name, api_key):
    """Pobiera skład drużyny na dany mecz (API-Football)"""
    if not fixture_id or not api_key:
        return None
    url = "https://v3.football.api-sports.io/fixtures/lineups"
    headers = {'x-apisports-key': api_key}
    params = {'fixture': fixture_id}
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        for lu in data.get('response', []):
            if team_name.lower() in lu['team']['name'].lower():
                return lu
        return None
    except:
        return None

def calculate_lineup_strength(lineup, key_players):
    """Oblicza siłę składu (0.5-1.0) na podstawie obecności kluczowych graczy"""
    if not lineup or not key_players:
        return 1.0
    starters = [p['player']['name'] for p in lineup['startXI']]
    present = sum(1 for kp in key_players if any(kp.lower() in s.lower() for s in starters))
    strength = 1.0 - (len(key_players) - present) * 0.1
    return max(0.5, min(1.0, strength))

# ------------------------------
# 3. FUNKCJE OBLICZENIOWE (POISSON)
# ------------------------------
def calculate_match_probs(lambda_home, lambda_away):
    home_win = draw = away_win = 0
    for i in range(10):
        for j in range(10):
            prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            if i > j: home_win += prob
            elif i == j: draw += prob
            else: away_win += prob
    return {"1": round(home_win*100,1), "X": round(draw*100,1), "2": round(away_win*100,1)}

def calculate_goal_markets(lambda_home, lambda_away):
    total = lambda_home + lambda_away
    return {
        'over_0.5': round((1-poisson.cdf(0,total))*100,1),
        'over_1.5': round((1-poisson.cdf(1,total))*100,1),
        'over_2.5': round((1-poisson.cdf(2,total))*100,1),
        'over_3.5': round((1-poisson.cdf(3,total))*100,1),
        'over_4.5': round((1-poisson.cdf(4,total))*100,1),
    }

def calculate_handicap(lambda_home, lambda_away, handicap=-1.5):
    prob = 0
    for i in range(10):
        for j in range(10):
            if i + handicap > j:
                prob += poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    return round(prob*100,1)

def simulate_cards_corners(style_home, style_away, league):
    avg = LEAGUE_AVERAGES.get(league, {"cards":3.8, "corners":10.0})
    lambda_cards = avg["cards"] * style_home * style_away
    lambda_corners = avg["corners"] * style_home * style_away
    card_probs = {
        'over_3.5_cards': round((1-poisson.cdf(3, lambda_cards))*100,1),
        'over_4.5_cards': round((1-poisson.cdf(4, lambda_cards))*100,1),
        'over_5.5_cards': round((1-poisson.cdf(5, lambda_cards))*100,1),
    }
    corner_probs = {
        'over_8.5_corners': round((1-poisson.cdf(8, lambda_corners))*100,1),
        'over_9.5_corners': round((1-poisson.cdf(9, lambda_corners))*100,1),
        'over_10.5_corners': round((1-poisson.cdf(10, lambda_corners))*100,1),
    }
    return card_probs, corner_probs

def generate_coupons(probs, goal_markets, card_probs, corner_probs, handicap_prob, threshold=60):
    suggestions = []
    if probs['1'] >= threshold: suggestions.append(("1X2 - Gospodarz", probs['1']))
    if probs['X'] >= threshold: suggestions.append(("1X2 - Remis", probs['X']))
    if probs['2'] >= threshold: suggestions.append(("1X2 - Gość", probs['2']))
    for k, v in goal_markets.items():
        if v >= threshold:
            label = k.replace('_', ' ').replace('over', 'Over').replace('.5', '.5 gola')
            suggestions.append((label, v))
    if handicap_prob >= threshold:
        suggestions.append(("Handicap -1.5 (Gospodarz)", handicap_prob))
    for k, v in card_probs.items():
        if v >= threshold:
            label = k.replace('_', ' ').replace('over', 'Over').replace('cards', 'kartek')
            suggestions.append((label, v))
    for k, v in corner_probs.items():
        if v >= threshold:
            label = k.replace('_', ' ').replace('over', 'Over').replace('corners', 'rożnych')
            suggestions.append((label, v))
    return sorted(suggestions, key=lambda x: x[1], reverse=True)[:6]

# ------------------------------
# 4. GENEROWANIE RAPORTU AI (GEMINI)
# ------------------------------
def get_available_model(api_key):
    genai.configure(api_key=api_key)
    try:
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                return model.name
    except:
        pass
    return "models/gemini-2.0-flash-exp"

def generate_llm_report(team_a, team_b, stats_a, stats_b, probs, goal_markets, league, api_key):
    if not api_key:
        return None
    model_name = get_available_model(api_key)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = f"""
Jesteś ekspertem od analityki piłkarskiej. Na podstawie poniższych danych przeprowadź analizę meczu {team_a} vs {team_b} w lidze {league}.

Statystyki z ostatnich 5 meczów:
{team_a}: średnia bramek zdobytych: {stats_a['avg_gf']}, straconych: {stats_a['avg_ga']}
{team_b}: średnia bramek zdobytych: {stats_b['avg_gf']}, straconych: {stats_b['avg_ga']}

Model Poissona:
- 1: {probs['1']}%, X: {probs['X']}%, 2: {probs['2']}%
- Over 2.5 gola: {goal_markets['over_2.5']}%

Napisz zwięzły raport w języku polskim zawierający:
1. Profil taktyczny (jak drużyny na siebie oddziałują)
2. Dominujący scenariusz meczu
3. Margines błędu (czynnik ryzyka)

Używaj stylu profesjonalnego, opartego na danych.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Błąd generowania raportu: {e}"

# ------------------------------
# 5. INTERFEJS UŻYTKOWNIKA
# ------------------------------
with st.sidebar:
    st.header("🔑 Klucze API")
    fd_key = st.text_input("Football-Data.org Key", type="password")
    api_key = st.text_input("API-Football Key (opcjonalny)", type="password")
    gemini_key = st.text_input("Google Gemini Key (opcjonalny)", type="password")
    st.caption("[Football-Data](https://www.football-data.org/) | [API-Football](https://www.api-football.com/) | [Gemini](https://aistudio.google.com/)")
    st.divider()
    st.header("⚙️ Styl gry")
    style_h = st.slider("Agresywność gospodarzy", 0.7, 1.3, 1.0, 0.05)
    style_a = st.slider("Agresywność gości", 0.7, 1.3, 1.0, 0.05)
    st.divider()
    threshold = st.slider("Próg pewności kuponów (%)", 50, 90, 65, 5)

col1, col2 = st.columns(2)
with col1:
    team_a = st.text_input("🏠 DRUŻYNA A", "Paris Saint-Germain")
with col2:
    team_b = st.text_input("✈️ DRUŻYNA B", "Olympique Lyonnais")

league = st.selectbox("🏆 Liga", list(LEAGUE_MAP_FD.keys()))

if st.button("🚀 Analizuj mecz", type="primary"):
    if not fd_key:
        st.error("❌ Podaj klucz Football-Data.org")
    else:
        with st.spinner("Pobieram dane..."):
            stats_a = fetch_team_stats_fd(team_a, LEAGUE_MAP_FD[league], fd_key)
            stats_b = fetch_team_stats_fd(team_b, LEAGUE_MAP_FD[league], fd_key)

        if stats_a is None or stats_b is None:
            st.stop()

        strength_a, strength_b = 1.0, 1.0
        if api_key:
            fixture_id = get_upcoming_fixture(team_a, team_b, LEAGUE_MAP_API[league], api_key)
            if fixture_id:
                lineup_a = fetch_lineup(fixture_id, team_a, api_key)
                lineup_b = fetch_lineup(fixture_id, team_b, api_key)
                strength_a = calculate_lineup_strength(lineup_a, KEY_PLAYERS.get(team_a, []))
                strength_b = calculate_lineup_strength(lineup_b, KEY_PLAYERS.get(team_b, []))
                if strength_a < 0.9:
                    st.warning(f"⚠️ {team_a} gra osłabionym składem (siła {strength_a*100:.0f}%)")
                if strength_b < 0.9:
                    st.warning(f"⚠️ {team_b} gra osłabionym składem (siła {strength_b*100:.0f}%)")

        # Obliczenia
        lambda_home = (stats_a['avg_gf'] + stats_b['avg_ga']) / 2 * LEAGUE_FACTORS[league] * strength_a
        lambda_away = (stats_b['avg_gf'] + stats_a['avg_ga']) / 2 * LEAGUE_FACTORS[league] * strength_b

        probs_1x2 = calculate_match_probs(lambda_home, lambda_away)
        goal_markets = calculate_goal_markets(lambda_home, lambda_away)
        handicap_prob = calculate_handicap(lambda_home, lambda_away)
        card_probs, corner_probs = simulate_cards_corners(style_h, style_a, league)

        # Raport AI
        if gemini_key:
            with st.spinner("🤖 Gemini pisze raport..."):
                ai_report = generate_llm_report(team_a, team_b, stats_a, stats_b, probs_1x2, goal_markets, league, gemini_key)
                if ai_report:
                    st.markdown("### 🧠 Analiza AI")
                    st.markdown(ai_report)

        st.divider()
        st.header(f"📊 {team_a} vs {team_b}")

        c1, c2, c3 = st.columns(3)
        c1.metric("1", f"{probs_1x2['1']}%")
        c2.metric("X", f"{probs_1x2['X']}%")
        c3.metric("2", f"{probs_1x2['2']}%")

        st.subheader("🎯 Over/Under Gole")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("O 0.5", f"{goal_markets['over_0.5']}%")
        g2.metric("O 1.5", f"{goal_markets['over_1.5']}%")
        g3.metric("O 2.5", f"{goal_markets['over_2.5']}%")
        g4.metric("O 3.5", f"{goal_markets['over_3.5']}%")
        g5.metric("O 4.5", f"{goal_markets['over_4.5']}%")

        st.subheader("⚖️ Handicap")
        st.metric("Gospodarz -1.5", f"{handicap_prob}%")

        st.subheader("🟨 Kartki")
        k1, k2, k3 = st.columns(3)
        k1.metric("O 3.5", f"{card_probs['over_3.5_cards']}%")
        k2.metric("O 4.5", f"{card_probs['over_4.5_cards']}%")
        k3.metric("O 5.5", f"{card_probs['over_5.5_cards']}%")

        st.subheader("🚩 Rożne")
        r1, r2, r3 = st.columns(3)
        r1.metric("O 8.5", f"{corner_probs['over_8.5_corners']}%")
        r2.metric("O 9.5", f"{corner_probs['over_9.5_corners']}%")
        r3.metric("O 10.5", f"{corner_probs['over_10.5_corners']}%")

        # Kupony
        coupons = generate_coupons(probs_1x2, goal_markets, card_probs, corner_probs, handicap_prob, threshold)
        if coupons:
            st.subheader(f"🎫 Sugerowane kupony (próg {threshold}%)")
            c_cols = st.columns(min(len(coupons), 3))
            for i, (label, prob) in enumerate(coupons):
                with c_cols[i % 3]:
                    st.metric(label=label, value=f"{prob:.1f}%", delta="Wysoka szansa")

        # Ostatnie mecze
        st.subheader(f"📋 Ostatnie 5 meczów {team_a}")
        st.dataframe(stats_a['recent_form'], hide_index=True)
        st.subheader(f"📋 Ostatnie 5 meczów {team_b}")
        st.dataframe(stats_b['recent_form'], hide_index=True)
