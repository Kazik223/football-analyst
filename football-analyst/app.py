import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import google.generativeai as genai
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# KONFIGURACJA STRONY
# ------------------------------
st.set_page_config(page_title="AI Football Analyst Pro", layout="wide")
st.title("🧠 Analityk Piłkarski AI + Rynki Bukmacherskie")
st.caption("Dane z Football-Data.org | Raport AI przez Gemini")

# ------------------------------
# 1. MAPY I STAŁE
# ------------------------------
LEAGUE_MAP = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "Ligue 1": "FL1",
    # W razie potrzeby dodaj kolejne ligi obsługiwane przez Football-Data.org
}

LEAGUE_FACTORS = {
    "Premier League": 1.0, "La Liga": 0.95, "Bundesliga": 1.1,
    "Serie A": 0.9, "Ligue 1": 0.98
}

# Średnie ligowe dla kartek i rożnych (wartości przykładowe, można zastąpić danymi z API)
LEAGUE_AVG_CARDS = 3.8   # średnia suma kartek na mecz
LEAGUE_AVG_CORNERS = 9.5 # średnia suma rzutów rożnych na mecz

# ------------------------------
# 2. FUNKCJE POBRANIA DANYCH Z API
# ------------------------------
@st.cache_data(ttl=3600, show_spinner="Pobieram dane z Football-Data.org...")
def fetch_team_stats(team_name, league_code, api_key):
    """Pobiera średnie bramki zdobyte/stracone z ostatnich 5 meczów ligowych."""
    if not api_key:
        return None, "Brak klucza API Football-Data.org"
    
    headers = {'X-Auth-Token': api_key}
    
    # Krok 1: Pobranie listy drużyn w lidze
    teams_url = f"https://api.football-data.org/v4/competitions/{league_code}/teams"
    try:
        resp = requests.get(teams_url, headers=headers)
        resp.raise_for_status()
        teams = resp.json()['teams']
    except Exception as e:
        return None, f"Błąd pobierania drużyn: {e}"
    
    # Znajdź ID drużyny (porównanie z ignorowaniem wielkości liter)
    team_id = None
    for t in teams:
        if team_name.lower() in t['name'].lower():
            team_id = t['id']
            break
    if not team_id:
        return None, f"Nie znaleziono drużyny '{team_name}'"
    
    # Krok 2: Pobranie ostatnich meczów
    matches_url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
    params = {'limit': 20, 'status': 'FINISHED'}
    try:
        resp = requests.get(matches_url, headers=headers, params=params)
        resp.raise_for_status()
        matches = resp.json()['matches']
    except Exception as e:
        return None, f"Błąd pobierania meczów: {e}"
    
    # Filtruj tylko mecze ligowe zakończone
    league_matches = []
    for m in matches:
        if m['competition']['code'] == league_code:
            if m['score']['fullTime']['home'] is not None and m['score']['fullTime']['away'] is not None:
                league_matches.append(m)
    
    if len(league_matches) < 3:
        return None, f"Za mało meczów ligowych ({len(league_matches)}) dla {team_name}"
    
    # Wybierz ostatnie 5 meczów
    recent = league_matches[-5:]
    
    gf_list = []
    ga_list = []
    matches_data = []
    for m in recent:
        is_home = m['homeTeam']['id'] == team_id
        gf = m['score']['fullTime']['home'] if is_home else m['score']['fullTime']['away']
        ga = m['score']['fullTime']['away'] if is_home else m['score']['fullTime']['home']
        opponent = m['awayTeam']['name'] if is_home else m['homeTeam']['name']
        gf_list.append(gf)
        ga_list.append(ga)
        matches_data.append({
            'date': m['utcDate'][:10],
            'opponent': opponent,
            'gf': gf,
            'ga': ga
        })
    
    avg_gf = np.mean(gf_list)
    avg_ga = np.mean(ga_list)
    
    return {
        'avg_xg': round(avg_gf, 2),     # Używamy rzeczywistych bramek jako proxy xG
        'avg_xga': round(avg_ga, 2),
        'possession': 50.0,             # API nie udostępnia
        'ppda': 10.0,
        'recent_form': pd.DataFrame(matches_data),
        'url_found': True
    }, None

# ------------------------------
# 3. MODEL POISSONA DLA GOLI
# ------------------------------
def poisson_goal_matrix(home_lambda, away_lambda, max_goals=6):
    """Tworzy macierz prawdopodobieństw wyników."""
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i, j] = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
    return matrix

def calculate_match_probs(home_xg, away_xg, home_xga, away_xga, league_factor=1.0):
    league_avg = 1.35 * league_factor
    lambda_home = (home_xg / league_avg) * (away_xga / league_avg) * league_avg
    lambda_away = (away_xg / league_avg) * (home_xga / league_avg) * league_avg * 0.95

    matrix = poisson_goal_matrix(lambda_home, lambda_away)
    
    home_win = np.sum(np.tril(matrix, -1))  # suma poniżej przekątnej (home > away)
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    
    likely_score = np.unravel_index(np.argmax(matrix), matrix.shape)
    
    return {
        "home_win": round(home_win * 100, 1),
        "draw": round(draw * 100, 1),
        "away_win": round(away_win * 100, 1),
        "likely_score": f"{likely_score[0]}-{likely_score[1]}",
        "lambda_home": round(lambda_home, 2),
        "lambda_away": round(lambda_away, 2),
        "goal_matrix": matrix
    }

# ------------------------------
# 4. RYNKI BUKMACHERSKIE
# ------------------------------
def calculate_markets(matrix, lambda_home, lambda_away, 
                      lambda_cards=None, lambda_corners=None):
    """Oblicza prawdopodobieństwa dla dodatkowych rynków."""
    markets = {}
    
    # --- Over/Under na gole ---
    total_goals_dist = np.zeros(matrix.shape[0] + matrix.shape[1])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            total_goals_dist[i+j] += matrix[i, j]
    
    markets['over_0.5'] = round(1 - total_goals_dist[0], 3)
    markets['over_1.5'] = round(1 - np.sum(total_goals_dist[:2]), 3)
    markets['over_2.5'] = round(1 - np.sum(total_goals_dist[:3]), 3)
    markets['over_3.5'] = round(1 - np.sum(total_goals_dist[:4]), 3)
    markets['over_4.5'] = round(1 - np.sum(total_goals_dist[:5]), 3)
    
    # --- Handicap Azjatycki -1.5 (Gospodarz wygrywa różnicą ≥2) ---
    ah_minus_1_5 = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i - 1.5 > j:
                ah_minus_1_5 += matrix[i, j]
    markets['ah_home_-1.5'] = round(ah_minus_1_5, 3)
    
    # Handicap +1.5 (Goście nie przegrywają różnicą ≥2)
    markets['ah_away_+1.5'] = round(1 - ah_minus_1_5, 3)
    
    # --- BTTS (Both Teams To Score) ---
    btts = 0
    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            btts += matrix[i, j]
    markets['btts_yes'] = round(btts, 3)
    
    # --- Kartki i rożne (symulacja lub rzeczywiste lambdy) ---
    # Używamy średnich ligowych, jeśli nie podano konkretnych wartości
    if lambda_cards is None:
        lambda_cards = LEAGUE_AVG_CARDS
    if lambda_corners is None:
        lambda_corners = LEAGUE_AVG_CORNERS
    
    # Over/Under na kartki (suma)
    markets['over_2.5_cards'] = round(1 - poisson.cdf(2, lambda_cards), 3)
    markets['over_3.5_cards'] = round(1 - poisson.cdf(3, lambda_cards), 3)
    markets['over_4.5_cards'] = round(1 - poisson.cdf(4, lambda_cards), 3)
    markets['over_5.5_cards'] = round(1 - poisson.cdf(5, lambda_cards), 3)
    
    # Over/Under na rożne (suma)
    markets['over_7.5_corners'] = round(1 - poisson.cdf(7, lambda_corners), 3)
    markets['over_8.5_corners'] = round(1 - poisson.cdf(8, lambda_corners), 3)
    markets['over_9.5_corners'] = round(1 - poisson.cdf(9, lambda_corners), 3)
    markets['over_10.5_corners'] = round(1 - poisson.cdf(10, lambda_corners), 3)
    
    return markets

# ------------------------------
# 5. GENEROWANIE RAPORTU AI (GEMINI)
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

def generate_llm_report(team_a, team_b, stats_a, stats_b, probs, markets, league, api_key):
    if not api_key:
        return None, "Brak klucza API Gemini."
    
    model_name = get_available_model(api_key)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = f"""
Jako ekspert analityki piłkarskiej, przygotuj zwięzły raport w języku polskim dla meczu {team_a} vs {team_b} ({league}).

Dane wejściowe (średnie z ostatnich 5 meczów):
{team_a}: gole zdobyte={stats_a['avg_xg']:.2f}, gole stracone={stats_a['avg_xga']:.2f}
{team_b}: gole zdobyte={stats_b['avg_xg']:.2f}, gole stracone={stats_b['avg_xga']:.2f}

Model Poissona:
- Prawdopodobieństwa: 1={probs['home_win']:.1f}%, X={probs['draw']:.1f}%, 2={probs['away_win']:.1f}%
- Najbardziej prawdopodobny wynik: {probs['likely_score']}
- Oczekiwane gole: {team_a} {probs['lambda_home']:.2f} - {probs['lambda_away']:.2f} {team_b}

Wybrane rynki bukmacherskie:
- BTTS: {markets['btts_yes']*100:.1f}%
- Over 2.5 gola: {markets['over_2.5']*100:.1f}%
- Handicap -1.5 dla {team_a}: {markets['ah_home_-1.5']*100:.1f}%

Struktura raportu:
1. Profil Taktyczny (2-3 zdania o stylu gry i kluczowych pojedynkach)
2. Dominujący Scenariusz (najbardziej prawdopodobny przebieg)
3. Margines Błędu (jeden czynnik ryzyka)
"""
    try:
        response = model.generate_content(prompt)
        return response.text, None
    except Exception as e:
        return None, f"Błąd API Gemini: {e}"

# ------------------------------
# 6. INTERFEJS UŻYTKOWNIKA
# ------------------------------
with st.sidebar:
    st.header("🔑 Klucze API")
    gemini_key = st.text_input("Klucz Gemini (opcjonalny)", type="password")
    st.caption("[Zdobądź klucz Gemini](https://aistudio.google.com/app/apikey)")
    
    football_api_key = st.text_input("Klucz Football-Data.org", type="password")
    st.caption("[Zdobądź darmowy klucz](https://www.football-data.org/client/register)")
    
    st.divider()
    st.header("⚙️ Ustawienia")
    use_ai_report = st.checkbox("Generuj raport AI", value=True)
    
    st.divider()
    st.markdown("**Uwaga**: Kartki i rożne są symulowane na podstawie średnich ligowych. Aby użyć rzeczywistych danych, wymagane jest płatne API.")

# Główny panel
col1, col2 = st.columns(2)
with col1:
    team_a = st.text_input("🏠 DRUŻYNA A (Gospodarz)", "Manchester City")
with col2:
    team_b = st.text_input("✈️ DRUŻYNA B (Goście)", "Arsenal")

league = st.selectbox("🏆 Liga", list(LEAGUE_MAP.keys()))

if st.button("🚀 Wykonaj Pełną Analizę", type="primary"):
    if not football_api_key:
        st.error("❌ Klucz API Football-Data.org jest wymagany!")
    else:
        league_code = LEAGUE_MAP[league]
        
        with st.spinner(f"Pobieram dane dla {team_a}..."):
            data_a, err_a = fetch_team_stats(team_a, league_code, football_api_key)
        with st.spinner(f"Pobieram dane dla {team_b}..."):
            data_b, err_b = fetch_team_stats(team_b, league_code, football_api_key)
        
        if err_a or err_b:
            if err_a: st.error(f"{team_a}: {err_a}")
            if err_b: st.error(f"{team_b}: {err_b}")
        elif data_a and data_b:
            factor = LEAGUE_FACTORS.get(league, 1.0)
            probs = calculate_match_probs(
                data_a['avg_xg'], data_b['avg_xg'],
                data_a['avg_xga'], data_b['avg_xga'], factor
            )
            
            markets = calculate_markets(
                probs['goal_matrix'],
                probs['lambda_home'],
                probs['lambda_away']
            )
            
            # Raport AI
            ai_report = None
            if gemini_key and use_ai_report:
                with st.spinner("🤖 Gemini pisze raport..."):
                    ai_report, ai_err = generate_llm_report(
                        team_a, team_b, data_a, data_b, probs, markets, league, gemini_key
                    )
                    if ai_err:
                        st.error(ai_err)
            
            # Wyświetlanie wyników
            st.divider()
            st.header(f"📊 Analiza: {team_a} vs {team_b}")
            
            if ai_report:
                st.markdown("### 🧠 Raport AI")
                st.markdown(ai_report)
            
            # Podstawowe statystyki
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"{team_a} śr. gole zdobyte", data_a['avg_xg'])
            col2.metric(f"{team_b} śr. gole zdobyte", data_b['avg_xg'])
            col3.metric(f"{team_a} śr. gole stracone", data_a['avg_xga'])
            col4.metric(f"{team_b} śr. gole stracone", data_b['avg_xga'])
            
            # Model Poissona (gole)
            st.subheader("🎲 Prawdopodobieństwa wyniku (1X2)")
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("🏠 Wygrana Gospodarzy", f"{probs['home_win']}%")
            pcol2.metric("🤝 Remis", f"{probs['draw']}%")
            pcol3.metric("✈️ Wygrana Gości", f"{probs['away_win']}%")
            st.info(f"**Najbardziej prawdopodobny wynik:** {probs['likely_score']} (λ: {probs['lambda_home']} - {probs['lambda_away']})")
            
            # Rynki bukmacherskie
            st.subheader("🎯 Dodatkowe rynki (prawdopodobieństwa)")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Gole Over/Under", "Handicap & BTTS", "Kartki (sym.)", "Rożne (sym.)"])
            
            with tab1:
                cols = st.columns(4)
                cols[0].metric("Over 0.5", f"{markets['over_0.5']*100:.1f}%")
                cols[1].metric("Over 1.5", f"{markets['over_1.5']*100:.1f}%")
                cols[2].metric("Over 2.5", f"{markets['over_2.5']*100:.1f}%")
                cols[3].metric("Over 3.5", f"{markets['over_3.5']*100:.1f}%")
                st.caption("Over 4.5: {:.1f}%".format(markets['over_4.5']*100))
            
            with tab2:
                cols = st.columns(3)
                cols[0].metric(f"Handicap -1.5 ({team_a})", f"{markets['ah_home_-1.5']*100:.1f}%")
                cols[1].metric(f"Handicap +1.5 ({team_b})", f"{markets['ah_away_+1.5']*100:.1f}%")
                cols[2].metric("BTTS (obie strzelą)", f"{markets['btts_yes']*100:.1f}%")
            
            with tab3:
                st.caption("Symulacja na podstawie średniej ligowej (ok. 3.8 kartki/mecz)")
                cols = st.columns(4)
                cols[0].metric("Over 2.5", f"{markets['over_2.5_cards']*100:.1f}%")
                cols[1].metric("Over 3.5", f"{markets['over_3.5_cards']*100:.1f}%")
                cols[2].metric("Over 4.5", f"{markets['over_4.5_cards']*100:.1f}%")
                cols[3].metric("Over 5.5", f"{markets['over_5.5_cards']*100:.1f}%")
            
            with tab4:
                st.caption("Symulacja na podstawie średniej ligowej (ok. 9.5 rożnego/mecz)")
                cols = st.columns(4)
                cols[0].metric("Over 7.5", f"{markets['over_7.5_corners']*100:.1f}%")
                cols[1].metric("Over 8.5", f"{markets['over_8.5_corners']*100:.1f}%")
                cols[2].metric("Over 9.5", f"{markets['over_9.5_corners']*100:.1f}%")
                cols[3].metric("Over 10.5", f"{markets['over_10.5_corners']*100:.1f}%")
            
            # Ostatnie mecze
            if not data_a['recent_form'].empty:
                st.subheader(f"📋 Ostatnie mecze {team_a}")
                st.dataframe(data_a['recent_form'], hide_index=True)
            if not data_b['recent_form'].empty:
                st.subheader(f"📋 Ostatnie mecze {team_b}")
                st.dataframe(data_b['recent_form'], hide_index=True)