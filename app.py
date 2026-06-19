import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import hashlib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Food Recommendation System",
    page_icon="🍛",
    layout="wide"
)

# =========================
# USER AUTH SYSTEM
# Uses a local JSON file to store users
# =========================
USERS_FILE = 'users.json'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(username, password, name):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        'password':      hash_password(password),
        'name':          name,
        'profile':       {},
        'liked_foods':   [],
        'disliked_foods':[],
        'food_log':      []
    }
    save_users(users)
    return True, "Account created successfully"

def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "Username not found"
    if users[username]['password'] != hash_password(password):
        return False, "Incorrect password"
    return True, users[username]

def save_user_profile(username, profile):
    users = load_users()
    if username in users:
        users[username]['profile'] = profile
        save_users(users)

def save_user_liked(username, liked, disliked):
    users = load_users()
    if username in users:
        users[username]['liked_foods']    = liked
        users[username]['disliked_foods'] = disliked
        save_users(users)

def save_food_log(username, food_log):
    users = load_users()
    if username in users:
        users[username]['food_log'] = food_log
        save_users(users)

def get_user_data(username):
    users = load_users()
    return users.get(username, {})

# =========================
# LOGIN / REGISTER PAGE
# =========================
def show_login_page():
    st.title("🍛 AI-Based Food Recommendation System")
    st.markdown("*Personalized Indian food recommendations based on your health profile*")
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        with tab_login:
            st.subheader("Welcome Back!")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", use_container_width=True, type="primary"):
                if username and password:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state['logged_in']  = True
                        st.session_state['username']   = username
                        st.session_state['user_name']  = result['name']
                        st.session_state['liked_foods']    = result.get('liked_foods', [])
                        st.session_state['disliked_foods'] = result.get('disliked_foods', [])
                        st.session_state['food_log']       = result.get('food_log', [])
                        st.session_state['saved_profile']  = result.get('profile', {})
                        st.success(f"Welcome back, {result['name']}!")
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.warning("Please enter username and password")

        with tab_register:
            st.subheader("Create Account")
            new_name     = st.text_input("Full Name",        key="reg_name")
            new_username = st.text_input("Username",         key="reg_user")
            new_password = st.text_input("Password",         type="password", key="reg_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="reg_confirm")

            if st.button("Create Account", use_container_width=True, type="primary"):
                if new_name and new_username and new_password:
                    if new_password != confirm_pass:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, msg = register_user(new_username, new_password, new_name)
                        if success:
                            st.success(f"{msg} — Please login now")
                        else:
                            st.error(msg)
                else:
                    st.warning("Please fill all fields")

# =========================
# LOAD & PREPARE DATA
# =========================
@st.cache_data(show_spinner=False)
def load_and_prepare():
    df_nutrition    = pd.read_csv('Indian_Food_Nutrition_Processed.csv')
    df_meta         = pd.read_csv('indian_food.csv')
    df_disease_food = pd.read_csv('pred_food.csv')
    df_diet_recs    = pd.read_csv('diet_recommendations_dataset.csv')
    df_meal_plan    = pd.read_csv('Personalized_Diet_Recommendations.csv')

    df_nutrition = df_nutrition.dropna().drop_duplicates()
    if 'Dish Name' not in df_nutrition.columns:
        df_nutrition.rename(columns={df_nutrition.columns[0]: 'Dish Name'}, inplace=True)
    df_nutrition['Dish Name'] = df_nutrition['Dish Name'].str.strip().str.title()

    df_meta = df_meta.dropna().drop_duplicates()
    df_meta.rename(columns={'name': 'Dish Name'}, inplace=True)
    df_meta['Dish Name'] = df_meta['Dish Name'].str.strip().str.title()
    for col in ['prep_time', 'cook_time']:
        df_meta[col] = pd.to_numeric(df_meta[col], errors='coerce')
    df_meta = df_meta.dropna(subset=['prep_time', 'cook_time'])

    df = pd.merge(df_nutrition, df_meta, on='Dish Name', how='left')
    df['diet']           = df['diet'].fillna('unknown')
    df['region']         = df['region'].fillna('unknown')
    df['course']         = df['course'].fillna('unknown')
    df['flavor_profile'] = df['flavor_profile'].fillna('unknown')
    df['prep_time']      = df['prep_time'].fillna(0)
    df['cook_time']      = df['cook_time'].fillna(0)
    df = df.fillna(0)

    def minmax(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r != 0 else s * 0

    df['Protein Density'] = df['Protein (g)'] / df['Calories (kcal)'].replace(0, np.nan)
    df['Fiber Density']   = df['Fibre (g)']   / df['Calories (kcal)'].replace(0, np.nan)
    df = df.fillna(0)

    df['Health Score'] = (
        0.30 * minmax(df['Protein (g)']) +
        0.25 * minmax(df['Fibre (g)']) +
        0.20 * minmax(df['Iron (mg)']) +
        0.15 * minmax(df['Calcium (mg)']) +
        0.10 * minmax(df['Vitamin C (mg)'])
    )
    df['Fitness Score'] = (
        0.30 * minmax(df['Protein (g)']) +
        0.20 * minmax(df['Fibre (g)']) +
        0.15 * minmax(df['Vitamin C (mg)']) +
        0.10 * minmax(df['Iron (mg)']) -
        0.15 * minmax(df['Free Sugar (g)']) -
        0.10 * minmax(df['Sodium (mg)'])
    )
    df['IsVeg'] = df['diet'].apply(lambda x: 1 if str(x).lower() == 'vegetarian' else 0)

    def map_meal_type(course):
        course = str(course).lower()
        if any(k in course for k in ['breakfast', 'morning', 'snack']):   return 'Breakfast'
        elif any(k in course for k in ['main', 'lunch', 'rice', 'bread']): return 'Lunch'
        elif any(k in course for k in ['dinner', 'dessert', 'sweet']):     return 'Dinner'
        else: return 'Lunch'
    df['Meal Type'] = df['course'].apply(map_meal_type)

    NUTRITION_FEATURES = [c for c in [
        'Calories (kcal)', 'Carbohydrates (g)', 'Protein (g)',
        'Fats (g)', 'Free Sugar (g)', 'Fibre (g)',
        'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)',
        'Vitamin C (mg)', 'Folate (µg)',
        'Protein Density', 'Fiber Density'
    ] if c in df.columns]

    scaler    = StandardScaler()
    df_scaled = scaler.fit_transform(df[NUTRITION_FEATURES])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    cluster_labels = {0:'High-Calorie', 1:'High-Protein', 2:'High-Fibre', 3:'Balanced', 4:'Low-Calorie'}
    df['Cluster Label'] = df['Cluster'].map(cluster_labels)

    df_disease_food['Food Name'] = df_disease_food['Food Name'].str.strip().str.title()

    disease_diet_map = (
        df_diet_recs.groupby('Disease_Type')['Diet_Recommendation']
        .agg(lambda x: x.value_counts().index[0]).to_dict()
    )

    # --- BASIC MODEL 1: Linear Regression ---
    lr_features = ['Age', 'Weight_kg', 'Height_cm', 'BMI']
    lr_data     = df_diet_recs[lr_features + ['Daily_Caloric_Intake']].dropna()
    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(
        lr_data[lr_features], lr_data['Daily_Caloric_Intake'], test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_lr_train, y_lr_train)
    lr_score = round(lr_model.score(X_lr_test, y_lr_test) * 100, 2)

    # --- CUSTOM MODEL 1: Gradient Boosting Regressor ---
    le_activity = LabelEncoder()
    df_diet_recs_clean = df_diet_recs.copy()
    df_diet_recs_clean['Activity_Encoded'] = le_activity.fit_transform(
        df_diet_recs_clean['Physical_Activity_Level'].fillna('Moderate'))
    custom_lr_features = [c for c in ['Age', 'Weight_kg', 'Height_cm', 'BMI',
                                       'Activity_Encoded', 'Weekly_Exercise_Hours']
                          if c in df_diet_recs_clean.columns]
    custom_lr_data = df_diet_recs_clean[custom_lr_features + ['Daily_Caloric_Intake']].dropna()
    X_gb_train, X_gb_test, y_gb_train, y_gb_test = train_test_split(
        custom_lr_data[custom_lr_features], custom_lr_data['Daily_Caloric_Intake'],
        test_size=0.2, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    gb_model.fit(X_gb_train, y_gb_train)
    gb_preds = gb_model.predict(X_gb_test)
    gb_score = round(r2_score(y_gb_test, gb_preds) * 100, 2)
    gb_mae   = round(mean_absolute_error(y_gb_test, gb_preds), 2)

    # --- BASIC MODEL 2: Random Forest ---
    rf_features = [c for c in ['Glycemic Index', 'Calories', 'Carbohydrates', 'Protein',
                                'Fat', 'Sodium Content', 'Potassium Content', 'Fiber Content']
                   if c in df_disease_food.columns]
    rf_data     = df_disease_food[rf_features + ['Suitable for Diabetes']].dropna()
    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
        rf_data[rf_features], rf_data['Suitable for Diabetes'], test_size=0.2, random_state=42)
    rf_model    = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_rf_train, y_rf_train)
    rf_accuracy = round(rf_model.score(X_rf_test, y_rf_test) * 100, 2)

    # --- CUSTOM MODEL 2: GBC Classifier with engineered features ---
    df_food_eng = df_disease_food.copy()
    df_food_eng['Fiber_Protein_Ratio']    = df_food_eng['Fiber Content']  / (df_food_eng['Protein'] + 1)
    df_food_eng['Sodium_Potassium_Ratio'] = df_food_eng['Sodium Content'] / (df_food_eng['Potassium Content'] + 1)
    df_food_eng['Carb_Fiber_Ratio']       = df_food_eng['Carbohydrates']  / (df_food_eng['Fiber Content'] + 1)
    df_food_eng['Calorie_Density']        = df_food_eng['Calories']       / (df_food_eng['Fiber Content'] + 1)
    custom_rf_features = [c for c in rf_features + ['Fiber_Protein_Ratio', 'Sodium_Potassium_Ratio',
                                                      'Carb_Fiber_Ratio', 'Calorie_Density']
                          if c in df_food_eng.columns]
    custom_rf_data = df_food_eng[custom_rf_features + ['Suitable for Diabetes']].dropna()
    X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(
        custom_rf_data[custom_rf_features], custom_rf_data['Suitable for Diabetes'],
        test_size=0.2, random_state=42)
    xgb_model    = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                               max_depth=5, subsample=0.8, random_state=42)
    xgb_model.fit(X_xgb_train, y_xgb_train)
    xgb_accuracy = round(accuracy_score(y_xgb_test, xgb_model.predict(X_xgb_test)) * 100, 2)

    # --- BASIC MODEL 3: KNN ---
    cf_features = [c for c in ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Daily_Caloric_Intake']
                   if c in df_diet_recs.columns]
    cf_data     = df_diet_recs[cf_features].dropna()
    cf_scaler   = StandardScaler()
    cf_scaled   = cf_scaler.fit_transform(cf_data)
    cf_model    = NearestNeighbors(n_neighbors=5, metric='cosine')
    cf_model.fit(cf_scaled)

    le_diet     = LabelEncoder()
    df_diet_cf  = df_diet_recs[cf_features + ['Diet_Recommendation']].dropna()
    y_cf_all    = le_diet.fit_transform(df_diet_cf['Diet_Recommendation'])
    X_cf_all    = df_diet_cf[cf_features].values
    split       = int(len(X_cf_all) * 0.8)
    knn_reg     = KNeighborsRegressor(n_neighbors=5)
    knn_reg.fit(X_cf_all[:split], y_cf_all[:split])
    knn_cf_score = round(knn_reg.score(X_cf_all[split:], y_cf_all[split:]) * 100, 2)

    # --- CUSTOM MODEL 3: SVD + GBC ---
    svd          = TruncatedSVD(n_components=4, random_state=42)
    X_cf_reduced = svd.fit_transform(X_cf_all)
    gbc_cf       = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbc_cf.fit(X_cf_reduced[:split], y_cf_all[:split])
    custom_cf_score = round(gbc_cf.score(X_cf_reduced[split:], y_cf_all[split:]) * 100, 2)

    return (df, scaler, NUTRITION_FEATURES,
            df_disease_food, df_food_eng, disease_diet_map, df_meal_plan,
            lr_model, lr_score,
            rf_model, rf_accuracy, rf_features,
            cf_model, cf_scaler, cf_data,
            gb_model, gb_score, gb_mae, custom_lr_features, le_activity,
            xgb_model, xgb_accuracy, custom_rf_features,
            svd, gbc_cf, custom_cf_score, knn_cf_score, le_diet,
            df_diet_recs)

# =========================
# HEALTH CALC FUNCTIONS
# =========================
def calculate_bmi(weight, height_cm):
    h   = height_cm / 100
    bmi = round(weight / (h ** 2), 2)
    if bmi < 18.5:   cat = 'Underweight'
    elif bmi < 25.0: cat = 'Normal weight'
    elif bmi < 30.0: cat = 'Overweight'
    else:            cat = 'Obese'
    return bmi, cat

def calculate_bmr(weight, height_cm, age, gender):
    if gender == 'Male':
        return round(10 * weight + 6.25 * height_cm - 5 * age + 5, 2)
    return round(10 * weight + 6.25 * height_cm - 5 * age - 161, 2)

def calculate_daily_calories(bmr, activity, goal):
    multipliers = {'Sedentary': 1.2, 'Moderate': 1.55, 'Active': 1.725}
    tdee = bmr * multipliers.get(activity, 1.2)
    if goal == 'Weight Loss':   tdee -= 500
    elif goal == 'Muscle Gain': tdee += 300
    return round(tdee, 2)

# =========================
# RECOMMENDATION FUNCTIONS
# =========================
def recommend(df, scaler, features, goal, daily_calories, is_veg, top_n,
              liked_foods=None, disliked_foods=None):
    filtered = df.copy()
    if is_veg == 'Vegetarian':
        filtered = filtered[filtered['IsVeg'] == 1]
    elif is_veg == 'Non-Vegetarian':
        filtered = filtered[filtered['IsVeg'] == 0]

    # Remove disliked foods
    if disliked_foods:
        filtered = filtered[~filtered['Dish Name'].isin(disliked_foods)]

    if goal == 'Weight Loss':
        filtered = filtered[
            (filtered['Calories (kcal)'] <= filtered['Calories (kcal)'].quantile(0.40)) &
            (filtered['Fibre (g)'] >= filtered['Fibre (g)'].quantile(0.50))]
    elif goal == 'Muscle Gain':
        filtered = filtered[filtered['Protein (g)'] >= filtered['Protein (g)'].quantile(0.60)]
    elif goal == 'Maintenance':
        low  = filtered['Calories (kcal)'].quantile(0.20)
        high = filtered['Calories (kcal)'].quantile(0.80)
        filtered = filtered[
            (filtered['Calories (kcal)'].between(low, high)) &
            (filtered['Health Score'] >= filtered['Health Score'].quantile(0.50))]

    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    filtered_scaled = scaler.transform(filtered[features])

    # If user has liked foods build preference vector from them
    if liked_foods:
        liked_df = df[df['Dish Name'].isin(liked_foods)]
        if not liked_df.empty:
            pref_vector = liked_df[features].mean().values
            target_vector = scaler.transform([pref_vector])
        else:
            target_vector = _build_target_vector(df, features, scaler, goal, daily_calories)
    else:
        target_vector = _build_target_vector(df, features, scaler, goal, daily_calories)

    similarity = cosine_similarity(target_vector, filtered_scaled).flatten()
    filtered['Match Score'] = (similarity * 100).round(1)

    result = filtered.sort_values('Match Score', ascending=False).head(top_n)
    display_cols = ['Dish Name', 'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)',
                    'Fats (g)', 'Fibre (g)', 'Health Score', 'Fitness Score', 'Match Score']
    for c in ['diet', 'region', 'course', 'flavor_profile']:
        if c in result.columns:
            display_cols.append(c)
    return result[[c for c in display_cols if c in result.columns]].reset_index(drop=True)

def _build_target_vector(df, features, scaler, goal, daily_calories):
    if goal == 'Weight Loss':
        target_cal = daily_calories * 0.30; target_protein = 20; target_fibre = 6; target_fats = 10
    elif goal == 'Muscle Gain':
        target_cal = daily_calories * 0.35; target_protein = 35; target_fibre = 3; target_fats = 15
    else:
        target_cal = daily_calories * 0.33; target_protein = 25; target_fibre = 4; target_fats = 12
    target_raw = df[features].mean().copy()
    target_raw['Calories (kcal)'] = target_cal
    if 'Protein (g)' in target_raw: target_raw['Protein (g)'] = target_protein
    if 'Fibre (g)'   in target_raw: target_raw['Fibre (g)']   = target_fibre
    if 'Fats (g)'    in target_raw: target_raw['Fats (g)']    = target_fats
    return scaler.transform([target_raw.values])

def generate_reason(row, df):
    reasons = []
    if row.get('Protein (g)', 0)       >= df['Protein (g)'].quantile(0.75):      reasons.append('💪 High Protein')
    if row.get('Fibre (g)', 0)          >= df['Fibre (g)'].quantile(0.75):        reasons.append('🌾 High Fibre')
    if row.get('Calories (kcal)', 9999) <= df['Calories (kcal)'].quantile(0.35):  reasons.append('🔥 Low Calorie')
    if row.get('Health Score', 0)       >= df['Health Score'].quantile(0.75):     reasons.append('⭐ Nutrient Dense')
    return ' | '.join(reasons) if reasons else '✅ Balanced Nutrition'

# =========================
# DISEASE FILTERING
# =========================
DISEASE_RULES = {
    'Diabetes':        {'description': 'Focus on low glycemic index, high fibre, low sugar',
                        'restrict': ['Free Sugar (g)', 'Carbohydrates (g)'],
                        'promote':  ['Fibre (g)', 'Protein (g)'],
                        'pred_filter': 'Suitable for Diabetes', 'diet_type': 'Low_Carb'},
    'Hypertension':    {'description': 'Focus on low sodium, high potassium',
                        'restrict': ['Sodium (mg)'],
                        'promote':  ['Fibre (g)', 'Calcium (mg)'],
                        'pred_filter': 'Suitable for Blood Pressure', 'diet_type': 'Low_Sodium'},
    'Obesity':         {'description': 'Focus on low calorie, high fibre',
                        'restrict': ['Calories (kcal)', 'Fats (g)', 'Free Sugar (g)'],
                        'promote':  ['Fibre (g)', 'Protein (g)'],
                        'pred_filter': None, 'diet_type': 'Balanced'},
    'High Cholesterol':{'description': 'Focus on low fat, high fibre',
                        'restrict': ['Fats (g)', 'Free Sugar (g)'],
                        'promote':  ['Fibre (g)', 'Protein (g)'],
                        'pred_filter': None, 'diet_type': 'Low_Carb'},
    'Anaemia':         {'description': 'Focus on high iron and Vitamin C',
                        'restrict': [],
                        'promote':  ['Iron (mg)', 'Vitamin C (mg)', 'Folate (µg)'],
                        'pred_filter': None, 'diet_type': 'Balanced'}
}

def filter_by_disease(df, df_disease_food, disease, is_veg, top_n=10):
    if disease not in DISEASE_RULES: return pd.DataFrame()
    rules    = DISEASE_RULES[disease]
    filtered = df.copy()
    if is_veg == 'Vegetarian':     filtered = filtered[filtered['IsVeg'] == 1]
    elif is_veg == 'Non-Vegetarian': filtered = filtered[filtered['IsVeg'] == 0]
    for col in rules['restrict']:
        if col in filtered.columns:
            filtered = filtered[filtered[col] <= filtered[col].quantile(0.35)]
    for col in rules['promote']:
        if col in filtered.columns:
            filtered = filtered[filtered[col] >= filtered[col].quantile(0.50)]
    pred_col = rules.get('pred_filter')
    if pred_col and pred_col in df_disease_food.columns:
        suitable = df_disease_food[df_disease_food[pred_col] == 1]['Food Name'].str.strip().str.title().tolist()
        mask = filtered['Dish Name'].apply(
            lambda n: any(s.lower() in n.lower() or n.lower() in s.lower() for s in suitable))
        if mask.sum() > 0: filtered = filtered[mask]
    if filtered.empty: return pd.DataFrame()
    filtered = filtered.copy()
    score = filtered['Health Score'].copy()
    for col in rules['promote']:
        if col in filtered.columns:
            r = filtered[col].max() - filtered[col].min()
            if r > 0: score += (filtered[col] - filtered[col].min()) / r
    filtered['Disease Score'] = score.round(3)
    filtered = filtered.sort_values('Disease Score', ascending=False)
    display_cols = ['Dish Name', 'Calories (kcal)', 'Protein (g)', 'Fibre (g)',
                    'Fats (g)', 'Sodium (mg)', 'Health Score', 'Disease Score']
    for c in ['diet', 'region', 'course', 'Free Sugar (g)', 'Iron (mg)', 'Vitamin C (mg)']:
        if c in filtered.columns: display_cols.append(c)
    return filtered[[c for c in display_cols if c in filtered.columns]].head(top_n).reset_index(drop=True)

def foods_to_avoid(df, disease, top_n=5):
    rules = DISEASE_RULES.get(disease, {})
    result = []
    for col in rules.get('restrict', []):
        if col in df.columns:
            result.append(df.nlargest(top_n, col)[['Dish Name', col]])
    if result:
        return pd.concat(result).drop_duplicates('Dish Name').head(top_n).reset_index(drop=True)
    return pd.DataFrame()

# =========================
# MEAL PLANNING
# =========================
def get_macro_targets(df_meal_plan, age, gender, bmi, disease=None):
    filtered = df_meal_plan.copy()
    filtered = filtered[filtered['Gender'].str.lower() == gender.lower()]
    if filtered.empty: filtered = df_meal_plan.copy()
    if disease and disease != 'None' and 'Chronic_Disease' in filtered.columns:
        d = filtered[filtered['Chronic_Disease'].str.contains(disease, case=False, na=False)]
        if not d.empty: filtered = d
    filtered = filtered.copy()
    filtered['BMI_diff'] = (filtered['BMI'] - bmi).abs()
    closest = filtered.nsmallest(10, 'BMI_diff')
    return {
        'calories':       round(closest['Recommended_Calories'].mean()),
        'protein':        round(closest['Recommended_Protein'].mean()),
        'carbs':          round(closest['Recommended_Carbs'].mean()),
        'fats':           round(closest['Recommended_Fats'].mean()),
        'meal_plan_type': closest['Recommended_Meal_Plan'].mode()[0] if not closest.empty else 'Balanced Diet'
    }

def generate_meal_plan(df, macro_targets, is_veg, disease=None):
    meal_calories = {'Breakfast': macro_targets['calories'] * 0.25,
                     'Lunch':     macro_targets['calories'] * 0.40,
                     'Dinner':    macro_targets['calories'] * 0.35}
    filtered = df.copy()
    if is_veg == 'Vegetarian':     filtered = filtered[filtered['IsVeg'] == 1]
    elif is_veg == 'Non-Vegetarian': filtered = filtered[filtered['IsVeg'] == 0]
    if disease and disease != 'None' and disease in DISEASE_RULES:
        for col in DISEASE_RULES[disease]['restrict']:
            if col in filtered.columns:
                filtered = filtered[filtered[col] <= filtered[col].quantile(0.40)]
    meal_plan  = {}
    used_foods = set()
    for meal, cal_target in meal_calories.items():
        meal_foods = filtered[filtered['Meal Type'] == meal].copy()
        if len(meal_foods) < 3: meal_foods = filtered.copy()
        meal_foods = meal_foods[~meal_foods['Dish Name'].isin(used_foods)].copy()
        if meal_foods.empty:
            meal_plan[meal] = pd.DataFrame()
            continue
        meal_foods['Cal_diff'] = (meal_foods['Calories (kcal)'] - cal_target).abs()
        meal_foods = meal_foods.sort_values(['Cal_diff', 'Health Score'], ascending=[True, False])
        top_foods  = meal_foods.head(3)
        used_foods.update(top_foods['Dish Name'].tolist())
        cols = [c for c in ['Dish Name', 'Calories (kcal)', 'Protein (g)',
                             'Carbohydrates (g)', 'Fats (g)', 'Fibre (g)', 'Health Score']
                if c in top_foods.columns]
        meal_plan[meal] = top_foods[cols].reset_index(drop=True)
    return meal_plan, macro_targets

# =========================
# MAIN APP STARTS HERE
# =========================

# Check login state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in']      = False
if 'liked_foods'    not in st.session_state: st.session_state['liked_foods']    = []
if 'disliked_foods' not in st.session_state: st.session_state['disliked_foods'] = []
if 'food_log'       not in st.session_state: st.session_state['food_log']       = []
if 'saved_profile'  not in st.session_state: st.session_state['saved_profile']  = {}

# Show login if not logged in
if not st.session_state['logged_in']:
    show_login_page()
    st.stop()

# =========================
# LOGGED IN — SHOW MAIN APP
# =========================
st.title("🍛 AI-Based Food Recommendation System")
st.markdown(f"*Welcome, **{st.session_state['user_name']}** — Personalized Indian food recommendations*")

# Logout button
if st.sidebar.button("🚪 Logout"):
    for key in ['logged_in', 'username', 'user_name', 'liked_foods',
                'disliked_foods', 'food_log', 'saved_profile']:
        st.session_state.pop(key, None)
    st.rerun()

# Load data
with st.spinner("Loading datasets and training ML models..."):
    try:
        (df, scaler, NUTRITION_FEATURES,
         df_disease_food, df_food_eng, disease_diet_map, df_meal_plan,
         lr_model, lr_score,
         rf_model, rf_accuracy, rf_features,
         cf_model, cf_scaler, cf_data,
         gb_model, gb_score, gb_mae, custom_lr_features, le_activity,
         xgb_model, xgb_accuracy, custom_rf_features,
         svd, gbc_cf, custom_cf_score, knn_cf_score, le_diet,
         df_diet_recs) = load_and_prepare()
        st.success(f"✅ {len(df)} food items loaded | 6 ML models trained")
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.stop()

st.divider()

# =========================
# SIDEBAR — USER PROFILE
# =========================
st.sidebar.header("👤 Your Health Profile")

# Load saved profile if exists
saved = st.session_state.get('saved_profile', {})

age       = st.sidebar.number_input("Age (years)",   min_value=10, max_value=100, value=saved.get('age', 21))
gender    = st.sidebar.selectbox("Gender",            ["Male", "Female"],
                                  index=["Male","Female"].index(saved.get('gender','Male')))
weight    = st.sidebar.number_input("Weight (kg)",   min_value=20.0, max_value=200.0,
                                     value=float(saved.get('weight', 72.0)), step=0.5)
height    = st.sidebar.number_input("Height (cm)",   min_value=100.0, max_value=250.0,
                                     value=float(saved.get('height', 175.0)), step=0.5)
activity  = st.sidebar.selectbox("Activity Level",   ["Sedentary", "Moderate", "Active"],
                                  index=["Sedentary","Moderate","Active"].index(saved.get('activity','Moderate')))
goal      = st.sidebar.selectbox("Your Goal",        ["Weight Loss", "Muscle Gain", "Maintenance"],
                                  index=["Weight Loss","Muscle Gain","Maintenance"].index(saved.get('goal','Weight Loss')))
diet_pref = st.sidebar.selectbox("Diet Preference",  ["Both", "Vegetarian", "Non-Vegetarian"],
                                  index=["Both","Vegetarian","Non-Vegetarian"].index(saved.get('diet_pref','Both')))
disease   = st.sidebar.selectbox("Health Condition", ["None","Diabetes","Hypertension",
                                                       "Obesity","High Cholesterol","Anaemia"],
                                  index=["None","Diabetes","Hypertension","Obesity",
                                         "High Cholesterol","Anaemia"].index(saved.get('disease','None')))
top_n     = st.sidebar.slider("Recommendations", min_value=5, max_value=20, value=10)

# Save profile button
if st.sidebar.button("💾 Save My Profile", use_container_width=True):
    profile = {'age': age, 'gender': gender, 'weight': weight, 'height': height,
               'activity': activity, 'goal': goal, 'diet_pref': diet_pref, 'disease': disease}
    save_user_profile(st.session_state['username'], profile)
    st.session_state['saved_profile'] = profile
    st.sidebar.success("Profile saved!")

st.sidebar.divider()
get_recs      = st.sidebar.button("🔍 Get Recommendations",  use_container_width=True)
get_disease   = st.sidebar.button("🏥 Disease Food Filter",  use_container_width=True)
get_meal_plan = st.sidebar.button("📅 Generate Meal Plan",   use_container_width=True)

# =========================
# HEALTH METRICS
# =========================
bmi, bmi_cat   = calculate_bmi(weight, height)
bmr            = calculate_bmr(weight, height, age, gender)
daily_calories = calculate_daily_calories(bmr, activity, goal)

lr_predicted   = round(lr_model.predict([[age, weight, height, bmi]])[0])
activity_map   = {'Sedentary': 0, 'Moderate': 1, 'Active': 2}
gb_input       = [age, weight, height, bmi, activity_map.get(activity, 1), 3.0]
gb_predicted   = round(gb_model.predict([gb_input[:len(custom_lr_features)]])[0])

col1, col2, col3, col4 = st.columns(4)
col1.metric("BMI",           f"{bmi}",              bmi_cat)
col2.metric("BMR",           f"{bmr} kcal",         "Base metabolic rate")
col3.metric("Basic LR",      f"{lr_predicted} kcal", f"R²: {lr_score}%")
col4.metric("Custom GB",     f"{gb_predicted} kcal", f"R²: {gb_score}% ↑")

bmi_color = {"Underweight":"🔵","Normal weight":"🟢","Overweight":"🟡","Obese":"🔴"}
st.markdown(f"**BMI Status:** {bmi_color.get(bmi_cat,'⚪')} {bmi_cat}")
if disease != 'None':
    st.info(f"🏥 **{disease}** — {DISEASE_RULES[disease]['description']}")

# Liked/Disliked summary
if st.session_state['liked_foods']:
    st.success(f"❤️ You have liked {len(st.session_state['liked_foods'])} foods — recommendations are personalised to your taste")

st.divider()

# ML Model comparison
with st.expander("🤖 ML Models — Basic vs Custom"):
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("**Calorie Prediction**")
        st.error(f"Basic LR: {lr_score}%")
        st.success(f"Custom GB: {gb_score}% (+{round(gb_score-lr_score,2)}%)")
    with mc2:
        st.markdown("**Disease Classification**")
        st.error(f"Basic RF: {rf_accuracy}%")
        st.success(f"Custom GBC: {xgb_accuracy}% (+{round(xgb_accuracy-rf_accuracy,2)}%)")
    with mc3:
        st.markdown("**User Matching**")
        st.error(f"Basic KNN: {knn_cf_score}%")
        st.success(f"Custom SVD+GBC: {custom_cf_score}% (+{round(custom_cf_score-knn_cf_score,2)}%)")

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🍽️ Recommendations",
    "🤝 Collaborative Filter",
    "🏥 Disease Filter",
    "📅 Meal Plan",
    "📊 Calorie Tracker",
    "❤️ My Preferences",
    "🏆 Top Healthy Foods",
    "🔬 Food Clusters"
])

# -------------------------
# TAB 1: RECOMMENDATIONS
# -------------------------
with tab1:
    st.subheader("🍽️ Personalised Recommendations")
    if st.session_state['liked_foods']:
        st.info(f"✨ Recommendations are personalised based on your {len(st.session_state['liked_foods'])} liked foods")

    if get_recs:
        with st.spinner("Finding best foods for you..."):
            is_veg = diet_pref if diet_pref != "Both" else None
            recs   = recommend(df, scaler, NUTRITION_FEATURES, goal, daily_calories,
                               is_veg, top_n,
                               st.session_state['liked_foods'],
                               st.session_state['disliked_foods'])

        if recs.empty:
            st.warning("No foods found. Try changing filters.")
        else:
            st.subheader(f"Top {len(recs)} Recommendations for {goal}")
            display_recs = recs.copy()
            for col in ['Health Score', 'Fitness Score']:
                if col in display_recs.columns:
                    display_recs[col] = display_recs[col].round(3)
            if 'Calories (kcal)' in display_recs.columns:
                display_recs['Calories (kcal)'] = display_recs['Calories (kcal)'].round(1)
            st.dataframe(display_recs, use_container_width=True, hide_index=True,
                         column_config={
                             "Match Score": st.column_config.ProgressColumn(
                                 "Match Score (%)", min_value=0, max_value=100),
                             "Health Score": st.column_config.NumberColumn("Health Score", format="%.3f")
                         })

            st.subheader("📋 Detailed View")
            for i, row in recs.iterrows():
                with st.expander(f"{i+1}. {row['Dish Name']}  —  Match: {row['Match Score']}%"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Calories", f"{row.get('Calories (kcal)','N/A')} kcal")
                    c2.metric("Protein",  f"{row.get('Protein (g)','N/A')} g")
                    c3.metric("Fibre",    f"{row.get('Fibre (g)','N/A')} g")
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Carbs",        f"{row.get('Carbohydrates (g)','N/A')} g")
                    c5.metric("Fats",         f"{row.get('Fats (g)','N/A')} g")
                    c6.metric("Health Score", f"{round(row.get('Health Score',0),3)}")
                    if 'region' in row and row['region'] != 'unknown':
                        st.write(f"📍 **Region:** {row['region']}  |  🍽️ **Course:** {row.get('course','N/A')}  |  😋 **Flavor:** {row.get('flavor_profile','N/A')}")
                    st.info(f"**Why this food?** {generate_reason(row, df)}")

                    # Like / Dislike buttons
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button(f"👍 Like", key=f"like_{i}_{row['Dish Name']}"):
                            if row['Dish Name'] not in st.session_state['liked_foods']:
                                st.session_state['liked_foods'].append(row['Dish Name'])
                                save_user_liked(st.session_state['username'],
                                                st.session_state['liked_foods'],
                                                st.session_state['disliked_foods'])
                                st.success("Added to liked foods!")
                    with b2:
                        if st.button(f"👎 Dislike", key=f"dislike_{i}_{row['Dish Name']}"):
                            if row['Dish Name'] not in st.session_state['disliked_foods']:
                                st.session_state['disliked_foods'].append(row['Dish Name'])
                                save_user_liked(st.session_state['username'],
                                                st.session_state['liked_foods'],
                                                st.session_state['disliked_foods'])
                                st.warning("Added to disliked foods!")
                    with b3:
                        if st.button(f"➕ Add to Log", key=f"log_{i}_{row['Dish Name']}"):
                            st.session_state['food_log'].append({
                                'Food':     row['Dish Name'],
                                'Calories': round(row.get('Calories (kcal)', 0), 1),
                                'Protein':  round(row.get('Protein (g)', 0), 1),
                                'Carbs':    round(row.get('Carbohydrates (g)', 0), 1),
                                'Fats':     round(row.get('Fats (g)', 0), 1)
                            })
                            save_food_log(st.session_state['username'],
                                          st.session_state['food_log'])
                            st.success("Added to today's log!")

            st.divider()
            st.subheader("📊 Calorie Comparison")
            st.bar_chart(recs[['Dish Name','Calories (kcal)']].set_index('Dish Name'))
            if all(c in recs.columns for c in ['Protein (g)','Carbohydrates (g)','Fats (g)']):
                st.subheader("📊 Macro Breakdown")
                st.bar_chart(recs[['Dish Name','Protein (g)','Carbohydrates (g)','Fats (g)']].set_index('Dish Name'))
    else:
        st.info("👈 Click **Get Recommendations** in the sidebar")

# -------------------------
# TAB 2: COLLABORATIVE FILTER
# -------------------------
with tab2:
    st.subheader("🤝 Collaborative Filtering")
    if get_recs:
        user_input_cf  = np.array([[age, weight, height, bmi, daily_calories]])
        user_scaled_cf = cf_scaler.transform(user_input_cf)
        distances, indices = cf_model.kneighbors(user_scaled_cf)
        similar_users  = df_diet_recs.iloc[indices[0]]
        common_diet    = similar_users['Diet_Recommendation'].mode()[0]

        user_svd       = svd.transform(user_input_cf)
        custom_enc     = gbc_cf.predict(user_svd)[0]
        diet_labels    = le_diet.classes_
        custom_diet    = diet_labels[custom_enc] if custom_enc < len(diet_labels) else common_diet

        col_b, col_c = st.columns(2)
        with col_b:
            st.markdown(f"### 🔵 Basic KNN — Score: {knn_cf_score}%")
            dcols = [c for c in ['Age','Weight_kg','BMI','Disease_Type','Diet_Recommendation']
                     if c in similar_users.columns]
            st.dataframe(similar_users[dcols].reset_index(drop=True),
                         use_container_width=True, hide_index=True)
            st.info(f"KNN Diet: **{common_diet}**")
        with col_c:
            st.markdown(f"### 🟢 Custom SVD+GBC — Score: {custom_cf_score}%")
            st.success(f"Custom Diet: **{custom_diet}**")

        st.divider()
        st.markdown("### 🍛 Recommended Foods")
        cf_filtered = df.copy()
        is_veg = diet_pref if diet_pref != "Both" else None
        if is_veg == 'Vegetarian':     cf_filtered = cf_filtered[cf_filtered['IsVeg'] == 1]
        elif is_veg == 'Non-Vegetarian': cf_filtered = cf_filtered[cf_filtered['IsVeg'] == 0]
        if custom_diet == 'Low_Carb':
            cf_filtered = cf_filtered[cf_filtered['Carbohydrates (g)'] <= cf_filtered['Carbohydrates (g)'].quantile(0.40)]
        elif custom_diet == 'Low_Sodium':
            cf_filtered = cf_filtered[cf_filtered['Sodium (mg)'] <= cf_filtered['Sodium (mg)'].quantile(0.35)]
        elif 'Protein' in custom_diet:
            cf_filtered = cf_filtered[cf_filtered['Protein (g)'] >= cf_filtered['Protein (g)'].quantile(0.60)]
        cf_result = cf_filtered.sort_values('Health Score', ascending=False).head(top_n)
        cols = [c for c in ['Dish Name','Calories (kcal)','Protein (g)','Carbohydrates (g)','Fibre (g)','Health Score']
                if c in cf_result.columns]
        st.dataframe(cf_result[cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.info("👈 Click **Get Recommendations** in the sidebar")

# -------------------------
# TAB 3: DISEASE FILTER
# -------------------------
with tab3:
    st.subheader("🏥 Disease-Based Food Recommendations")
    if disease == 'None':
        st.warning("Select a health condition from the sidebar.")
    else:
        st.markdown(f"**{disease}** — {DISEASE_RULES[disease]['description']}")
        mc1, mc2 = st.columns(2)
        mc1.error(f"🔵 Basic RF: {rf_accuracy}%")
        mc2.success(f"🟢 Custom GBC: {xgb_accuracy}% (+{round(xgb_accuracy-rf_accuracy,2)}%)")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ✅ Foods to Eat")
            if get_disease:
                is_veg = diet_pref if diet_pref != "Both" else None
                disease_recs = filter_by_disease(df, df_disease_food, disease, is_veg, top_n)
                if disease_recs.empty:
                    st.warning("No matching foods found.")
                else:
                    st.dataframe(disease_recs, use_container_width=True, hide_index=True)
                    if disease == 'Diabetes':
                        st.markdown("#### 🩺 Glycemic Index Reference")
                        gi = df_disease_food[df_disease_food['Suitable for Diabetes']==1][
                            ['Food Name','Glycemic Index','Fiber Content','Calories']
                        ].sort_values('Glycemic Index').head(10)
                        st.dataframe(gi, use_container_width=True, hide_index=True)
                    if disease == 'Hypertension':
                        st.markdown("#### 🩺 Low Sodium Reference")
                        bp = df_disease_food[df_disease_food['Suitable for Blood Pressure']==1][
                            ['Food Name','Sodium Content','Potassium Content','Magnesium Content']
                        ].sort_values('Sodium Content').head(10)
                        st.dataframe(bp, use_container_width=True, hide_index=True)
            else:
                st.info("Click **Disease Food Filter** in the sidebar")
        with col_b:
            st.markdown("### ❌ Foods to Avoid")
            if get_disease:
                avoid_df = foods_to_avoid(df, disease)
                if not avoid_df.empty:
                    st.dataframe(avoid_df, use_container_width=True, hide_index=True)

        if disease == 'Diabetes':
            st.divider()
            st.markdown("### 🌲 Feature Importance — Custom GBC")
            fi_df = pd.DataFrame({'Feature': custom_rf_features,
                                  'Importance': xgb_model.feature_importances_}
                                 ).sort_values('Importance', ascending=False)
            st.bar_chart(fi_df.set_index('Feature'))

# -------------------------
# TAB 4: MEAL PLAN
# -------------------------
with tab4:
    st.subheader("📅 Personalized Day Meal Plan")
    if get_meal_plan:
        with st.spinner("Building meal plan..."):
            macro_targets = get_macro_targets(df_meal_plan, age, gender, bmi,
                                              disease if disease != 'None' else None)
            is_veg = diet_pref if diet_pref != "Both" else None
            meal_plan, targets = generate_meal_plan(df, macro_targets, is_veg,
                                                    disease if disease != 'None' else None)

        st.markdown("### 🎯 Daily Macro Targets")
        st.caption(f"Plan Type: **{targets['meal_plan_type']}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calories",      f"{targets['calories']} kcal")
        m2.metric("Protein",       f"{targets['protein']} g")
        m3.metric("Carbohydrates", f"{targets['carbs']} g")
        m4.metric("Fats",          f"{targets['fats']} g")
        st.divider()

        cal_split = pd.DataFrame({'Meal': ['Breakfast','Lunch','Dinner'],
                                  'Calories': [round(targets['calories']*0.25),
                                               round(targets['calories']*0.40),
                                               round(targets['calories']*0.35)]})
        st.bar_chart(cal_split.set_index('Meal'))
        st.divider()

        total_cals = 0
        for meal, foods_df in meal_plan.items():
            icon = {'Breakfast':'🌅','Lunch':'☀️','Dinner':'🌙'}.get(meal,'🍽️')
            meal_target = round(targets['calories'] * (0.25 if meal=='Breakfast' else 0.40 if meal=='Lunch' else 0.35))
            st.markdown(f"### {icon} {meal} — Target: {meal_target} kcal")
            if foods_df.empty:
                st.warning(f"No foods found for {meal}.")
            else:
                st.dataframe(foods_df.round(1), use_container_width=True, hide_index=True)
                # Add all to log
                if st.button(f"➕ Add {meal} to Log", key=f"log_meal_{meal}"):
                    for _, row in foods_df.iterrows():
                        st.session_state['food_log'].append({
                            'Food':     row['Dish Name'],
                            'Calories': round(row.get('Calories (kcal)',0),1),
                            'Protein':  round(row.get('Protein (g)',0),1),
                            'Carbs':    round(row.get('Carbohydrates (g)',0),1),
                            'Fats':     round(row.get('Fats (g)',0),1)
                        })
                    save_food_log(st.session_state['username'], st.session_state['food_log'])
                    st.success(f"{meal} added to your calorie tracker!")
                if 'Calories (kcal)' in foods_df.columns:
                    meal_total  = foods_df['Calories (kcal)'].sum()
                    total_cals += meal_total
                    st.caption(f"Total: **{round(meal_total)} kcal**")
            st.divider()
        st.success(f"✅ Total planned: **{round(total_cals)} kcal** | Target: **{targets['calories']} kcal**")
    else:
        st.info("👈 Click **Generate Meal Plan** in the sidebar")

# -------------------------
# TAB 5: CALORIE TRACKER
# -------------------------
with tab5:
    st.subheader("📊 Daily Calorie Tracker")
    st.caption("Track your food intake for the day")

    # Manual food add
    st.markdown("### ➕ Add Food Manually")
    col_search, col_btn = st.columns([3, 1])
    with col_search:
        food_to_add = st.selectbox("Search food:", [''] + sorted(df['Dish Name'].tolist()))
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Add Food", use_container_width=True):
            if food_to_add:
                food_data = df[df['Dish Name'] == food_to_add].iloc[0]
                st.session_state['food_log'].append({
                    'Food':     food_to_add,
                    'Calories': round(food_data.get('Calories (kcal)', 0), 1),
                    'Protein':  round(food_data.get('Protein (g)', 0), 1),
                    'Carbs':    round(food_data.get('Carbohydrates (g)', 0), 1),
                    'Fats':     round(food_data.get('Fats (g)', 0), 1)
                })
                save_food_log(st.session_state['username'], st.session_state['food_log'])
                st.success(f"{food_to_add} added to log!")

    st.divider()

    # Show log
    if st.session_state['food_log']:
        st.markdown("### 📋 Today's Food Log")
        log_df     = pd.DataFrame(st.session_state['food_log'])
        total_cals = log_df['Calories'].sum()
        total_prot = log_df['Protein'].sum()
        total_carb = log_df['Carbs'].sum()
        total_fats = log_df['Fats'].sum()

        # Progress metrics
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total Calories", f"{round(total_cals)} kcal",
                  f"{round(daily_calories - total_cals)} remaining")
        t2.metric("Total Protein",  f"{round(total_prot)} g")
        t3.metric("Total Carbs",    f"{round(total_carb)} g")
        t4.metric("Total Fats",     f"{round(total_fats)} g")

        # Calorie progress bar
        progress = min(total_cals / daily_calories, 1.0)
        st.markdown(f"**Calorie Progress: {round(total_cals)} / {daily_calories} kcal**")
        st.progress(progress)
        if total_cals > daily_calories:
            st.warning(f"⚠️ You have exceeded your daily calorie target by {round(total_cals - daily_calories)} kcal")
        elif total_cals >= daily_calories * 0.90:
            st.success("✅ You are close to your daily calorie target")
        else:
            st.info(f"💡 {round(daily_calories - total_cals)} kcal remaining for today")

        st.dataframe(log_df, use_container_width=True, hide_index=True)

        # Macro breakdown chart
        st.subheader("📊 Macro Breakdown")
        macro_chart = pd.DataFrame({
            'Macro':  ['Protein', 'Carbs', 'Fats'],
            'Amount': [round(total_prot), round(total_carb), round(total_fats)]
        })
        st.bar_chart(macro_chart.set_index('Macro'))

        # Clear log
        if st.button("🗑️ Clear Today's Log"):
            st.session_state['food_log'] = []
            save_food_log(st.session_state['username'], [])
            st.rerun()
    else:
        st.info("No foods logged yet. Add foods from recommendations or search above.")

# -------------------------
# TAB 6: MY PREFERENCES
# -------------------------
with tab6:
    st.subheader("❤️ My Food Preferences")
    st.caption("Your liked and disliked foods are used to personalise your recommendations")

    col_liked, col_disliked = st.columns(2)

    with col_liked:
        st.markdown("### 👍 Liked Foods")
        if st.session_state['liked_foods']:
            for food in st.session_state['liked_foods']:
                c1, c2 = st.columns([3, 1])
                c1.write(f"✅ {food}")
                if c2.button("Remove", key=f"rm_like_{food}"):
                    st.session_state['liked_foods'].remove(food)
                    save_user_liked(st.session_state['username'],
                                    st.session_state['liked_foods'],
                                    st.session_state['disliked_foods'])
                    st.rerun()
        else:
            st.info("No liked foods yet. Like foods from the Recommendations tab.")

    with col_disliked:
        st.markdown("### 👎 Disliked Foods")
        if st.session_state['disliked_foods']:
            for food in st.session_state['disliked_foods']:
                c1, c2 = st.columns([3, 1])
                c1.write(f"❌ {food}")
                if c2.button("Remove", key=f"rm_dislike_{food}"):
                    st.session_state['disliked_foods'].remove(food)
                    save_user_liked(st.session_state['username'],
                                    st.session_state['liked_foods'],
                                    st.session_state['disliked_foods'])
                    st.rerun()
        else:
            st.info("No disliked foods yet.")

    if st.button("🗑️ Clear All Preferences"):
        st.session_state['liked_foods']    = []
        st.session_state['disliked_foods'] = []
        save_user_liked(st.session_state['username'], [], [])
        st.rerun()

# -------------------------
# TAB 7: TOP HEALTHY FOODS
# -------------------------
with tab7:
    st.subheader("🏆 Top Healthy Foods")
    veg_filter = st.radio("Filter:", ["All","Vegetarian Only","Non-Veg Only"], horizontal=True)
    top_result = df.copy()
    if veg_filter == "Vegetarian Only": top_result = top_result[top_result['IsVeg']==1]
    elif veg_filter == "Non-Veg Only":  top_result = top_result[top_result['IsVeg']==0]
    top_result = top_result.sort_values('Health Score', ascending=False)
    cols = [c for c in ['Dish Name','Calories (kcal)','Protein (g)','Fibre (g)',
                         'Health Score','Fitness Score','diet','region'] if c in top_result.columns]
    st.dataframe(top_result[cols].head(15).reset_index(drop=True), use_container_width=True, hide_index=True)

    st.subheader("🗺️ By Region")
    regions = sorted([r for r in df['region'].unique() if r != 'unknown'])
    if regions:
        selected_region = st.selectbox("Select Region:", regions)
        region_df = df[df['region']==selected_region].sort_values('Health Score', ascending=False)
        cols = [c for c in ['Dish Name','Calories (kcal)','Protein (g)',
                             'Health Score','course','diet'] if c in region_df.columns]
        st.dataframe(region_df[cols].head(10).reset_index(drop=True), use_container_width=True, hide_index=True)

# -------------------------
# TAB 8: CLUSTERS
# -------------------------
with tab8:
    st.subheader("🔬 Food Clusters (K-Means)")
    cluster_summary = (
        df.groupby('Cluster Label')[['Calories (kcal)','Protein (g)','Fibre (g)','Health Score']]
        .mean().round(2)
    )
    st.dataframe(cluster_summary, use_container_width=True)
    selected_cluster = st.selectbox("Explore cluster:", df['Cluster Label'].unique())
    cluster_foods = df[df['Cluster Label']==selected_cluster][
        ['Dish Name','Calories (kcal)','Protein (g)','Fibre (g)','Health Score']
    ].sort_values('Health Score', ascending=False).head(10)
    st.dataframe(cluster_foods.reset_index(drop=True), use_container_width=True, hide_index=True)
