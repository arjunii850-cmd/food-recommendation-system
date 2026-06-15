import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
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
# LOAD & PREPARE DATA
# =========================
@st.cache_data
def load_and_prepare():
    # --- Primary datasets ---
    df_nutrition = pd.read_csv('Indian_Food_Nutrition_Processed.csv')
    df_meta      = pd.read_csv('indian_food.csv')

    # --- New datasets ---
    df_disease_food = pd.read_csv('pred_food.csv')
    df_diet_recs    = pd.read_csv('diet_recommendations_dataset.csv')
    df_meal_plan    = pd.read_csv('Personalized_Diet_Recommendations.csv')

    # --- Clean nutrition ---
    df_nutrition.dropna(inplace=True)
    df_nutrition.drop_duplicates(inplace=True)
    if 'Dish Name' not in df_nutrition.columns:
        df_nutrition.rename(columns={df_nutrition.columns[0]: 'Dish Name'}, inplace=True)
    df_nutrition['Dish Name'] = df_nutrition['Dish Name'].str.strip().str.title()

    # --- Clean metadata ---
    df_meta.dropna(inplace=True)
    df_meta.drop_duplicates(inplace=True)
    df_meta.rename(columns={'name': 'Dish Name'}, inplace=True)
    df_meta['Dish Name'] = df_meta['Dish Name'].str.strip().str.title()
    for col in ['prep_time', 'cook_time']:
        df_meta[col] = pd.to_numeric(df_meta[col], errors='coerce')
    df_meta.dropna(subset=['prep_time', 'cook_time'], inplace=True)

    # --- Merge nutrition + metadata ---
    df = pd.merge(df_nutrition, df_meta, on='Dish Name', how='left')
    df['diet']           = df['diet'].fillna('unknown')
    df['region']         = df['region'].fillna('unknown')
    df['course']         = df['course'].fillna('unknown')
    df['flavor_profile'] = df['flavor_profile'].fillna('unknown')
    df['prep_time']      = df['prep_time'].fillna(0)
    df['cook_time']      = df['cook_time'].fillna(0)
    df.fillna(0, inplace=True)

    # --- Feature Engineering ---
    def minmax(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r != 0 else s * 0

    df['Protein Density'] = df['Protein (g)'] / df['Calories (kcal)'].replace(0, np.nan)
    df['Fiber Density']   = df['Fibre (g)']   / df['Calories (kcal)'].replace(0, np.nan)
    df.fillna(0, inplace=True)

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

    # --- Meal type mapping ---
    def map_meal_type(course):
        course = str(course).lower()
        if any(k in course for k in ['breakfast', 'morning', 'snack']):
            return 'Breakfast'
        elif any(k in course for k in ['main', 'lunch', 'rice', 'bread']):
            return 'Lunch'
        elif any(k in course for k in ['dinner', 'dessert', 'sweet']):
            return 'Dinner'
        else:
            return 'Lunch'
    df['Meal Type'] = df['course'].apply(map_meal_type)

    # --- Nutrition features ---
    NUTRITION_FEATURES = [c for c in [
        'Calories (kcal)', 'Carbohydrates (g)', 'Protein (g)',
        'Fats (g)', 'Free Sugar (g)', 'Fibre (g)',
        'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)',
        'Vitamin C (mg)', 'Folate (µg)',
        'Protein Density', 'Fiber Density'
    ] if c in df.columns]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[NUTRITION_FEATURES])

    # --- K-Means Clustering ---
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    cluster_labels = {0:'High-Calorie', 1:'High-Protein', 2:'High-Fibre', 3:'Balanced', 4:'Low-Calorie'}
    df['Cluster Label'] = df['Cluster'].map(cluster_labels)

    # --- Clean disease food dataset ---
    df_disease_food['Food Name'] = df_disease_food['Food Name'].str.strip().str.title()

    # --- Disease diet map ---
    disease_diet_map = (
        df_diet_recs.groupby('Disease_Type')['Diet_Recommendation']
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )

    # =============================================
    # ML MODEL 1: LINEAR REGRESSION
    # Predicts daily calorie intake from user profile
    # Trained on diet_recommendations_dataset.csv
    # =============================================
    lr_features = ['Age', 'Weight_kg', 'Height_cm', 'BMI']
    lr_data = df_diet_recs[lr_features + ['Daily_Caloric_Intake']].dropna()

    X_lr = lr_data[lr_features]
    y_lr = lr_data['Daily_Caloric_Intake']
    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(
        X_lr, y_lr, test_size=0.2, random_state=42
    )
    lr_model = LinearRegression()
    lr_model.fit(X_lr_train, y_lr_train)
    lr_score = round(lr_model.score(X_lr_test, y_lr_test) * 100, 2)

    # =============================================
    # ML MODEL 2: RANDOM FOREST CLASSIFIER
    # Predicts if a food is suitable for Diabetes
    # Trained on pred_food.csv
    # =============================================
    rf_features = ['Glycemic Index', 'Calories', 'Carbohydrates',
                   'Protein', 'Fat', 'Sodium Content',
                   'Potassium Content', 'Fiber Content']
    rf_features = [c for c in rf_features if c in df_disease_food.columns]

    rf_data = df_disease_food[rf_features + ['Suitable for Diabetes']].dropna()
    X_rf = rf_data[rf_features]
    y_rf = rf_data['Suitable for Diabetes']
    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
        X_rf, y_rf, test_size=0.2, random_state=42
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_rf_train, y_rf_train)
    rf_accuracy = round(rf_model.score(X_rf_test, y_rf_test) * 100, 2)

    # =============================================
    # ML MODEL 3: COLLABORATIVE FILTERING (KNN)
    # Finds similar users and recommends their foods
    # Trained on diet_recommendations_dataset.csv
    # =============================================
    cf_features = ['Age', 'Weight_kg', 'Height_cm', 'BMI', 'Daily_Caloric_Intake']
    cf_data = df_diet_recs[cf_features].dropna()
    cf_scaler = StandardScaler()
    cf_scaled = cf_scaler.fit_transform(cf_data)
    cf_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    cf_model.fit(cf_scaled)

    return (df, scaler, NUTRITION_FEATURES,
            df_disease_food, disease_diet_map, df_meal_plan,
            lr_model, lr_score,
            rf_model, rf_accuracy, rf_features,
            cf_model, cf_scaler, cf_data,
            df_diet_recs)

# =========================
# HEALTH CALC FUNCTIONS
# =========================
def calculate_bmi(weight, height_cm):
    h = height_cm / 100
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
# COSINE SIMILARITY RECOMMENDATION
# =========================
def recommend(df, scaler, features, goal, daily_calories, is_veg, top_n):
    filtered = df.copy()
    if is_veg == 'Vegetarian':
        filtered = filtered[filtered['IsVeg'] == 1]
    elif is_veg == 'Non-Vegetarian':
        filtered = filtered[filtered['IsVeg'] == 0]

    if goal == 'Weight Loss':
        filtered = filtered[
            (filtered['Calories (kcal)'] <= filtered['Calories (kcal)'].quantile(0.40)) &
            (filtered['Fibre (g)'] >= filtered['Fibre (g)'].quantile(0.50))
        ]
    elif goal == 'Muscle Gain':
        filtered = filtered[filtered['Protein (g)'] >= filtered['Protein (g)'].quantile(0.60)]
    elif goal == 'Maintenance':
        low  = filtered['Calories (kcal)'].quantile(0.20)
        high = filtered['Calories (kcal)'].quantile(0.80)
        filtered = filtered[
            (filtered['Calories (kcal)'].between(low, high)) &
            (filtered['Health Score'] >= filtered['Health Score'].quantile(0.50))
        ]

    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    filtered_scaled = scaler.transform(filtered[features])

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

    target_vector = scaler.transform([target_raw.values])
    similarity = cosine_similarity(target_vector, filtered_scaled).flatten()
    filtered['Match Score'] = (similarity * 100).round(1)

    result = filtered.sort_values('Match Score', ascending=False).head(top_n)
    display_cols = ['Dish Name', 'Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)',
                    'Fats (g)', 'Fibre (g)', 'Health Score', 'Fitness Score', 'Match Score']
    for c in ['diet', 'region', 'course', 'flavor_profile']:
        if c in result.columns:
            display_cols.append(c)
    display_cols = [c for c in display_cols if c in result.columns]
    return result[display_cols].reset_index(drop=True)

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
    'Diabetes': {
        'description': 'Focus on low glycemic index foods, high fibre, low sugar',
        'restrict':    ['Free Sugar (g)', 'Carbohydrates (g)'],
        'promote':     ['Fibre (g)', 'Protein (g)'],
        'pred_filter': 'Suitable for Diabetes',
        'diet_type':   'Low_Carb'
    },
    'Hypertension': {
        'description': 'Focus on low sodium foods, high potassium',
        'restrict':    ['Sodium (mg)'],
        'promote':     ['Fibre (g)', 'Calcium (mg)'],
        'pred_filter': 'Suitable for Blood Pressure',
        'diet_type':   'Low_Sodium'
    },
    'Obesity': {
        'description': 'Focus on low calorie, high fibre foods',
        'restrict':    ['Calories (kcal)', 'Fats (g)', 'Free Sugar (g)'],
        'promote':     ['Fibre (g)', 'Protein (g)'],
        'pred_filter': None,
        'diet_type':   'Balanced'
    },
    'High Cholesterol': {
        'description': 'Focus on low fat, high fibre foods',
        'restrict':    ['Fats (g)', 'Free Sugar (g)'],
        'promote':     ['Fibre (g)', 'Protein (g)'],
        'pred_filter': None,
        'diet_type':   'Low_Carb'
    },
    'Anaemia': {
        'description': 'Focus on high iron and Vitamin C foods',
        'restrict':    [],
        'promote':     ['Iron (mg)', 'Vitamin C (mg)', 'Folate (µg)'],
        'pred_filter': None,
        'diet_type':   'Balanced'
    }
}

def filter_by_disease(df, df_disease_food, disease, is_veg, top_n=10):
    if disease not in DISEASE_RULES:
        return pd.DataFrame()

    rules    = DISEASE_RULES[disease]
    filtered = df.copy()

    if is_veg == 'Vegetarian':
        filtered = filtered[filtered['IsVeg'] == 1]
    elif is_veg == 'Non-Vegetarian':
        filtered = filtered[filtered['IsVeg'] == 0]

    for col in rules['restrict']:
        if col in filtered.columns:
            filtered = filtered[filtered[col] <= filtered[col].quantile(0.35)]

    for col in rules['promote']:
        if col in filtered.columns:
            filtered = filtered[filtered[col] >= filtered[col].quantile(0.50)]

    pred_col = rules.get('pred_filter')
    if pred_col and pred_col in df_disease_food.columns:
        suitable_foods = df_disease_food[
            df_disease_food[pred_col] == 1
        ]['Food Name'].str.strip().str.title().tolist()
        mask = filtered['Dish Name'].apply(
            lambda name: any(s.lower() in name.lower() or name.lower() in s.lower()
                             for s in suitable_foods)
        )
        if mask.sum() > 0:
            filtered = filtered[mask]

    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    score = filtered['Health Score'].copy()
    for col in rules['promote']:
        if col in filtered.columns:
            r = filtered[col].max() - filtered[col].min()
            if r > 0:
                score += (filtered[col] - filtered[col].min()) / r
    filtered['Disease Score'] = score.round(3)
    filtered = filtered.sort_values('Disease Score', ascending=False)

    display_cols = ['Dish Name', 'Calories (kcal)', 'Protein (g)',
                    'Fibre (g)', 'Fats (g)', 'Sodium (mg)',
                    'Health Score', 'Disease Score']
    for c in ['diet', 'region', 'course', 'Free Sugar (g)', 'Iron (mg)', 'Vitamin C (mg)']:
        if c in filtered.columns:
            display_cols.append(c)
    display_cols = [c for c in display_cols if c in filtered.columns]
    return filtered[display_cols].head(top_n).reset_index(drop=True)

def foods_to_avoid(df, disease, top_n=5):
    rules = DISEASE_RULES.get(disease, {})
    avoid = df.copy()
    result = []
    for col in rules.get('restrict', []):
        if col in avoid.columns:
            worst = avoid.nlargest(top_n, col)[['Dish Name', col]]
            result.append(worst)
    if result:
        return pd.concat(result).drop_duplicates('Dish Name').head(top_n).reset_index(drop=True)
    return pd.DataFrame()

# =========================
# MEAL PLANNING
# =========================
def get_macro_targets(df_meal_plan, age, gender, bmi, disease=None):
    filtered = df_meal_plan.copy()
    filtered = filtered[filtered['Gender'].str.lower() == gender.lower()]
    if filtered.empty:
        filtered = df_meal_plan.copy()

    if disease and disease != 'None' and 'Chronic_Disease' in filtered.columns:
        disease_filtered = filtered[
            filtered['Chronic_Disease'].str.contains(disease, case=False, na=False)
        ]
        if not disease_filtered.empty:
            filtered = disease_filtered

    filtered = filtered.copy()
    filtered['BMI_diff'] = (filtered['BMI'] - bmi).abs()
    closest = filtered.nsmallest(10, 'BMI_diff')

    return {
        'calories':       round(closest['Recommended_Calories'].mean()),
        'protein':        round(closest['Recommended_Protein'].mean()),
        'carbs':          round(closest['Recommended_Carbs'].mean()),
        'fats':           round(closest['Recommended_Fats'].mean()),
        'meal_plan_type': closest['Recommended_Meal_Plan'].mode()[0]
            if not closest.empty else 'Balanced Diet'
    }

def generate_meal_plan(df, macro_targets, is_veg, disease=None):
    meal_calories = {
        'Breakfast': macro_targets['calories'] * 0.25,
        'Lunch':     macro_targets['calories'] * 0.40,
        'Dinner':    macro_targets['calories'] * 0.35
    }
    filtered = df.copy()
    if is_veg == 'Vegetarian':
        filtered = filtered[filtered['IsVeg'] == 1]
    elif is_veg == 'Non-Vegetarian':
        filtered = filtered[filtered['IsVeg'] == 0]

    if disease and disease != 'None' and disease in DISEASE_RULES:
        for col in DISEASE_RULES[disease]['restrict']:
            if col in filtered.columns:
                filtered = filtered[filtered[col] <= filtered[col].quantile(0.40)]

    meal_plan  = {}
    used_foods = set()

    for meal, cal_target in meal_calories.items():
        meal_foods = filtered[filtered['Meal Type'] == meal].copy()
        if len(meal_foods) < 3:
            meal_foods = filtered.copy()
        meal_foods = meal_foods[~meal_foods['Dish Name'].isin(used_foods)]
        if meal_foods.empty:
            meal_plan[meal] = pd.DataFrame()
            continue
        meal_foods = meal_foods.copy()
        meal_foods['Cal_diff'] = (meal_foods['Calories (kcal)'] - cal_target).abs()
        meal_foods = meal_foods.sort_values(['Cal_diff', 'Health Score'], ascending=[True, False])
        top_foods  = meal_foods.head(3)
        used_foods.update(top_foods['Dish Name'].tolist())
        cols = ['Dish Name', 'Calories (kcal)', 'Protein (g)',
                'Carbohydrates (g)', 'Fats (g)', 'Fibre (g)', 'Health Score']
        cols = [c for c in cols if c in top_foods.columns]
        meal_plan[meal] = top_foods[cols].reset_index(drop=True)

    return meal_plan, macro_targets

# =========================
# UI STARTS HERE
# =========================
st.title("🍛 AI-Based Food Recommendation System")
st.markdown("*Personalized Indian food recommendations based on your health profile*")

with st.spinner("Loading datasets and training ML models..."):
    try:
        (df, scaler, NUTRITION_FEATURES,
         df_disease_food, disease_diet_map, df_meal_plan,
         lr_model, lr_score,
         rf_model, rf_accuracy, rf_features,
         cf_model, cf_scaler, cf_data,
         df_diet_recs) = load_and_prepare()
        st.success(f"✅ All datasets loaded — {len(df)} Indian food items | 3 ML models trained")
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}\n\nMake sure all CSV files are in the same folder as app.py")
        st.stop()

st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("👤 Your Health Profile")
age       = st.sidebar.number_input("Age (years)",   min_value=10, max_value=100, value=21)
gender    = st.sidebar.selectbox("Gender",            ["Male", "Female"])
weight    = st.sidebar.number_input("Weight (kg)",   min_value=20.0, max_value=200.0, value=72.0, step=0.5)
height    = st.sidebar.number_input("Height (cm)",   min_value=100.0, max_value=250.0, value=175.0, step=0.5)
activity  = st.sidebar.selectbox("Activity Level",   ["Sedentary", "Moderate", "Active"])
goal      = st.sidebar.selectbox("Your Goal",        ["Weight Loss", "Muscle Gain", "Maintenance"])
diet_pref = st.sidebar.selectbox("Diet Preference",  ["Both", "Vegetarian", "Non-Vegetarian"])
disease   = st.sidebar.selectbox("Health Condition", ["None", "Diabetes", "Hypertension",
                                                       "Obesity", "High Cholesterol", "Anaemia"])
top_n     = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10)

get_recs      = st.sidebar.button("🔍 Get Recommendations",  use_container_width=True)
get_disease   = st.sidebar.button("🏥 Disease Food Filter",  use_container_width=True)
get_meal_plan = st.sidebar.button("📅 Generate Meal Plan",   use_container_width=True)

# =========================
# HEALTH SUMMARY
# =========================
bmi, bmi_cat   = calculate_bmi(weight, height)
bmr            = calculate_bmr(weight, height, age, gender)
daily_calories = calculate_daily_calories(bmr, activity, goal)

# --- ML MODEL 1: Linear Regression calorie prediction ---
lr_predicted_calories = round(lr_model.predict([[age, weight, height, bmi]])[0])

col1, col2, col3, col4 = st.columns(4)
col1.metric("BMI",             f"{bmi}",                 bmi_cat)
col2.metric("BMR",             f"{bmr} kcal",            "Base metabolic rate")
col3.metric("Daily Calories",  f"{daily_calories} kcal", "Formula-based")
col4.metric("ML Predicted",    f"{lr_predicted_calories} kcal", f"Linear Regression (R²: {lr_score}%)")

bmi_color = {"Underweight": "🔵", "Normal weight": "🟢", "Overweight": "🟡", "Obese": "🔴"}
st.markdown(f"**BMI Status:** {bmi_color.get(bmi_cat, '⚪')} {bmi_cat}")
st.caption(f"💡 Formula-based estimate: {daily_calories} kcal | ML model prediction: {lr_predicted_calories} kcal")

if disease != 'None':
    st.info(f"🏥 Health Condition: **{disease}** — {DISEASE_RULES[disease]['description']}")

st.divider()

# =========================
# ML MODELS INFO BAR
# =========================
with st.expander("🤖 ML Models Active in This App"):
    m1, m2, m3 = st.columns(3)
    m1.success(f"**Linear Regression**\nCalorie Prediction\nR² Score: {lr_score}%")
    m2.success(f"**Random Forest Classifier**\nDisease Food Suitability\nAccuracy: {rf_accuracy}%")
    m3.success(f"**KNN Collaborative Filtering**\nSimilar User Matching\nK=5 neighbours")

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🍽️ Recommendations",
    "🤝 Collaborative Filter",
    "🏥 Disease Filter",
    "📅 Meal Plan",
    "🏆 Top Healthy Foods",
    "🔬 Food Clusters"
])

# -------------------------
# TAB 1: COSINE SIMILARITY RECOMMENDATIONS
# -------------------------
with tab1:
    st.subheader("🍽️ Content-Based Recommendations")
    st.caption("Uses Cosine Similarity to match your nutrition target with food vectors")
    if get_recs:
        with st.spinner("Finding the best foods for you..."):
            is_veg = diet_pref if diet_pref != "Both" else None
            recs = recommend(df, scaler, NUTRITION_FEATURES, goal, daily_calories, is_veg, top_n)

        if recs.empty:
            st.warning("No foods found. Try changing diet preference or goal.")
        else:
            st.subheader(f"Top {len(recs)} Recommendations for {goal}")
            display_recs = recs.copy()
            display_recs['Health Score']    = display_recs['Health Score'].round(3)
            display_recs['Fitness Score']   = display_recs['Fitness Score'].round(3)
            display_recs['Calories (kcal)'] = display_recs['Calories (kcal)'].round(1)
            st.dataframe(
                display_recs, use_container_width=True, hide_index=True,
                column_config={
                    "Match Score": st.column_config.ProgressColumn(
                        "Match Score (%)", min_value=0, max_value=100),
                    "Health Score": st.column_config.NumberColumn("Health Score", format="%.3f"),
                }
            )

            st.subheader("📋 Detailed View")
            for i, row in recs.iterrows():
                with st.expander(f"{i+1}. {row['Dish Name']}  —  Match: {row['Match Score']}%"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Calories", f"{row.get('Calories (kcal)', 'N/A')} kcal")
                    c2.metric("Protein",  f"{row.get('Protein (g)', 'N/A')} g")
                    c3.metric("Fibre",    f"{row.get('Fibre (g)', 'N/A')} g")
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Carbs",        f"{row.get('Carbohydrates (g)', 'N/A')} g")
                    c5.metric("Fats",         f"{row.get('Fats (g)', 'N/A')} g")
                    c6.metric("Health Score", f"{round(row.get('Health Score', 0), 3)}")
                    if 'region' in row and row['region'] != 'unknown':
                        st.write(f"📍 **Region:** {row['region']}  |  🍽️ **Course:** {row.get('course','N/A')}  |  😋 **Flavor:** {row.get('flavor_profile','N/A')}")
                    st.info(f"**Why this food?** {generate_reason(row, df)}")

            st.divider()
            st.subheader("📊 Calorie Comparison")
            st.bar_chart(recs[['Dish Name', 'Calories (kcal)']].set_index('Dish Name'))
            if all(c in recs.columns for c in ['Protein (g)', 'Carbohydrates (g)', 'Fats (g)']):
                st.subheader("📊 Macro Breakdown")
                st.bar_chart(recs[['Dish Name', 'Protein (g)', 'Carbohydrates (g)', 'Fats (g)']].set_index('Dish Name'))
    else:
        st.info("👈 Set your profile in the sidebar and click **Get Recommendations**")

# -------------------------
# TAB 2: COLLABORATIVE FILTERING (KNN)
# -------------------------
with tab2:
    st.subheader("🤝 Collaborative Filtering Recommendations")
    st.caption("Finds users with similar health profiles and recommends foods they benefit from")

    if get_recs:
        with st.spinner("Finding similar user profiles..."):
            # Scale user input same as training data
            user_input = np.array([[age, weight, height, bmi, daily_calories]])
            user_scaled = cf_scaler.transform(user_input)

            # Find 5 nearest neighbours
            distances, indices = cf_model.kneighbors(user_scaled)
            similar_users = df_diet_recs.iloc[indices[0]]

            st.markdown("### 👥 Similar User Profiles Found")
            display_cols = ['Age', 'Weight_kg', 'Height_cm', 'BMI',
                            'Disease_Type', 'Daily_Caloric_Intake', 'Diet_Recommendation']
            display_cols = [c for c in display_cols if c in similar_users.columns]
            st.dataframe(similar_users[display_cols].reset_index(drop=True),
                         use_container_width=True, hide_index=True)

            # Get most common diet recommendation from similar users
            common_diet = similar_users['Diet_Recommendation'].mode()[0]
            st.success(f"📋 Based on similar users, recommended diet type: **{common_diet}**")

            # Map diet recommendation to food filtering
            st.markdown("### 🍛 Foods Recommended Based on Similar Users")
            cf_filtered = df.copy()
            is_veg = diet_pref if diet_pref != "Both" else None
            if is_veg == 'Vegetarian':
                cf_filtered = cf_filtered[cf_filtered['IsVeg'] == 1]
            elif is_veg == 'Non-Vegetarian':
                cf_filtered = cf_filtered[cf_filtered['IsVeg'] == 0]

            # Apply diet type filter
            if common_diet == 'Low_Carb':
                cf_filtered = cf_filtered[
                    cf_filtered['Carbohydrates (g)'] <= cf_filtered['Carbohydrates (g)'].quantile(0.40)
                ]
            elif common_diet == 'Low_Sodium':
                cf_filtered = cf_filtered[
                    cf_filtered['Sodium (mg)'] <= cf_filtered['Sodium (mg)'].quantile(0.35)
                ]
            elif common_diet == 'High_Protein':
                cf_filtered = cf_filtered[
                    cf_filtered['Protein (g)'] >= cf_filtered['Protein (g)'].quantile(0.60)
                ]

            cf_result = cf_filtered.sort_values('Health Score', ascending=False).head(top_n)
            cols = ['Dish Name', 'Calories (kcal)', 'Protein (g)',
                    'Carbohydrates (g)', 'Fibre (g)', 'Health Score']
            cols = [c for c in cols if c in cf_result.columns]
            st.dataframe(cf_result[cols].reset_index(drop=True),
                         use_container_width=True, hide_index=True)

            # Similarity distances
            st.markdown("### 📏 Similarity Distances to Nearest Users")
            dist_df = pd.DataFrame({
                'Neighbour': [f"User {i+1}" for i in range(len(distances[0]))],
                'Distance':  distances[0].round(4)
            })
            st.bar_chart(dist_df.set_index('Neighbour'))
    else:
        st.info("👈 Click **Get Recommendations** in the sidebar to run collaborative filtering")

# -------------------------
# TAB 3: DISEASE FILTER WITH RANDOM FOREST
# -------------------------
with tab3:
    st.subheader("🏥 Disease-Based Food Recommendations")
    st.caption("Uses Random Forest Classifier to predict food suitability for your condition")

    if disease == 'None':
        st.warning("Please select a health condition from the sidebar.")
    else:
        st.markdown(f"**Condition:** {disease} | **Guidance:** {DISEASE_RULES[disease]['description']}")
        st.info(f"🌲 Random Forest Classifier Accuracy: **{rf_accuracy}%** (trained on {len(df_disease_food)} foods)")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### ✅ Foods to Eat")
            if get_disease:
                with st.spinner(f"Filtering foods for {disease}..."):
                    is_veg = diet_pref if diet_pref != "Both" else None
                    disease_recs = filter_by_disease(df, df_disease_food, disease, is_veg, top_n)

                if disease_recs.empty:
                    st.warning("No matching foods found.")
                else:
                    st.dataframe(disease_recs, use_container_width=True, hide_index=True)

                    # Random Forest prediction for each recommended food
                    if disease == 'Diabetes':
                        st.markdown("#### 🌲 Random Forest Suitability Predictions")
                        st.caption("Predicting diabetes suitability for each recommended food using RF model")

                        preds = []
                        for _, row in disease_recs.iterrows():
                            food_match = df_disease_food[
                                df_disease_food['Food Name'].str.lower().str.contains(
                                    row['Dish Name'].lower()[:5], na=False
                                )
                            ]
                            if not food_match.empty:
                                food_features = food_match[rf_features].fillna(0).iloc[0:1]
                                pred = rf_model.predict(food_features)[0]
                                prob = rf_model.predict_proba(food_features)[0][1]
                                preds.append({
                                    'Food': row['Dish Name'],
                                    'RF Prediction': '✅ Suitable' if pred == 1 else '❌ Not Suitable',
                                    'Confidence': f"{round(prob*100, 1)}%"
                                })
                        if preds:
                            st.dataframe(pd.DataFrame(preds), use_container_width=True, hide_index=True)

                        st.markdown("#### 🩺 Glycemic Index Reference")
                        gi_foods = df_disease_food[df_disease_food['Suitable for Diabetes'] == 1][
                            ['Food Name', 'Glycemic Index', 'Fiber Content', 'Calories']
                        ].sort_values('Glycemic Index').head(10)
                        st.dataframe(gi_foods, use_container_width=True, hide_index=True)

                    if disease == 'Hypertension':
                        st.markdown("#### 🩺 Low Sodium Reference")
                        bp_foods = df_disease_food[df_disease_food['Suitable for Blood Pressure'] == 1][
                            ['Food Name', 'Sodium Content', 'Potassium Content', 'Magnesium Content']
                        ].sort_values('Sodium Content').head(10)
                        st.dataframe(bp_foods, use_container_width=True, hide_index=True)
            else:
                st.info("Click **Disease Food Filter** in the sidebar")

        with col_b:
            st.markdown("### ❌ Foods to Avoid")
            if get_disease:
                avoid_df = foods_to_avoid(df, disease, top_n=8)
                if not avoid_df.empty:
                    st.dataframe(avoid_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 📌 Nutritional Rules Applied")
        rules = DISEASE_RULES[disease]
        rc1, rc2 = st.columns(2)
        with rc1:
            st.error("**Nutrients to Restrict:**\n\n" +
                     "\n".join([f"• {r}" for r in rules['restrict']]) if rules['restrict'] else "None")
        with rc2:
            st.success("**Nutrients to Promote:**\n\n" +
                       "\n".join([f"• {p}" for p in rules['promote']]) if rules['promote'] else "None")

        # Feature importance from Random Forest
        if disease == 'Diabetes':
            st.divider()
            st.markdown("### 🌲 Random Forest Feature Importance")
            st.caption("Which nutritional factors matter most for diabetes suitability")
            importance_df = pd.DataFrame({
                'Feature':    rf_features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            st.bar_chart(importance_df.set_index('Feature'))

# -------------------------
# TAB 4: MEAL PLAN
# -------------------------
with tab4:
    st.subheader("📅 Personalized Day Meal Plan")
    st.caption("Macro targets matched from Personalized Diet Recommendations dataset using BMI proximity")

    if get_meal_plan:
        with st.spinner("Building your meal plan..."):
            macro_targets = get_macro_targets(
                df_meal_plan, age, gender, bmi,
                disease if disease != 'None' else None
            )
            is_veg = diet_pref if diet_pref != "Both" else None
            meal_plan, targets = generate_meal_plan(
                df, macro_targets, is_veg,
                disease if disease != 'None' else None
            )

        st.markdown("### 🎯 Your Daily Macro Targets")
        st.caption(f"Meal Plan Type: **{targets['meal_plan_type']}** — matched from dataset")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Daily Calories", f"{targets['calories']} kcal")
        m2.metric("Protein",        f"{targets['protein']} g")
        m3.metric("Carbohydrates",  f"{targets['carbs']} g")
        m4.metric("Fats",           f"{targets['fats']} g")

        st.divider()
        st.markdown("### 🕐 Calorie Distribution")
        cal_split = pd.DataFrame({
            'Meal':     ['Breakfast', 'Lunch', 'Dinner'],
            'Calories': [
                round(targets['calories'] * 0.25),
                round(targets['calories'] * 0.40),
                round(targets['calories'] * 0.35)
            ]
        })
        st.bar_chart(cal_split.set_index('Meal'))
        st.divider()

        meal_icons = {'Breakfast': '🌅', 'Lunch': '☀️', 'Dinner': '🌙'}
        total_cals = 0
        for meal, foods_df in meal_plan.items():
            icon = meal_icons.get(meal, '🍽️')
            meal_cal_target = round(targets['calories'] * (0.25 if meal == 'Breakfast' else 0.40 if meal == 'Lunch' else 0.35))
            st.markdown(f"### {icon} {meal} — Target: {meal_cal_target} kcal")
            if foods_df.empty:
                st.warning(f"No foods found for {meal}.")
            else:
                st.dataframe(foods_df.round(1), use_container_width=True, hide_index=True)
                if 'Calories (kcal)' in foods_df.columns:
                    meal_total = foods_df['Calories (kcal)'].sum()
                    total_cals += meal_total
                    st.caption(f"Total for {meal}: **{round(meal_total)} kcal**")
            st.divider()

        st.success(f"✅ Total planned: **{round(total_cals)} kcal** | Target: **{targets['calories']} kcal**")
    else:
        st.info("👈 Click **Generate Meal Plan** in the sidebar")

# -------------------------
# TAB 5: TOP HEALTHY FOODS
# -------------------------
with tab5:
    st.subheader("🏆 Top Healthy Foods")
    veg_filter = st.radio("Filter:", ["All", "Vegetarian Only", "Non-Veg Only"], horizontal=True)
    top_result = df.copy()
    if veg_filter == "Vegetarian Only": top_result = top_result[top_result['IsVeg'] == 1]
    elif veg_filter == "Non-Veg Only":  top_result = top_result[top_result['IsVeg'] == 0]
    top_result = top_result.sort_values('Health Score', ascending=False)
    cols = [c for c in ['Dish Name', 'Calories (kcal)', 'Protein (g)', 'Fibre (g)',
                         'Health Score', 'Fitness Score', 'diet', 'region'] if c in top_result.columns]
    st.dataframe(top_result[cols].head(15).reset_index(drop=True), use_container_width=True, hide_index=True)

    st.subheader("🗺️ By Region")
    regions = sorted([r for r in df['region'].unique() if r != 'unknown'])
    if regions:
        selected_region = st.selectbox("Select Region:", regions)
        region_df = df[df['region'] == selected_region].sort_values('Health Score', ascending=False)
        cols = [c for c in ['Dish Name', 'Calories (kcal)', 'Protein (g)',
                             'Health Score', 'course', 'diet'] if c in region_df.columns]
        st.dataframe(region_df[cols].head(10).reset_index(drop=True), use_container_width=True, hide_index=True)

# -------------------------
# TAB 6: CLUSTERS
# -------------------------
with tab6:
    st.subheader("🔬 Food Clusters (K-Means)")
    st.caption("Foods grouped by nutritional similarity using K-Means clustering (k=5)")
    cluster_summary = (
        df.groupby('Cluster Label')[['Calories (kcal)', 'Protein (g)', 'Fibre (g)', 'Health Score']]
        .mean().round(2)
    )
    st.dataframe(cluster_summary, use_container_width=True)

    selected_cluster = st.selectbox("Explore cluster:", df['Cluster Label'].unique())
    cluster_foods = df[df['Cluster Label'] == selected_cluster][
        ['Dish Name', 'Calories (kcal)', 'Protein (g)', 'Fibre (g)', 'Health Score']
    ].sort_values('Health Score', ascending=False).head(10)
    st.dataframe(cluster_foods.reset_index(drop=True), use_container_width=True, hide_index=True)
