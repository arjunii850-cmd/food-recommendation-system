# 🍛 AI-Based Food Recommendation System

A personalized Indian food recommendation system built using Machine Learning 
that suggests meals based on a user's health profile, fitness goals, 
nutritional requirements, and disease conditions.

## 🔗 Live App
[Click here to open the app](https://your-username-food-recommendation-system.streamlit.app)

## 📌 Project Overview
This project was developed as part of the NTCC (Industry Project) at 
Amity School of Engineering and Technology, Amity University.

The system takes a user's health profile as input and delivers 
personalized Indian food recommendations using content-based filtering, 
cosine similarity, and K-Means clustering.

## ✅ Features
- BMI, BMR and Daily Calorie Calculator
- Goal-based food recommendations (Weight Loss / Muscle Gain / Maintenance)
- Disease-based food filtering (Diabetes, Hypertension, Obesity, Anaemia, High Cholesterol)
- Personalized Meal Planning (Breakfast / Lunch / Dinner)
- Region-based Indian food recommendations (North / South / East / West)
- Vegetarian and Non-Vegetarian filtering
- K-Means food clustering by nutritional similarity
- Health Score and Fitness Score ranking
- Interactive charts and nutrition breakdown

## 🗂️ Datasets Used
| Dataset | Source | Purpose |
|---|---|---|
| Indian_Food_Nutrition_Processed.csv | Kaggle | Primary nutrition data |
| indian_food.csv | Kaggle | Food metadata (region, course, diet) |
| pred_food.csv | Kaggle | Disease suitability (Diabetes, BP) |
| diet_recommendations_dataset.csv | Kaggle | Disease to diet type mapping |
| Personalized_Diet_Recommendations.csv | Kaggle | Macro targets for meal planning |

## 🤖 Machine Learning Techniques
- **Cosine Similarity** — Content-based recommendation engine
- **K-Means Clustering** — Nutritional food grouping
- **StandardScaler** — Feature normalization
- **Percentile-based filtering** — Goal and disease specific food filtering
- **BMI Proximity Matching** — Profile matching for meal plan macro targets

## 🛠️ Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Google Colab (development)
- GitHub + Streamlit Cloud (deployment)

## 🚀 How to Run Locally
```bash
git clone https://github.com/your-username/AI-Based-Food-Recommendation-System
cd AI-Based-Food-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
AI-Based-Food-Recommendation-System/
│
├── app.py                                   # Main Streamlit application
│
├── requirements.txt                         # Python dependencies
│
├── datasets/
│   ├── Indian_Food_Nutrition_Processed.csv  # Primary nutrition dataset
│   ├── indian_food.csv                      # Food metadata (region, course, diet)
│   ├── pred_food.csv                        # Disease suitability dataset
│   ├── diet_recommendations_dataset.csv     # Disease to diet type mapping
│   └── Personalized_Diet_Recommendations.csv # Meal planning macro targets
│
└── README.md                                # Project documentation

## 👨‍💻 Developer
**Arjun Arora**
B.Tech CSE — Amity School of Engineering and Technology
Amity University Uttar Pradesh, Noida
NTCC Project — 2026
