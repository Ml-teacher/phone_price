import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

st.set_page_config(page_title="Phone Price Predictor", layout="wide")

# Header banner
st.markdown(
    """
    <div style='background-color:#4CAF50;padding:10px;border-radius:10px;margin-bottom:20px'>
        <h1 style='color:white;text-align:center;font-family:sans-serif;'>AiAcademy Course - Mobile Phone Price Predictor</h1>
    </div>
    """, unsafe_allow_html=True
)

@st.cache_data
def load_data():
    df = pd.read_csv("Mobile phone pricee.csv")
    return df

data = load_data()
cat_features = ['Brand', 'Model']

st.title("📱 Mobile Phone Price Predictor")

st.markdown("### Data Overview")
st.dataframe(data.head(), use_container_width=True)

st.markdown("### 📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.boxplot(data=data, x="Brand", y="Price", ax=ax1)
    ax1.set_title("Price Distribution by Brand")
    ax1.tick_params(axis='x', rotation=90)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=data, x="Total_Camera_MP", y="Price", hue="Brand", ax=ax2)
    ax2.set_title("Total Camera MP vs Price")
    ax2.tick_params(axis='x', rotation=90)
    st.pyplot(fig2)

st.markdown("### ⚙️ Model Training")

# Features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train CatBoost with categorical feature names directly
model = CatBoostRegressor(
    iterations=200,
    max_depth=2,
    verbose=0,
    early_stopping_rounds=50
)
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

st.success("Model trained successfully!")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error on test set: {mse:.2f}")

st.markdown("### 🎛 Predict Phone Price")

brand = st.sidebar.selectbox("Brand", sorted(data["Brand"].unique()))
model_name = st.sidebar.selectbox("Model", sorted(data[data["Brand"] == brand]["Model"].unique()))

storage = st.sidebar.slider("Storage (GB)", int(data['Storage'].min()), int(data['Storage'].max()), step=32)
ram = st.sidebar.slider("RAM (GB)", int(data['RAM'].min()), int(data['RAM'].max()), step=2)
screen_size = st.sidebar.slider("Screen Size (inches)", float(data['Screen_Size'].min()), float(data['Screen_Size'].max()), step=0.1)
battery = st.sidebar.slider("Battery Capacity (mAh)", int(data['Battery'].min()), int(data['Battery'].max()), step=100)
camera_mp = st.sidebar.slider("Total Camera MP", int(data['Total_Camera_MP'].min()), int(data['Total_Camera_MP'].max()), step=5)

input_df = pd.DataFrame([{
    "Brand": brand,
    "Model": model_name,
    "Storage": storage,
    "RAM": ram,
    "Screen_Size": screen_size,
    "Battery": battery,
    "Total_Camera_MP": camera_mp
}])

prediction = model.predict(input_df)[0]

st.sidebar.markdown("## 💰 Predicted Price:")
st.sidebar.markdown(f"### ${prediction:,.2f}")
