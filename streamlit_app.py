import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error

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
    # No preprocessing here — load raw data directly
    return df

data = load_data()
cat_features = ['Brand', 'Model']

st.title("📱 Mobile Phone Price Predictor")

st.markdown("### Data Overview")
st.dataframe(data.head(), use_container_width=True)

st.markdown("### 📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=data, x="Brand", y="Price", ax=ax1)
    ax1.set_title("Price Distribution by Brand")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    # Assuming 'Camera (MP)' column exists; else adapt or skip
    if 'Camera (MP)' in data.columns:
        sns.scatterplot(data=data, x="Camera (MP)", y="Price", hue="Brand", ax=ax2)
        ax2.set_title("Camera MP vs Price")
    else:
        ax2.text(0.5, 0.5, "No Camera (MP) data", ha='center')
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

st.markdown("### 🎛 Predict Phone Price")

brand = st.sidebar.selectbox("Brand", sorted(data["Brand"].unique()))
model_name = st.sidebar.selectbox("Model", sorted(data[data["Brand"] == brand]["Model"].unique()))

# For other numerical features, use min/max from raw data or hardcode reasonable ranges:
storage = st.sidebar.slider("Storage (GB)", int(data['Storage'].min()), int(data['Storage'].max()), step=32)
ram = st.sidebar.slider("RAM (GB)", int(data['RAM'].min()), int(data['RAM'].max()), step=2)
screen = st.sidebar.slider("Screen Size (inches)", float(data['Screen Size (inches)'].min()), float(data['Screen Size (inches)'].max()), step=0.1)
battery = st.sidebar.slider("Battery Capacity (mAh)", int(data['Battery Capacity (mAh)'].min()), int(data['Battery Capacity (mAh)'].max()), step=100)
camera_mp = st.sidebar.slider("Camera (MP)", int(data['Camera (MP)'].min()), int(data['Camera (MP)'].max()), step=5)

input_df = pd.DataFrame([{
    "Brand": brand,
    "Model": model_name,
    "Storage": storage,
    "RAM": ram,
    "Screen Size (inches)": screen,
    "Battery Capacity (mAh)": battery,
    "Camera (MP)": camera_mp
}])

prediction = model.predict(input_df)[0]

st.sidebar.markdown("## 💰 Predicted Price:")
st.sidebar.markdown(f"### ${prediction:,.2f}")
