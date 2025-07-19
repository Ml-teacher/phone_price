import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re
import seaborn as sns

st.set_page_config(page_title="Phone Price Predictor", layout="wide")

# Add a header banner
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
    df.columns = df.columns.str.strip()

    # Convert to string first to avoid .str accessor error, then clean and convert to int
    df['Storage'] = df['Storage'].fillna("").astype(str).str.replace("GB", "").str.strip()
    df['Storage'] = pd.to_numeric(df['Storage'], errors='coerce').fillna(0).astype(int)

    df['RAM'] = df['RAM'].fillna("").astype(str).str.replace("GB", "").str.strip()
    df['RAM'] = pd.to_numeric(df['RAM'], errors='coerce').fillna(0).astype(int)

    df.rename(columns={'Battery Capacity (mAh)': 'Battery'}, inplace=True)

    def total_camera_mp(s):
        numbers = re.findall(r'\d+\.?\d*', str(s))
        return sum(float(n) for n in numbers)

    df['Total_Camera_MP'] = df['Camera (MP)'].apply(total_camera_mp)
    df.drop(columns=['Camera (MP)'], inplace=True)

    df.rename(columns={
        'Screen Size (inches)': 'Screen_Size',
        'Price ($)': 'Price'
    }, inplace=True)

    # Optional: fill or drop any missing values here if needed
    df.fillna(0, inplace=True)

    return df

data = load_data()
cat_features = ['Brand', 'Model']

st.title("📱 Mobile Phone Price Predictor")

st.markdown("### Data Overview")
st.dataframe(data.head(), use_container_width=True)

st.markdown("### 📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="Brand", y="Price", ax=ax1)
    ax1.set_title("Price Distribution by Brand")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=data, x="Total_Camera_MP", y="Price", hue="Brand", ax=ax2)
    ax2.set_title("Camera MP vs Price")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

st.markdown("### ⚙️ Model Training")

X = data.drop("Price", axis=1)
y = data["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost Regressor with early stopping
model = CatBoostRegressor(
    iterations=200,
    max_depth=2,
    verbose=0,
    early_stopping_rounds=50
)

# Fit model with categorical features by name
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test)
)

st.success("Model trained successfully!")

# Evaluate model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error on test set: {mse:.2f}")

st.markdown("### 🎛 Predict Phone Price")

brand = st.sidebar.selectbox("Brand", sorted(data["Brand"].unique()))
model_name = st.sidebar.selectbox("Model", sorted(data[data["Brand"] == brand]["Model"].unique()))
storage = st.sidebar.slider("Storage (GB)", 32, 512, step=32)
ram = st.sidebar.slider("RAM (GB)", 2, 24, step=2)
screen = st.sidebar.slider("Screen Size (inches)", 4.0, 7.5, step=0.1)
battery = st.sidebar.slider("Battery Capacity (mAh)", 2000, 6000, step=100)
camera_mp = st.sidebar.slider("Total Camera MP", 5, 200, step=5)

input_df = pd.DataFrame([{
    "Brand": brand,
    "Model": model_name,
    "Storage": storage,
    "RAM": ram,
    "Screen_Size": screen,
    "Battery": battery,
    "Total_Camera_MP": camera_mp
}])

prediction = model.predict(input_df)[0]

st.sidebar.markdown("## 💰 Predicted Price:")
st.sidebar.markdown(f"### ${prediction:,.2f}")
