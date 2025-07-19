import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re
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
    df.columns = df.columns.str.strip()
    # Convert Storage and RAM to integer
    df['Storage'] = df['Storage'].str.replace("GB", "").str.strip().astype(int)
    df['RAM'] = df['RAM'].str.replace("GB", "").str.strip().astype(int)
    # Rename battery column
    df.rename(columns={'Battery Capacity (mAh)': 'Battery'}, inplace=True)

    # Calculate total camera MP if the column exists
    if 'Camera (MP)' in df.columns:
        def total_camera_mp(s):
            numbers = re.findall(r'\d+\.?\d*', str(s))
            return sum(float(n) for n in numbers)
        df['Total_Camera_MP'] = df['Camera (MP)'].apply(total_camera_mp)
        df.drop(columns=['Camera (MP)'], inplace=True)
    else:
        # If already exists or missing, do nothing or ensure it's int
        if 'Total_Camera_MP' not in df.columns:
            df['Total_Camera_MP'] = 0  # fallback value

    # Rename columns for consistency
    df.rename(columns={
        'Screen Size (inches)': 'Screen_Size',
        'Price ($)': 'Price'
    }, inplace=True)

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
    sns.scatterplot(data=data, x="Total_Camera_MP", y="Price", hue="Brand", ax=ax2)
    ax2.set_title("Camera MP vs Price")
    st.pyplot(fig2)

st.markdown("### ⚙️ Model Training")

# Features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Pools with categorical features for CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
eval_pool = Pool(X_test, y_test, cat_features=cat_features)

# Initialize and train model
model = CatBoostRegressor(
    iterations=200,
    max_depth=2,
    verbose=0,
    early_stopping_rounds=50
)
model.fit(train_pool, eval_set=eval_pool)

st.success("Model trained successfully!")

# Predict and evaluate
y_pred = model.predict(eval_pool)
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

# Use Pool to specify categorical features for prediction
input_pool = Pool(input_df, cat_features=cat_features)
prediction = model.predict(input_pool)[0]

st.sidebar.markdown("## 💰 Predicted Price:")
st.sidebar.markdown(f"### ${prediction:,.2f}")
