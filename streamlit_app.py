import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import re

st.set_page_config(page_title="Phone Price Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Mobile phone pricee.csv")
    df.columns = df.columns.str.strip()
    df['Storage'] = df['Storage'].str.replace("GB", "").str.strip().astype(int)
    df['RAM'] = df['RAM'].str.replace("GB", "").str.strip().astype(int)
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

X = data.drop("Price", axis=1)
y = data["Price"]
cat_idx = [X.columns.get_loc(c) for c in cat_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_idx)

model = CatBoostRegressor(verbose=0)
model.fit(train_pool)

st.success("Model trained successfully!")

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
