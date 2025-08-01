import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import datetime
import streamlit.components.v1 as components
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# 🔹 모델 & 토크나이저 불러오기
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
model = tf.keras.models.load_model("phishing_model.keras")

# 🔹 CSV 로그 초기화
LOG_FILE = "log.csv"
try:
    log_df = pd.read_csv(LOG_FILE)
except FileNotFoundError:
    log_df = pd.DataFrame(columns=["time", "input", "prediction", "score"])

# 🔹 Streamlit UI
st.title("🔍 피싱 URL 탐지 웹앱")
st.markdown("### URL을 입력하면 **피싱 가능성**을 판별합니다.")

user_input = st.text_area("URL을 입력하세요", "")

if st.button("판별하기"):
    if user_input.strip() == "":
        st.warning("⚠️ URL을 입력해주세요.")
    else:
        # 입력 전처리
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=100)

        # 예측
        prediction = model.predict(padded)[0][0]
        score = float(prediction) * 100

        # 결과 출력
        if prediction > 0.5:
            st.error(f"🚨 **피싱 가능성 높음!** (확률: {score:.2f}%)")
        else:
            st.success(f"✅ **안전한 URL일 가능성 높음** (확률: {score:.2f}%)")

        # 로그 저장
        new_log = pd.DataFrame([{
            "time": datetime.datetime.now(),
            "input": user_input,
            "prediction": "phishing" if prediction > 0.5 else "safe",
            "score": score
        }])
        log_df = pd.concat([log_df, new_log], ignore_index=True)
        try:
            log_df.to_csv(LOG_FILE, index=False)
            st.info("✅ 탐지 결과가 **log.csv**에 저장되었습니다.")
        except PermissionError:
            components.html(
                """
                <script>
                alert("❌ log.csv 파일에 접근할 수 없습니다. Excel을 닫고 다시
                시도해주세요!");
                </script>
                """,
                height=0
            )

        #st.info("✅ 탐지 결과가 **log.csv**에 #저장되었습니다.")

# 🔹 통계 시각화
st.subheader("📊 예측 통계")

if len(log_df) > 0:
    fig, ax = plt.subplots()
    log_df["prediction"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["#ff6b6b", "#4ecdc4"], ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)
