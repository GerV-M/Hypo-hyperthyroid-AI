import os
import pandas as pd
import streamlit as st
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

age = st.number_input("Возраст")
sex = st.selectbox("Гендер", [0, 1])
on_thyroxine = st.selectbox("На тироксине", [0, 1])
on_antithyroid_meds = st.selectbox("На антитиреоидных препаратах", [0, 1])
sick = st.selectbox("Болезнь", [0, 1])
pregnant = st.selectbox("Беременность", [0, 1])
thyroid_surgery = st.selectbox("Операция", [0, 1])
I131_treatment = st.selectbox("Терапия радиойодом", [0, 1])
lithium = st.selectbox("Литий", [0, 1])
goitre = st.selectbox("Зоб", [0, 1])
tumor = st.selectbox("Опухоль", [0, 1])
hypopituitary = st.selectbox("Гипопитуитаризм", [0, 1])
psych = st.selectbox("Психические заболевания", [0, 1])
TSH = st.number_input("TSH")
T3 = st.number_input("T3")
TT4 = st.number_input("TT4")
T4U = st.number_input("T4U")
FTI = st.number_input("FTI")

if st.button("Предсказать"):
    model = joblib.load(os.path.join(BASE_DIR, "RFMDump.joblib"))
    preprocessing = joblib.load(os.path.join(BASE_DIR, "RFMPreprocessing.joblib"))
    imputer = preprocessing["imputer"]
    scaler = preprocessing["scaler"]
    numeric_cols = preprocessing["numeric_cols"]
    categorical_cols = preprocessing["categorical_cols"]
    mode_values = preprocessing["mode_values"]

    sample = pd.DataFrame({"age":[age], "sex": [sex], "on_thyroxine": [on_thyroxine], "on_antithyroid_meds": [on_antithyroid_meds], "sick": [sick], "pregnant": [pregnant], "thyroid_surgery": [thyroid_surgery],
                           "I131_treatment": [I131_treatment], "lithium": [lithium], "goitre": [goitre], "tumor": [tumor], "hypopituitary": [hypopituitary], "psych": [psych],
                           "TSH":[TSH], "T3":[T3], "TT4":[TT4], "T4U":[T4U], "FTI":[FTI]})

    for col in categorical_cols:
        if col in sample.columns:
            sample[col] = sample[col].fillna(mode_values[col])

    sample[numeric_cols] = imputer.transform(sample[numeric_cols])
    sample[numeric_cols] = scaler.transform(sample[numeric_cols])

    pred = model.predict(sample)[0]

    result = {0: 'Здоров', 1: 'Гипертиреоз', 2: 'Гипотиреоз'}
    st.write(f"Предполагаемый диагноз: {result[pred]}")

image_path = os.path.join(BASE_DIR, "importance.png")
st.image(image_path, caption="Важность признаков")