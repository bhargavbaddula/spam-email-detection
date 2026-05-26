import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
data = {
    'Message': [
        'Win money now',
        'Hello friend',
        'Claim free prize',
        'Meeting tomorrow',
        'Congratulations you won lottery',
        'Free recharge available',
        'How are you doing',
        'Get cash reward now',
        'Let us study together',
        'Exclusive offer just for you'
    ],
    'Label': [
        'Spam',
        'Ham',
        'Spam',
        'Ham',
        'Spam',
        'Spam',
        'Ham',
        'Spam',
        'Ham',
        'Spam'
    ]
}
df = pd.DataFrame(data)
cv = CountVectorizer()
X = cv.fit_transform(df['Message'])
y = df['Label']
model = MultinomialNB()
model.fit(X, y)
st.title("📧 Spam Email Detection")
msg = st.text_input("Enter Message")
if st.button("Predict"):
    data = cv.transform([msg])
    prediction = model.predict(data)
    st.success(f"Prediction: {prediction[0]}")
