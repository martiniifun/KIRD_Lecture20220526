import streamlit as st
st.text_input("이름을 입력하세요 : ", key="name")

# You can access the value at any point with:
st.session_state.name