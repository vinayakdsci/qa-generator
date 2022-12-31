import streamlit as st
import generator
from generator import *




text_ar = st.text_area("Enter text:")

if(st.button("Generate!")):
  text_ar = text_ar.replace("\n"," ")
  pairs = creator(text_ar)
  for pair in pairs:
    st.text(pair)



