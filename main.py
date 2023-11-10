import streamlit as st
import numpy as np
import pandas as pd

import home
 
import predict_multi
import predict_single
 

pages_dict = {"Home": home,
              "Student performance using single linear regression": predict_single,
              "Student performance using multiple linear regression": predict_multi

              }

st.sidebar.title("Navigation")
user_choice = st.sidebar.radio("Go to", tuple(pages_dict.keys()))
if user_choice == "Home":
  home.app()
else:
  selected_page = pages_dict[user_choice]
  selected_page.app()