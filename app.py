import pickle
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth

st.set_page_config(page_title="Dashboard", page_icon="bar_chart:", layout="wide")

names = ["Angelica Lim", "Loren Buslon"]
usernames = ["Angelicalim", "Lorenbuslon"]

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

# Now you can use the hashed_passwords in your authenticator
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    "dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
      st.error("Username/Password is incorrect")

if authentication_status == None:
     st.warning("please enter your username and passwords")
        

if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.header(f"Welcome {name}")

    dashboard_page = st.Page(
       page="pages/Dashboard.py",
       title="Dashboard",
       icon=":material/bar_chart:",
       default=True,
    )

    Data_mining_1_page = st.Page(
        page="pages/Customers_segmentation.py",
        title="Customer Segmentation",
        icon=":material/support_agent:",
    )

    Data_mining_2_page = st.Page(
        page="pages/Sales_forecasting.py",
        title="Predict Analysis",
        icon=":material/collections_bookmark:",
    )

    User_guide_page = st.Page(
        page="pages/user_guide.py",
        title="User Guide",
        icon=":material/collections_bookmark:",
    )

    pg = st.navigation(
    {
        "Dashboard": [dashboard_page],
        "Data Mining": [Data_mining_1_page, Data_mining_2_page],
        "User Guide": [User_guide_page],
    }
    )

    st.logo("assets/logo.png")

    pg.run()

