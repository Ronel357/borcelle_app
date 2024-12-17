import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="bar_chart:", layout="wide")

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

