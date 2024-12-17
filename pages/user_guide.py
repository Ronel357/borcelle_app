import streamlit as st

# Load CSS for custom styling
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("./assets/style1.css")

# Page title
st.title("Dashboard")

# Function to display an image with description
def display_image_section(image_src, alt_text, description):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image_src, caption=alt_text, use_column_width=True)
    with col2:
        st.write(description)

st.markdown("### Date Range")
display_image_section(
    "assets/img/date.png", 
    "Image 1", 
    """
    The **Date Range** Selection feature allows users to filter the visualizations and analytics based on a specific time period. 
    
    **Start Date**:  Use the date picker labeled "Start Date" to select the start of the desired date range for analysis.  
    **End Date**: Use the date picker labeled "End Date" to select the end of the desired date range.
    """
)


st.markdown("### Pick Region")
display_image_section(
    "assets/img/region.png", 
    "Image 2", 
     """
    The **Filter by Region** feature allows users to customize the dashboard by selecting specific regions for analysis. 
    This interactive filter ensures that the visualizations and metrics dynamically update to display data only for the chosen region(s).

    **How It Works:**

    - **Region Selection:**
        - On the left sidebar, you will find a dropdown labeled **Pick Your Region**.
        - Use this dropdown to select one or multiple regions from the available options.
        - The list of regions is dynamically populated based on the unique regions in the dataset.

    - **Filtering Logic:**
        - If no region is selected, the dashboard will display data for all regions.
        - If one or more regions are selected, the application filters the dataset to include only data from the selected region(s).
    """
)

st.markdown("### Pick State")
display_image_section(
    "assets/img/state.png", 
    "Image 3", 
    """
    The **Filter by State** feature allows users to refine their analysis by selecting specific states within the previously chosen regions. 
    This adds another layer of customization to the dashboard, ensuring that visualizations and metrics are focused on the states of interest.
    
    **How It Works:**
    
    - **State Selection:**
        - On the sidebar, locate the dropdown labeled **Pick Your State**.
        - Use the dropdown to select one or more states from the available options.
        - The list of states is dynamically generated based on the selected regions from the **Filter by Region** step.
    
    - **Filtering Logic:**
        - If no state is selected, the dashboard displays data for all states in the filtered regions.
        - If specific states are selected, the dataset is filtered to include only rows where the "state" column matches the selected states.
    """
)

st.markdown("### Pick City")
display_image_section(
    "assets/img/city.png", 
    "Image 4", 
     """
    The **Filter by City** feature allows users to drill down even further by selecting specific cities within the previously selected states. 
    This enables users to focus their analysis on a more granular level, ensuring that visualizations and metrics reflect data for the chosen cities.

    **How It Works:**

    - **City Selection:**
        - On the sidebar, find the dropdown labeled **Pick the City**.
        - Use the dropdown to select one or more cities from the list of available options.
        - The list of cities is dynamically populated based on the states selected in the **Filter by State** step.

    - **Filtering Logic:**
        - If no city is selected, the dashboard displays data for all cities in the selected states.
        - If one or more cities are selected, the dataset is filtered to include only rows where the "city" column matches the selected city(ies).
    """
)

st.markdown("### filter by Category or Sub-Category")
display_image_section(
    "assets/img/filter.png", 
    "Image 5", 
   """
    The **Filter by Category or Sub-Category** feature allows users to filter the dashboard data based on product categories or sub-categories. 
    This feature helps users analyze sales performance by specific product types, offering a more tailored view of the data.

    **How It Works:**

    - **Select Filter Type:**
        - In the sidebar, you will see a dropdown labeled **Filter by** with two options: **Category** and **Sub-Category**.
        - Choose either **Category** or **Sub-Category** depending on which level of granularity you want to filter by.
    """
)

st.markdown("### Example Pick Category ")
display_image_section(
    "assets/img/category.png", 
    "Image 6", 
    """
   
    - **Category Selection:**
        - If you select **Category**, another dropdown will appear labeled **Pick Category**. This will list all available categories in the dataset.
        - You can select one or more categories from the list.
        - If no categories are selected, the dataset will include all categories.
        - The selected category filter will be applied to the dataset, and the dashboard will show relevant data and visualizations based on the chosen categories.

    - **Sub-Category Selection:**
        - If you select **Sub-Category**, a dropdown labeled **Pick Sub-Category** will appear. It will display all available sub-categories.
        - You can select one or more sub-categories to filter by.
        - If no sub-categories are selected, the dataset will include all sub-categories.
    """
)

st.markdown("### For TreeMap")
display_image_section(
    "assets/img/treemap.png", 
    "Image 7", 
     """
    The **Select Region for TreeMap** feature allows users to choose a specific region to visualize the data in a TreeMap chart. 
    The TreeMap provides a clear, hierarchical representation of sales, customer information, or other metrics by region, making it easier to compare performance visually.

    **How It Works:**

    - **Region Selection:**
        - In the sidebar, you will find a dropdown labeled **Select Region For TreeMap**.
        - This dropdown contains four options: **West**, **South**, **East**, and **Central**.
        - Users can select any one of these regions to focus the TreeMap visualization on that specific region.

    - **TreeMap Update:**
        - Once a region is selected, the TreeMap will automatically update to display sales data, focusing on **Category and Sub-Category** within the chosen region.
        - The TreeMap will visualize sales performance by **Category and Sub-Category** for the selected region.
        - Each block in the TreeMap represents a different **Category or Sub-Category**, with the size of the block corresponding to the total sales for that category or sub-category.
    """
)

st.title("Clustering")

st.markdown("### Elbow Curve")
display_image_section(
    "assets/img/elbow.png", 
    "Image 8", 
    """
    The **Elbow Curve Settings** feature allows users to specify the maximum number of clusters to evaluate when determining the optimal number of clusters for a clustering algorithm, such as K-Means. 
    This setting is particularly useful for generating an Elbow Curve, a tool used to identify the ideal number of clusters by visualizing the relationship between the number of clusters and the within-cluster sum of squares (WCSS).

    **How It Works:**

    - **Select Maximum Number of Clusters:**
        - In the sidebar, under the **Elbow Curve Settings** section, you will see a slider labeled **Select Maximum Number of Clusters**.
        - Use this slider to choose the maximum number of clusters you want to evaluate. The available range is between **2 and 10**, with the default set to **6**.
        - The number selected will determine how many different cluster counts the algorithm will consider when plotting the Elbow Curve.
    """
)

st.markdown("### Number of Cluster")
display_image_section(
    "assets/img/clustering.png", 
    "Image 9", 
    """
    The **Number of Clusters** setting allows users to specify the exact number of clusters they want to use for the clustering algorithm, such as K-Means. 
    This feature enables users to define the level of granularity for their clustering analysis based on the optimal number of clusters determined earlier, or based on specific needs.

    **How It Works:**

    - **Select Number of Clusters:**
        - In the sidebar, under the **Clustering Settings** section, you will find the **"Select Number of Clusters"** input box.
        - Use the input field to enter the number of clusters you'd like to create. The value can range from **2** to the maximum number of clusters specified in the **Elbow Curve Settings** (which defaults to **6**).
        - You can adjust the number of clusters by entering a value directly or using the up/down arrows to increment or decrement the value.
    """
)

st.markdown("### Insight")
display_image_section(
    "assets/img/insight.png", 
    "Image 10", 
   """
    The **Insights and Recommendations** section allows users to view key findings and actionable insights based on the data analysis. This section can provide valuable information to help users make data-driven decisions.

    **How It Works:**

    - **Select Insights:**
        - In the sidebar under the **Insights** section, you will find a list of available insights to choose from. 
        - Users can select which insights they want to view, based on the analysis done on the dataset (e.g., clustering results, sales trends, customer behavior, etc.).
        
    - **View Insight:**
        - After selecting an insight, the dashboard will update to display the corresponding visualization, statistical analysis, or recommendation related to the selected insight.
        
    - **Turn Insights On or Off:**
        - If you prefer not to view an insight, you can deselect it. The dashboard will hide that specific insight, keeping the view uncluttered and focused on the data points you care about.
    """
)