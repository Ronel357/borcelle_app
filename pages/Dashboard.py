import streamlit as st
import plotly.express as px
import pandas as pd
import psycopg2
import os
import altair as alt
import warnings
import plotly.figure_factory as ff
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards


warnings.filterwarnings('ignore')


# Initialize PostgreSQL connection
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()

# Query execution function
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        return pd.DataFrame(data, columns=columns)

st.title(":bar_chart: Dashboard")

# Load CSS for custom styling
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("./assets/style.css")

# Fetch data from PostgreSQL
query = """
    SELECT 
        fs."order_date", 
        fs."sales", 
        fs."ship_mode", 
        p."category" AS "Category", 
        p."sub_category" AS "Sub-Category", 
        c."segment", 
        p."product_name", 
        fs."order_id", 
        fs."product_id", 
        fs."customer_id", 
        c."region", 
        c."state", 
        c."country", 
        c."city", 
        c."postal_code",
        c."customer_name"  
    FROM fact_sales fs
    JOIN dim_product p ON fs."product_id" = p."product_id"
    JOIN dim_customer c ON fs."customer_id" = c."customer_id"
    JOIN dim_time t ON fs."order_date" = t."order_date";
"""
df = run_query(query)

# Convert Order Date to datetime
df["order_date"] = pd.to_datetime(df["order_date"], format='%m/%d/%Y', errors='coerce')

# Date range selection
startDate = df["order_date"].min()
endDate = df["order_date"].max()

col1, col2 = st.columns([1, 2])  
with col1:
    st.subheader("Date Range Selection")
    date1 = st.date_input("Start Date", startDate)
    date2 = st.date_input("End Date", endDate)

    # Convert date1 and date2 to datetime64[ns]
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)


# Function to query ship mode percentages for a date range
def get_ship_modes(date1, date2):
    query = f"""
    SELECT ship_mode, 
           COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS percentage
    FROM fact_sales fs
    WHERE fs."order_date" BETWEEN '{date1}' AND '{date2}'
    GROUP BY ship_mode;
    """
    return run_query(query)

# Get ship modes data
ship_modes_df = get_ship_modes(date1, date2)

# Manually define ship modes and their percentages (this can be updated based on the query result)
first_class_percentage = ship_modes_df.loc[ship_modes_df['ship_mode'] == "First Class", "percentage"].values[0] if "First Class" in ship_modes_df['ship_mode'].values else 0
second_class_percentage = ship_modes_df.loc[ship_modes_df['ship_mode'] == "Second Class", "percentage"].values[0] if "Second Class" in ship_modes_df['ship_mode'].values else 0
standard_class_percentage = ship_modes_df.loc[ship_modes_df['ship_mode'] == "Standard Class", "percentage"].values[0] if "Standard Class" in ship_modes_df['ship_mode'].values else 0

def plot_pie(
        indicator_number=1.86,
        indicator_color="#228B22", 
        indicator_suffix="%",
        indicator_title="Current Ratio",
        max_bound=100,
        title_font_color="white",  
        number_font_color="white",  
):
    fig = go.Figure(
        go.Pie(
            labels=["", ""],  
            values=[indicator_number, max_bound - indicator_number],  
            hole=0.6, 
            marker=dict(
                colors=['#21D375', '#262730']  
            ),
            textinfo="none",  
            hoverinfo="label+percent", 
        )
    )
    
    # Add custom annotation for the percentage number in the center
    fig.add_annotation(
        text=f"{indicator_number:.0f}%",  
        font=dict(size=24, color=number_font_color),  
        showarrow=False,  
        align="center", 
        x=0.5,  
        y=0.5,  
        xanchor="center", 
        yanchor="middle",  
    )
    
    # Add the indicator title at the bottom
    fig.add_annotation(
        text=indicator_title,
        font=dict(size=18, color=title_font_color),  
        showarrow=False, 
        align="center", 
        x=0.5,  
        y=-0.1,  
        xanchor="center", 
        yanchor="top",  
    )
    
    fig.update_layout(
        height=250,  # Adjust height
        width=350,  # Adjust width
        margin=dict(t=5, b=50, l=10, r=10),  # Adjust margins to make space for title
        showlegend=False  # Hide the legend
    )
    return fig



# Example usage for displaying the pie chart with percentage in the center and title at the bottom
with col2:

    cols = st.columns(3)  # We have 3 modes: First Class, Second Class, and Standard Class

    # Create a pie chart for "First Class"
    with cols[0]:
        fig = plot_pie(
            indicator_number=first_class_percentage,
            indicator_color="#3498DB",
            indicator_suffix="%",
            indicator_title="First Class",
            max_bound=100
        )
        st.plotly_chart(fig, use_container_width=True)

    # Create a pie chart for "Second Class"
    with cols[1]:
        fig = plot_pie(
            indicator_number=second_class_percentage,
            indicator_color="#3498DB",
            indicator_suffix="%",
            indicator_title="Second Class",
            max_bound=100
        )
        st.plotly_chart(fig, use_container_width=True)

    # Create a pie chart for "Standard Class"
    with cols[2]:
        fig = plot_pie(
            indicator_number=standard_class_percentage,
            indicator_color="#3498DB",
            indicator_suffix="%",
            indicator_title="Standard Class",
            max_bound=100
        )
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

if date1 > date2:
        st.error("Start Date cannot be after End Date. Please adjust the dates.")
        filtered_df = pd.DataFrame()  # Empty DataFrame in case of error
else:
        filtered_df = df[(df["order_date"] >= date1) & (df["order_date"] <= date2)].copy()


# Sidebar filters
st.sidebar.header("Choose Your Filter: ")

# Filter by Region
region = st.sidebar.multiselect("Pick Your Region", df["region"].unique())
df2 = df if not region else df[df["region"].isin(region)]

# Filter by State
state = st.sidebar.multiselect("Pick Your State", df2["state"].unique())
df3 = df2 if not state else df2[df2["state"].isin(state)]

# Filter by City
city = st.sidebar.multiselect("Pick the City", df3["city"].unique())
filtered_df = df3 if not city else df3[df3["city"].isin(city)]

# Filter by Category or Sub-Category
filter_type = st.sidebar.selectbox("Filter by", ["Category", "Sub-Category"])

# Category and Sub-Category selection
if filter_type == "Category":
    category = st.sidebar.multiselect("Pick Category", filtered_df["Category"].unique())
    filtered_df = filtered_df[filtered_df["Category"].isin(category)] if category else filtered_df
    # Calculate category wise sales
    sales_df = filtered_df.groupby(by=["Category"], as_index=False)["sales"].sum()
    sales_label = "Category"
else:  # Sub-Category
    sub_category = st.sidebar.multiselect("Pick Sub-Category", filtered_df["Sub-Category"].unique())
    filtered_df = filtered_df[filtered_df["Sub-Category"].isin(sub_category)] if sub_category else filtered_df
    # Calculate sub-category wise sales
    sales_df = filtered_df.groupby(by=["Sub-Category"], as_index=False)["sales"].sum()
    sales_label = "Sub-Category"

# Metrics Calculation
total_unique_items = filtered_df["product_name"].nunique()
unique_sales = filtered_df.groupby("product_name", as_index=False)["sales"].sum()
total_sales = unique_sales["sales"].sum()
median_sales = unique_sales["sales"].median()
max_price = filtered_df["sales"].max()
min_price = filtered_df["sales"].min()
top_selling_product = unique_sales.sort_values(by="sales", ascending=False).iloc[0]
max_price_product = filtered_df[filtered_df["sales"] == max_price].iloc[0]["product_name"]
min_price_product = filtered_df[filtered_df["sales"] == min_price].iloc[0]["product_name"]
price_range = max_price - min_price

# Display metrics

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="Total Number of All Items", value=total_unique_items, delta="All Unique Items in Dataset")
with col2:
    st.metric(label="Sum of Product Price USD", value=f"${total_sales:,.2f}", delta=f"Median: ${median_sales:,.0f}")
with col3:
    st.metric(label="Top Product Sales", value=f"${top_selling_product['sales']:,.0f}", delta=top_selling_product['product_name'])
with col4:
    st.metric(label="Minimum Price", value=f"${min_price:,.0f}", delta=min_price_product)
with col5:
    st.metric(label="Maximum Price", value=f"${max_price:,.0f}", delta=max_price_product)

# Apply custom styles to the metric cards
style_metric_cards(background_color="#262730", border_left_color="#21D375", border_color="#00060a")

# Sales Visualizations
col111, col222 = st.columns((2, 2))  # Ensure a tuple of two values for column proportions

with col111:
    
    # Create the bar chart
    fig = px.bar(
        sales_df,
        x=sales_label,
        y="sales",
        text=['${:,.2f}'.format(x) for x in sales_df["sales"]],
        template="seaborn"
    )

    fig.update_traces(
        marker_color="#21D375",
        hovertemplate="<b>%{x}</b><br>Sales: %{y:$,.2f}<extra></extra>"
    )
    

    # Remove gridlines
    fig.update_layout(
        title=f"{sales_label} Wise Sales",
        plot_bgcolor="#262730",
        paper_bgcolor="#262730",
        xaxis=dict(showgrid=False),  # Remove vertical gridlines
        yaxis=dict(showgrid=False),  # Remove horizontal gridlines
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

with col222:
    # Create the pie chart
    fig = px.pie(
        filtered_df,
        values="sales",
        names="region",
        hole=0.5, # Create a donut chart
        color_discrete_sequence=px.colors.sequential.Greens_r
    )

    # Add custom text and position
    fig.update_traces(
        text=filtered_df["region"],
        textposition="outside",
    )

    # Adjust legend position
    fig.update_layout(
        title="Region Wise Sales",
        plot_bgcolor="#262730",
        paper_bgcolor="#262730",
        legend=dict(
            orientation="h",       
            yanchor="top",         
            y=-0.1,                
            xanchor="center",      
            x=0.5                 
        )
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

# Data Download Section
cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Category_ViewData"):
        st.write(sales_df.style.background_gradient(cmap="Blues"))
        csv = sales_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="category.csv", mime="text/csv",
                           help='Click here to download the data as a csv file')

with cl2:
    with st.expander("Region_ViewData"):
        region = filtered_df.groupby(by="region", as_index=False)["sales"].sum()
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="region.csv", mime="text/csv",
                           help='Click here to download the data as a csv file')


# Filter the data based on the selected date range
filtered_df = df[(df['order_date'] >= date1) & (df['order_date'] <= date2)]

if not filtered_df.empty:
    # Time series analysis using filtered data
    # Extract month and year from the 'order_date'
    filtered_df['month'] = filtered_df['order_date'].dt.strftime('%B')  # Full month names
    filtered_df['year'] = filtered_df['order_date'].dt.year             # Year

    # Group data by year and month, aggregating sales
    grouped_df = (
        filtered_df.groupby(['year', 'month'], sort=False)['sales']
        .sum()
        .reset_index()
    )

    # Map months to their numerical order for correct sorting
    month_order = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    grouped_df['month_num'] = grouped_df['month'].map(month_order)

    # Sort by year and month number
    grouped_df.sort_values(by=['year', 'month_num'], inplace=True)

    # Define a color palette for consistent colors
    color_palette = px.colors.qualitative.Set1  # Use a qualitative color scheme
    year_colors = {year: color_palette[i % len(color_palette)] for i, year in enumerate(grouped_df['year'].unique())}

    # Create the figure
    fig2 = go.Figure()

    # Add line traces (not in the legend)
    for year in grouped_df['year'].unique():
        year_data = grouped_df[grouped_df['year'] == year]
        fig2.add_scatter(
            x=year_data['month'],
            y=year_data['sales'],
            mode="lines",  # Lines only
            line=dict(shape="spline", color=year_colors[year]),  # Use consistent color
            name=str(year),  # Internal label
            legendgroup=str(year),  # Group legend items
            showlegend=False  # Hide the line trace in the legend
        )

    # Add marker-only traces (visible in the legend)
    for year in grouped_df['year'].unique():
        year_data = grouped_df[grouped_df['year'] == year]
        fig2.add_scatter(
            x=year_data['month'],
            y=year_data['sales'],
            mode="markers",  # Markers only
            marker=dict(size=8, color=year_colors[year]),  # Use consistent color
            name=str(year),  # Legend label
            legendgroup=str(year),  # Group legend items
            showlegend=True  # Show in the legend
        )

    # Customize layout
    fig2.update_layout(
        title="Time Series Analysis of Sales by Year",
        plot_bgcolor="#262730",
        paper_bgcolor="#262730",
        xaxis=dict(
            tickmode="array",
            tickvals=grouped_df["month"].unique(),
            showgrid=False,
            title="Month",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            title="Sales",
            zeroline=False,
        ),
        legend=dict(
            title="Year"  # Optional legend title
        ),
        margin=dict(t=50, b=50, l=50, r=50),
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("No data available for the selected date range.")


# Display the data under an expander
with st.expander("View Data of TimeSeries:"):
    # Display the data with a background gradient
    st.write(df.T.style.background_gradient(cmap="Blues"))
    
    # Convert the DataFrame to CSV for download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data", data=csv, file_name="TimeSeries.csv", mime="text/csv")


col1, col2 = st.columns(2)

with col1:
    try:
        # Segment-wise Sales Pie Chart
        fig1 = px.pie(filtered_df, values="sales", names="segment", template="plotly_dark")

        # Define a green color palette for the segments
        segment_color_map = {
            "Home Office": "#006400",
            "Office Supplies": "#228B22",
            "Corporate": "#21D375",
            "Consumer": "#98FB98"
        }

        # Update the pie chart with the colors based on the dictionary
        fig1.update_traces(
            text=filtered_df["segment"],
            textposition="inside",
            marker=dict(
                colors=[segment_color_map.get(segment, "#000000") for segment in filtered_df["segment"]]
            )
        )

        # Add a title to the chart and move the legend to the bottom
        fig1.update_layout(
            title="Segment Wise Sales",
            plot_bgcolor="#262730",
            paper_bgcolor="#262730",
            title_font_size=18,
            legend=dict(
                orientation="h",  # Horizontal orientation
                y=-0.2,  # Position below the chart
                x=0.5,  # Center the legend horizontally
                xanchor="center",
                yanchor="top"
            )
        )

        # Display the Segment-wise Sales Pie Chart
        st.plotly_chart(fig1, use_container_width=True, key="pie_segment_sales")


        # Create the Category/Sub-Category-wise Sales Pie Chart
        fig2 = px.pie(
            sales_df,  # Ensure filtered_df is used here
            values="sales",
            names=sales_label,  # Dynamically use sales_label for either "Category" or "Sub-Category"
            template="plotly_dark",
           hole=0.7 if sales_label == "Sub-Category" else 0
        )

        # Define color palettes for both Category and Sub-Category
        category_color_map = {
            "Furniture": "#006400",  # Dark Green
            "Office Supplies": "#228B22",  # Forest Green
            "Technology": "#21D375",  # Pale Green
            "Other": "#2E8B57"  # Sea Green
        }

        sub_category_color_map = {
        "Chairs": "#006400", 
        "Binders": "#228B22", 
        "Phones": "#007542",  
        "Accessories": "#2E8B57", 
        "Storage": "#21D375",
        "Paper": "#58BB43",
        "Furnishings": "#78D23D",
        "Appliances": "#9BE931",
        "Tables": "#C1FF1C",
        "Bookcases": "#006400",
        "Art": "#21D375",
        "Envelops": "#228B22",
        "Machines": "#007542",
        "Labels": "#2E8B57",
        "Supplies": "#21D375",
        "Copiers": "#58BB43",
        "fasteners": "#78D23D"
    }

        # Select the appropriate color map based on sales_label
        color_map = category_color_map if sales_label == "Category" else sub_category_color_map
        
        
        fig2.update_traces(
        text=sales_df[sales_label],
        textposition="inside",
         marker=dict(
        colors=[color_map.get(label, "#32CD32") for label in sales_df[sales_label]],  # Keep same base color
        line=dict(
            color=[color_map.get(label, "#32CD32") for label in sales_df[sales_label]],  # Same border color
        )
    )
)


        # Customize layout and title dynamically
        fig2.update_layout(
            title=f"{sales_label} Wise Sales",  # Dynamically set the title based on sales_label
            plot_bgcolor="#262730",
            paper_bgcolor="#262730",
            title_font_size=18,
            legend=dict(
                orientation="h",
                y=-0.2,
                x=0.5,
                xanchor="center",
                yanchor="top"
            )
        )

        # Display the Category/Sub-Category Sales Pie Chart
        st.plotly_chart(fig2, use_container_width=True, key="pie_category_sales")

    except Exception as e:
        st.error(f"Error generating Segment or Category/Sub-Category-wise Sales chart: {e}")

region = st.sidebar.selectbox(
    "Select Region For TreeMap",
    ("West", "South", "East", "Central"),
    key="region_selectbox"
)


with col2:
    # Region-wise Sales Subdivided Treemap
    try:
        # Ensure 'region' column exists in the DataFrame
        if 'region' not in filtered_df.columns:
            st.error("The 'region' column does not exist in the DataFrame.")
        else:
            # Drop NaN values in 'region' and ensure data type alignment
            filtered_df = filtered_df.dropna(subset=['region'])
            filtered_df['region'] = filtered_df['region'].astype(str)

            # Filter data based on the selected region
            region_df = filtered_df[filtered_df['region'] == region]

            # Check if the filtered data is empty
            if region_df.empty:
                st.warning(f"No data available for the region: {region}")
            else:
                # Create treemap chart with a green color palette, using 'Sub-Category' for legend
                fig4 = px.treemap(
                    region_df,
                    path=["Category", "Sub-Category"],  # Hierarchical data for treemap
                    values="sales",  # Size of each rectangle
                    color="Sub-Category",  # Use 'Sub-Category' for legend by name
                    color_discrete_sequence=[
                        "#98FB98",  # Pale Green
                        "#21D375",  # Lime Green
                        "#228B22",  # Forest Green
                        "#006400"   # Dark Green
                    ],
                    labels={"sales": "Total Sales", "Category": "Product Category", "Sub-Category": "Sub-Category"}
                )

                # Update layout and appearance
                fig4.update_layout(
                    title=f"Categorical Sales by {region}",
                    margin=dict(t=50, l=25, r=25, b=25),  # Margins for better layout
                    plot_bgcolor="#262730",
                    paper_bgcolor="#262730",
                    height=500,  # Adjusted height
                    showlegend=True  # Ensure the legend is displayed
                )

                # Display the treemap chart with the same container width as the map
                st.plotly_chart(fig4, use_container_width=True, key=f"treemap_{region}_sales")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    # List of all US state abbreviations (50 states)
    all_states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
        "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
        "VA", "WA", "WV", "WI", "WY"
    ]

    # Map full state names to their abbreviations
    state_abbr = {
        "California": "CA", "New York": "NY", "Texas": "TX", "Florida": "FL", "Illinois": "IL",
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "Colorado": "CO",
        "Connecticut": "CT", "Delaware": "DE", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
        "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
        "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "North Carolina": "NC",
        "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
        "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
        "Wisconsin": "WI", "Wyoming": "WY"
    }

    # Group data by state and sum the sales
    state_sales = filtered_df.groupby("state", as_index=False)["sales"].sum()

    # Replace full state names with abbreviations
    state_sales["state"] = state_sales["state"].replace(state_abbr)

    # Ensure all states are included in the DataFrame, even those with zero sales
    missing_states = [state for state in all_states if state not in state_sales["state"].values]

    # Create DataFrame for missing states with 0 sales
    missing_states_df = pd.DataFrame({"state": missing_states, "sales": [0] * len(missing_states)})

    # Combine existing data with the missing states data
    state_sales = pd.concat([state_sales, missing_states_df], ignore_index=True)

    # Remove any duplicate entries (if they exist)
    state_sales = state_sales.drop_duplicates(subset="state")

    # Create the choropleth map with Plotly Express
    fig_map = px.choropleth(
        state_sales,
        locations="state",
        locationmode="USA-states",
        color="sales",
        color_continuous_scale="greens",  # Color scale from low to high sales
        scope="usa",
        labels={"Sales": "Total Sales"},
        title="Sales by State"
    )

    # Update the layout with fixed size
    fig_map.update_layout(
    height=400,  # Set the height of the map
    width=900,   # Set the width of the map
    title_font_size=18,
    plot_bgcolor="#262730",  # Background color outside the map
    paper_bgcolor="#262730",  # Background color of the chart area
    geo=dict(
        bgcolor="#262730"  # Background color inside the map
    ),
    coloraxis_showscale=True
)

    # Display the map in Streamlit with the same container width as the treemap
    st.plotly_chart(fig_map, use_container_width=True)
    