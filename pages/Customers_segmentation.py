import psycopg2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st

# Initialize PostgreSQL connection
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

# Query execution function
def run_query(query, conn):
    return pd.read_sql(query, conn)

# Load CSS for custom styling
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("./assets/style1.css")

# Customer Segmentation Page
st.title("Customer Segmentation Using Clustering")

# Step 1: Fetch Data from PostgreSQL
query = """
SELECT 
    SUM(fs.sales) AS total_spending, 
    COUNT(fs.order_id) AS frequency,
    c.customer_id
FROM fact_sales fs
JOIN dim_customer c ON fs.customer_id = c.customer_id
GROUP BY c.customer_id;
"""

with init_connection() as conn:
    df = run_query(query, conn)

# Display customer data for reference
st.subheader("Customer Data")
st.write(df.head())

# Step 2: Data Preprocessing
features = df[["total_spending", "frequency"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Sidebar: Elbow Curve Max Clusters
st.sidebar.header("Elbow Curve Settings")
max_clusters = st.sidebar.slider("Select Maximum Number of Clusters", 2, 10, 6)


# Calculate inertia for different cluster counts
inertia = []
for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve using Matplotlib
plt.figure(figsize=(9, 4))
plt.plot(range(1, max_clusters + 1), inertia, marker='o', linestyle='--')
plt.title("Elbow Curve for Optimal Clusters", fontsize=14)
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Inertia", fontsize=12)
plt.grid(visible=True, linestyle='--', linewidth=0.7)  # Add gridlines for clarity
st.pyplot(plt)  # Display the plot in Streamlit

# Sidebar: Number of Clusters
st.sidebar.header("Clustering Settings")
optimal_clusters = st.sidebar.number_input(
    "Select Number of Clusters",
    min_value=2,
    max_value=max_clusters,
    value=3,
    step=1,
)

# Step 5: Perform Clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled_features)

# Step 6: Extract Cluster Centroids
centroids = kmeans.cluster_centers_

# Create a DataFrame for centroids
centroid_df = pd.DataFrame(centroids, columns=["total_spending", "frequency"])

# Sort centroids by total_spending and frequency (descending)
centroid_df = centroid_df.sort_values(["total_spending", "frequency"], ascending=[False, False])

# Generate dynamic labels based on the number of clusters
dynamic_labels = [f"Cluster {i+1}" for i in range(optimal_clusters)]

# Assign labels to centroids
centroid_df["cluster_label"] = dynamic_labels

# Map cluster numbers to labels
dynamic_cluster_labels = dict(zip(centroid_df.index, centroid_df["cluster_label"]))

# Apply cluster labels to main DataFrame
df["cluster_label"] = df["cluster"].map(dynamic_cluster_labels)

# Sort clusters by total_spending (descending) for display
cluster_options = (
    df.groupby("cluster_label")
    .agg({"total_spending": "mean"})
    .sort_values("total_spending", ascending=False)
    .index.tolist()
)

# Step 7: Visualizations

# Customer Clusters Scatter Plot
st.subheader("Customer Clusters")

# Filter the data based on the selected clusters from the sidebar
selected_clusters = st.multiselect(
    "Select Clusters to Display",
    options=cluster_options,
    default=cluster_options  # Show all clusters by default
)

# Filter data for the selected clusters
filtered_df = df[df["cluster_label"].isin(selected_clusters)]

# Scatter plot using Plotly Express
scatter_fig = px.scatter(
    filtered_df,
    x="frequency",
    y="total_spending",
    color="cluster_label",
    title="Customer Segmentation (Spending vs Frequency)",
    labels={"cluster_label": "Cluster", "frequency": "Purchase Frequency", "total_spending": "Total Spending"},
    hover_data=["customer_id"],
)

# Update the plot for better visuals
scatter_fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
scatter_fig.update_layout(
    legend_title="Cluster",
    xaxis_title="Purchase Frequency",
    yaxis_title="Total Spending",
)

# Display the scatter plot
st.plotly_chart(scatter_fig, use_container_width=True)

# Cluster Distribution Bar Chart
st.subheader("Cluster Distribution")

# Count the number of customers per cluster
cluster_counts = df["cluster_label"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]

# Plot the bar chart
bar_fig = px.bar(
    cluster_counts,
    x="Cluster",
    y="Count",
    color="Cluster",
    title="Number of Customers per Cluster",
    text="Count",
)

# Update bar chart appearance
bar_fig.update_traces(
    texttemplate='%{text}',  # Show count text
    textposition='outside',  # Place the text outside of bars
    marker=dict(line=dict(width=2, color='DarkSlateGrey'))  # Add border to bars
)

# Display the bar chart
st.plotly_chart(bar_fig, use_container_width=True)

# Additional Section: Insights and Recommendations
st.sidebar.header("Insights")

# Filter for displaying insights
visualize_options = st.sidebar.multiselect(
    "Select Visualization to Display",
    ["Insights and Recommendations"],
    default=["Insights and Recommendations"],
)

# Display Insights and Recommendations if selected
if "Insights and Recommendations" in visualize_options:
    st.subheader("Insights and Recommendations")

    # Filter by selected clusters
    selected_insight_clusters = st.multiselect(
        "Select Clusters for Insights",
        cluster_options,
        default=cluster_options
    )

    # Filter data for selected clusters
    insight_df = df[df["cluster_label"].isin(selected_insight_clusters)]

    st.write("### Key Insights:")
    for cluster in selected_insight_clusters:
        cluster_data = insight_df[insight_df["cluster_label"] == cluster]
        spending_avg = cluster_data["total_spending"].mean()
        freq_avg = cluster_data["frequency"].mean()
        st.markdown(
            f"""
            - **{cluster}:**  
              Average Spending: **${spending_avg:,.2f}**  
              Average Purchase Frequency: **{freq_avg:.2f} times**
            """
        )

    # Recommendations based on average spending and frequency
    st.write("### Recommendations:")

    # Define thresholds for high/low spending and frequency
    high_spending_threshold = df["total_spending"].quantile(0.75)
    low_spending_threshold = df["total_spending"].quantile(0.25)
    high_frequency_threshold = df["frequency"].quantile(0.75)
    low_frequency_threshold = df["frequency"].quantile(0.25)

    # Generate recommendations based on the selected clusters
    for cluster in selected_insight_clusters:
        cluster_data = df[df["cluster_label"] == cluster]
        avg_spending = cluster_data["total_spending"].mean()
        avg_frequency = cluster_data["frequency"].mean()

        if avg_spending >= high_spending_threshold and avg_frequency >= high_frequency_threshold:
            st.markdown(
                f"- **{cluster}:** High-value and frequent customers. Consider implementing loyalty programs, exclusive offers, and personalized services to retain these customers."
            )
        elif avg_spending <= low_spending_threshold and avg_frequency <= low_frequency_threshold:
            st.markdown(
                f"- **{cluster}:** Low-value and infrequent customers. Engage them with targeted promotions, discounts, and personalized communications to increase both spending and frequency."
            )
        else:
            st.markdown(
                f"- **{cluster}:** Moderate value and frequency. Review customer behavior and consider tailored strategies, such as upselling or cross-selling, to improve engagement and boost value."
            )
