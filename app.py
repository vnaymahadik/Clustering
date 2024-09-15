import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
import plotly.express as px
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(page_title="Consumer Complaints Clustering Analysis", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("1.csv")
    # Clean data
    df['Date received'] = pd.to_datetime(df['Date received'], format='mixed')
    df['Date sent to company'] = pd.to_datetime(df['Date sent to company'], format='mixed')
    df['Timely response?'] = df['Timely response?'].map({'Yes': 1, 'No': 0})
    df['Consumer disputed?'] = df['Consumer disputed?'].map({'Yes': 1, 'No': 0})
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
    
    return df
df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "EDA", "Clustering Analysis", "Recommendations"])

# Introduction
if page == "Introduction":
    st.title("Consumer Complaints Clustering Analysis")
    st.write("""
    Welcome to the Consumer Complaints Clustering Analysis app! This application is designed to help analyze and cluster consumer complaints data using various machine learning techniques.

    Here's what you can do with this app:
    1. Explore the data through Exploratory Data Analysis (EDA)
    2. Perform clustering analysis using different algorithms
    3. Visualize the results and gain insights
    4. Get recommendations based on the analysis

    Navigate through the different sections using the sidebar on the left. Start with the EDA to understand the data, then move on to the Clustering Analysis to dive deeper into patterns and groupings within the complaints.

    This tool is particularly useful for:
    - Consumer protection agencies
    - Financial institutions
    - Regulatory bodies
    - Researchers in consumer behavior

    Enjoy exploring the data and discovering meaningful insights!
    """)

# EDA
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    st.subheader("Data Overview")
    st.write(df.head())
    st.write(f"Shape of the dataset: {df.shape}")

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    st.subheader("Complaint Distribution by Product")
    product_counts = df['Product'].value_counts()
    fig = px.bar(x=product_counts.index, y=product_counts.values, 
                 labels={'x': 'Product', 'y': 'Count'},
                 title="Complaint Distribution by Product")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    st.write("""
    Insights:
    - The bar chart shows that mortgage-related complaints are the most common, followed by credit card and bank account issues.
    - This suggests that financial products with long-term commitments or daily usage tend to generate more complaints.
    - Consumer loan and student loan complaints are less frequent, possibly due to their more specific nature.
    """)

    st.subheader("Complaint Distribution by State")
    state_counts = df['State'].value_counts().head(10)
    fig = px.bar(x=state_counts.index, y=state_counts.values, 
                 labels={'x': 'State', 'y': 'Count'},
                 title="Top 10 States by Number of Complaints",
                 color=state_counts.values,
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    st.write("""
    Insights:
    - The chart shows that complaints are not evenly distributed across states.
    - States with larger populations or major financial centers tend to have more complaints.
    - This could be due to higher population density, more financial activity, or potentially different regulatory environments.
    - Consider investigating why certain states have disproportionately high complaint rates relative to their population.
    """)

    st.subheader("Timely Response Distribution")
    timely_response = df['Timely response?'].value_counts()
    fig = px.pie(values=timely_response.values, names=timely_response.index, 
                 title="Timely Response Distribution",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig)
    st.write("""
    Insights:
    - The pie chart shows that the majority of complaints receive a timely response.
    - This suggests that companies are generally responsive to consumer complaints.
    - However, there's still room for improvement in addressing all complaints in a timely manner.
    - Investigate factors contributing to untimely responses to further improve customer service.
    """)

    st.subheader("Consumer Disputed Distribution")
    consumer_disputed = df['Consumer disputed?'].value_counts()
    fig = px.pie(values=consumer_disputed.values, names=consumer_disputed.index, 
                 title="Consumer Disputed Distribution",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig)
    st.write("""
    Insights:
    - The pie chart indicates that a significant portion of complaints are not disputed by consumers.
    - This could suggest that many complaints are resolved satisfactorily.
    - For disputed cases, it's important to understand the reasons and improve resolution processes.
    - Consider analyzing the characteristics of disputed vs. non-disputed complaints for further insights.
    """)

    st.subheader("Complaints Over Time")
    df['Month'] = df['Date received'].dt.to_period('M')
    monthly_complaints = df.groupby('Month').size().reset_index(name='Count')
    monthly_complaints['Month'] = monthly_complaints['Month'].astype(str)
    fig = px.line(monthly_complaints, x='Month', y='Count', 
                  title='Number of Complaints Over Time',
                  labels={'Count': 'Number of Complaints', 'Month': 'Month'})
    st.plotly_chart(fig)
    st.write("""
    Insights:
    - The line chart shows the trend of complaints over time.
    - Look for any seasonal patterns or significant spikes in complaints.
    - Consider external factors (e.g., economic conditions, policy changes) that might influence complaint volumes.
    - Use this information to anticipate periods of high complaint volume and allocate resources accordingly.
    """)

    st.subheader("Distribution of Complaint Submission Methods")
    submission_method = df['Submitted via'].value_counts()
    fig = px.pie(values=submission_method.values, names=submission_method.index, 
                 title="Distribution of Complaint Submission Methods",
                 color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig)
    st.write("""
    Insights:
    - This chart shows how consumers are submitting their complaints.
    - Understanding preferred submission methods can help in optimizing the complaint intake process.
    - Consider improving the user experience for the most common submission methods.
    - For less used methods, investigate if there are barriers to access or usability issues.
    """)

    st.subheader("Correlation Between Numeric Variables")
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    fig = px.imshow(correlation_matrix, 
                    labels=dict(color="Correlation"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap of Numeric Variables")
    st.plotly_chart(fig)
    st.write("""
    Insights:
    - The heatmap shows correlations between numeric variables in the dataset.
    - Strong positive correlations appear in red, while strong negative correlations appear in blue.
    - Look for unexpected correlations that might provide insights into relationships between different aspects of complaints.
    - This can help in identifying potential factors that might be related to complaint outcomes or consumer satisfaction.
    """)

    st.subheader("Product vs. Consumer Disputed")
    product_disputed = df.groupby('Product')['Consumer disputed?'].mean().sort_values(ascending=False)
    fig = px.bar(x=product_disputed.index, y=product_disputed.values, 
                 labels={'x': 'Product', 'y': 'Proportion of Disputed Complaints'},
                 title="Proportion of Disputed Complaints by Product",
                 color=product_disputed.values,
                 color_continuous_scale='Reds')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    st.write("""
    Insights:
    - This chart shows which products have the highest proportion of disputed complaints.
    - Products with higher dispute rates may require more attention in terms of complaint resolution processes.
    - Consider investigating why certain products lead to more disputes and implement targeted improvements.
    - This information can be valuable for prioritizing areas for customer service enhancement.
    """)

# Clustering Analysis
elif page == "Clustering Analysis":
    st.title("Clustering Analysis")

    @st.cache_data
    def prepare_data(df, selected_features):
        X = df[selected_features]
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        return X_scaled

    @st.cache_data
    def perform_pca(X_scaled, n_components):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca, pca.explained_variance_ratio_

    # Feature selection
    st.subheader("Feature Selection")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != 'Complaint ID']  # Exclude Complaint ID
    selected_features = st.multiselect("Select features for clustering:", numeric_columns, default=numeric_columns[:5])

    if len(selected_features) < 2:
        st.warning("Please select at least two features for clustering.")
    else:
        # Prepare data for clustering
        X_scaled = prepare_data(df, selected_features)

        # PCA
        st.subheader("PCA Insights")
        n_components = min(len(selected_features), 10)  # Limit to 10 components for performance
        X_pca, explained_variance_ratio = perform_pca(X_scaled, n_components)

        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        fig = px.line(x=range(1, len(explained_variance_ratio) + 1), y=cumulative_variance_ratio,
                      labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance Ratio'},
                      title='PCA Cumulative Explained Variance Ratio')
        st.plotly_chart(fig)

        st.write("""
        Insights from PCA:
        - The graph shows how much variance is explained by each principal component.
        - A steep curve indicates that a few components capture most of the variance.
        - Use this to determine the optimal number of components for dimensionality reduction.
        """)

        if n_components == 2:
            st.write("Using 2 PCA components as there are only 2 features or components available.")
            n_components_for_clustering = 2
        else:
            n_components_for_clustering = st.slider("Select number of PCA components:", 
                                                    min_value=2, 
                                                    max_value=n_components, 
                                                    value=min(3, n_components))
        
        X_pca = X_pca[:, :n_components_for_clustering]

        # Clustering algorithm selection
        st.subheader("Clustering Algorithm")
        algorithm = st.selectbox("Select clustering algorithm:", 
                                 ["K-Means", "Hierarchical Agglomerative", "DBSCAN", "Gaussian Mixture"])

        @st.cache_data
        def perform_clustering(_X, algorithm, n_clusters=None, eps=None, min_samples=None):
            if algorithm == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif algorithm == "Hierarchical Agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
            else:  # Gaussian Mixture
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            
            labels = model.fit_predict(_X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(_X, labels)
            else:
                score = 0
            return labels, score

        # Use a subset of data for parameter tuning
        n_samples = min(10000, X_pca.shape[0])
        X_subset = X_pca[:n_samples]

        if algorithm == "DBSCAN":
            eps_values = [0.1, 0.5, 1.0, 1.5, 2.0]
            min_samples_values = [3, 5, 10, 15, 20]
            results = []
            for eps in eps_values:
                for min_samples in min_samples_values:
                    labels, score = perform_clustering(X_subset, algorithm, eps=eps, min_samples=min_samples)
                    n_clusters = len(np.unique(labels[labels >= 0]))
                    results.append((eps, min_samples, n_clusters, score))
            
            results_df = pd.DataFrame(results, columns=['eps', 'min_samples', 'n_clusters', 'silhouette_score'])
            fig = px.scatter_3d(results_df, x='eps', y='min_samples', z='silhouette_score', 
                                color='n_clusters', size='silhouette_score',
                                title='DBSCAN - Parameter Tuning')
            st.plotly_chart(fig)
            
            best_result = results_df.loc[results_df['silhouette_score'].idxmax()]
            st.write(f"Recommended parameters: eps={best_result['eps']}, min_samples={best_result['min_samples']}")
            st.write(f"Number of clusters: {best_result['n_clusters']}")
            st.write(f"Silhouette score: {best_result['silhouette_score']:.3f}")

            st.write("""
            DBSCAN Insights:
            - DBSCAN doesn't require specifying the number of clusters beforehand.
            - It can identify noise points that don't belong to any cluster.
            - The 'eps' parameter defines the neighborhood distance, while 'min_samples' sets the minimum cluster size.
            - Higher silhouette scores indicate better-defined clusters.
            - Be cautious of parameter combinations that result in too many or too few clusters.
            """)

            eps = st.slider("Select eps value:", min_value=0.1, max_value=2.0, value=best_result['eps'], step=0.1)
            min_samples = st.slider("Select min_samples value:", min_value=2, max_value=20, value=int(best_result['min_samples']))
            final_labels, final_score = perform_clustering(X_pca, algorithm, eps=eps, min_samples=min_samples)

        else:
            results = []
            for n_clusters in range(2, 11):
                labels, score = perform_clustering(X_subset, algorithm, n_clusters=n_clusters)
                results.append((n_clusters, score))
            
            results_df = pd.DataFrame(results, columns=['n_clusters', 'silhouette_score'])
            fig = px.line(results_df, x='n_clusters', y='silhouette_score', 
                          title=f'{algorithm} - Silhouette Score vs Number of Clusters')
            st.plotly_chart(fig)
            
            best_result = results_df.loc[results_df['silhouette_score'].idxmax()]
            st.write(f"Recommended number of clusters: {best_result['n_clusters']}")
            st.write(f"Best silhouette score: {best_result['silhouette_score']:.3f}")

            if algorithm == "K-Means":
                st.write("""
                K-Means Insights:
                - K-Means works best for spherical clusters of similar size.
                - The elbow in the silhouette score graph suggests the optimal number of clusters.
                - Higher silhouette scores indicate better-defined clusters.
                - Consider the trade-off between the number of clusters and the silhouette score.
                """)
            elif algorithm == "Hierarchical Agglomerative":
                st.write("""
                Hierarchical Agglomerative Clustering Insights:
                - This method creates a hierarchy of clusters, allowing for different levels of granularity.
                - It doesn't assume a particular shape for the clusters.
                - The dendrogram (not shown here) can provide additional insights into the clustering structure.
                - Consider the balance between the number of clusters and the silhouette score.
                """)
            else:  # Gaussian Mixture
                st.write("""
                Gaussian Mixture Model Insights:
                - GMM assumes that the data is generated from a mixture of Gaussian distributions.
                - It's more flexible than K-Means and can capture clusters of different shapes and sizes.
                - The optimal number of components balances model complexity and fit.
                - Higher silhouette scores generally indicate better-defined clusters.
                """)

            n_clusters = st.slider("Select number of clusters:", min_value=2, max_value=10, value=int(best_result['n_clusters']))
            final_labels, final_score = perform_clustering(X_pca, algorithm, n_clusters=n_clusters)

        st.subheader("Final Clustering Results")
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=final_labels, 
                         labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
                         title=f"Clustering Results using {algorithm}")
        st.plotly_chart(fig)

        st.write(f"Final Silhouette Score: {final_score:.3f}")

        # Display cluster statistics
        st.subheader("Cluster Statistics")
        cluster_stats = pd.DataFrame({'Cluster': final_labels, 'Product': df['Product']})
        cluster_product_counts = cluster_stats.groupby('Cluster')['Product'].value_counts().unstack(fill_value=0)
        
        fig = px.imshow(cluster_product_counts, 
                        labels=dict(x="Product", y="Cluster", color="Count"),
                        x=cluster_product_counts.columns,
                        y=cluster_product_counts.index,
                        color_continuous_scale='Viridis',
                        title="Heatmap of Products per Cluster")
        st.plotly_chart(fig)

        st.write("""
        Cluster Statistics Insights:
        - The heatmap shows the distribution of products across different clusters.
        - Darker colors indicate a higher concentration of a particular product in a cluster.
        - Look for patterns: Are certain products dominating specific clusters?
        - This can reveal groupings of complaints that might not be apparent from individual features.
        - Consider investigating clusters with high concentrations of specific products to understand common issues.
        """)
# Recommendations
else:
    st.title("Recommendations")
    st.write("""
    Based on the clustering analysis of consumer complaints, here are some recommendations:

    1. **Focus on High-Volume Products**: Pay special attention to products with the highest number of complaints, such as Mortgages and Credit Cards. Develop targeted strategies to address common issues in these areas.

    2. **Improve Response Time**: Given the importance of timely responses, implement processes to ensure quick and efficient handling of complaints across all product categories.

    3. **Address Regional Variations**: Analyze clustering results by state to identify regional patterns in complaints. Tailor your approach and resources accordingly for different geographical areas.

    4. **Enhance Dispute Resolution**: For clusters with high rates of consumer disputes, review and improve your dispute resolution processes to increase customer satisfaction and reduce escalations.

    5. **Product-Specific Strategies**: Use the cluster statistics to develop product-specific strategies. For example, if a particular cluster shows a high concentration of mortgage-related complaints, focus on improving mortgage servicing and communication.

    6. **Predictive Modeling**: Utilize the clustering results to build predictive models that can anticipate potential issues before they escalate into formal complaints.

    7. **Customer Education**: For clusters that suggest a lack of customer understanding (e.g., frequent complaints about terms and conditions), develop targeted educational materials and improve communication.

    8. **Continuous Monitoring**: Regularly update the clustering analysis to track changes in complaint patterns over time and adjust strategies accordingly.

    9. **Cross-Functional Collaboration**: Share insights from the clustering analysis with different departments (e.g., product development, customer service, legal) to drive company-wide improvements.

    10. **Regulatory Compliance**: Use the clustering results to ensure compliance with regulatory requirements and identify areas that may need additional attention to meet industry standards.

    By implementing these recommendations, you can work towards reducing the number of complaints, improving customer satisfaction, and enhancing your overall service quality.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ by Your Name")