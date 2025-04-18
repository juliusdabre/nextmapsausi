
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(layout='wide', page_title='Property Analysis Dashboard')

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('Suburbtrends SA2 Excel March 2025.xlsx', sheet_name='House SA2', header=2)
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

# Sidebar features
st.sidebar.title('Property Analysis Features')
feature = st.sidebar.selectbox(
    'Select Feature',
    ['Sales Turnover Score (SA2)', 'Sales Turnover Score (SA3)', 
     'Inventory Score (SA3)', 'Growth Gap Index (SA3)', 
     '1 Year Growth (SA3)', '10 Year Growth (SA3)', 
     'Remoteness (SA2)', 'Fully Owned Score (SA2)', 
     'Yield Score (SA2)', 'Buy Affordability Score (SA2)', 
     'Rent Affordability Score (SA2)', 'Rental Turnover Score (SA2)', 
     'Socio economics', 'Investor Score (Out Of 100)']
)

# Main content
st.title('Property Market Analysis Dashboard')

# Create two columns for the layout
col1, col2 = st.columns(2)

with col1:
    # Feature Score Map
    st.subheader(f'{feature} Distribution')
    
    fig_map = px.scatter_mapbox(
        df,
        lat='Lat',
        lon='Long',
        color=feature,
        size=np.ones(len(df)) * 10,  # Uniform size
        color_continuous_scale=['red', 'yellow', 'green'],
        range_color=[10, 100],
        mapbox_style='carto-positron',
        zoom=4,
        title=f'{feature} Geographic Distribution'
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Trend Analysis
    st.subheader(f'{feature} Distribution Analysis')
    fig_hist = px.histogram(
        df,
        x=feature,
        nbins=30,
        title=f'Distribution of {feature}'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Price Change Heatmap
    st.subheader('Price Change Analysis')
    
    # Calculate price changes
    price_changes = pd.DataFrame({
        'SA2': df['SA2'],
        '2M Price Change': ((df['Sale_Median_Now'] - df['Sale_Median_3m Ago']) / df['Sale_Median_3m Ago'] * 100),
        '12M Price Change': ((df['Sale_Median_Now'] - df['Sale_Median_12m Ago']) / df['Sale_Median_12m Ago'] * 100)
    }).melt(id_vars=['SA2'], var_name='Period', value_name='Price Change %')
    
    fig_heatmap = px.density_heatmap(
        price_changes,
        x='Period',
        y='Price Change %',
        title='Price Changes Distribution'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Socio-economic Analysis
    st.subheader('Socio-economic Analysis')
    fig_scatter = px.scatter(
        df,
        x='Socio economics',
        y=feature,
        title=f'Socio-economic Score vs {feature}',
        trendline='ols'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Additional Statistics
st.subheader('Summary Statistics')
col3, col4, col5 = st.columns(3)

with col3:
    st.metric('Average Score', f"{df[feature].mean():.1f}")
with col4:
    st.metric('Median Score', f"{df[feature].median():.1f}")
with col5:
    st.metric('Score Range', f"{df[feature].min():.1f} - {df[feature].max():.1f}")

# Correlation Analysis
st.subheader('Feature Correlations')
correlation_features = [
    'Sales Turnover Score (SA2)', 'Sales Turnover Score (SA3)',
    'Inventory Score (SA3)', 'Growth Gap Index (SA3)',
    '1 Year Growth (SA3)', '10 Year Growth (SA3)',
    'Socio economics', 'Investor Score (Out Of 100)'
]
correlation_matrix = df[correlation_features].corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_matrix,
    x=correlation_features,
    y=correlation_features,
    text=correlation_matrix.round(2),
    texttemplate='%{text}',
    textfont={'size': 10},
    colorscale='RdBu'
))
fig_corr.update_layout(
    title='Feature Correlation Matrix',
    height=500
)
st.plotly_chart(fig_corr, use_container_width=True)
