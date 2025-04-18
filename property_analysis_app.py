
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
    # Read the Excel file
    df = pd.read_excel('Suburbtrends SA2 Excel March 2025.xlsx', sheet_name='House SA2', header=2)
    
    # Clean column names - replace newlines with underscores and strip whitespace
    df.columns = [c.strip().replace('
', '_') for c in df.columns]
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.title('Property Analysis')

# Add filters
min_price = st.sidebar.number_input('Minimum Sale Price', 
                                  min_value=float(df['Sale_Median_Now'].min()),
                                  max_value=float(df['Sale_Median_Now'].max()),
                                  value=float(df['Sale_Median_Now'].min()))

max_price = st.sidebar.number_input('Maximum Sale Price',
                                  min_value=float(df['Sale_Median_Now'].min()),
                                  max_value=float(df['Sale_Median_Now'].max()),
                                  value=float(df['Sale_Median_Now'].max()))

selected_states = st.sidebar.multiselect('Select States',
                                       options=sorted(df['State'].unique()),
                                       default=sorted(df['State'].unique())[0])

# Filter data
filtered_df = df[
    (df['Sale_Median_Now'] >= min_price) &
    (df['Sale_Median_Now'] <= max_price) &
    (df['State'].isin(selected_states))
]

# Main content
st.title('Property Market Analysis')

# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Map
    st.subheader('Property Locations')
    fig_map = px.scatter_mapbox(filtered_df,
                               lat='Lat',
                               lon='Long',
                               color='Sale_Median_Now',
                               size='Investor_Score_(Out_Of_100)',
                               hover_name='SA2',
                               hover_data=['State', 'Sale_Median_Now', 'Yield'],
                               color_continuous_scale='Viridis',
                               zoom=3,
                               title='Property Locations by Price and Investor Score')
    
    fig_map.update_layout(mapbox_style='carto-positron')
    fig_map.update_layout(margin={'r':0,'t':30,'l':0,'b':0})
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    # Price Distribution
    st.subheader('Price Distribution')
    fig_hist = px.histogram(filtered_df,
                           x='Sale_Median_Now',
                           nbins=30,
                           title='Distribution of Property Prices')
    st.plotly_chart(fig_hist, use_container_width=True)

# Bottom section
st.subheader('Property Details')
st.dataframe(filtered_df[[
    'SA2', 'State', 'Sale_Median_Now', 'Yield',
    'Buy_Affordability_(Years)', 'Investor_Score_(Out_Of_100)'
]].sort_values('Sale_Median_Now', ascending=False))

# Show summary statistics
st.subheader('Summary Statistics')
col3, col4, col5 = st.columns(3)

with col3:
    st.metric('Average Price', f"${filtered_df['Sale_Median_Now'].mean():,.0f}")
    
with col4:
    st.metric('Average Yield', f"{filtered_df['Yield'].mean():.2f}%")
    
with col5:
    st.metric('Average Investor Score', f"{filtered_df['Investor_Score_(Out_Of_100)'].mean():.1f}")
