
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(layout='wide', page_title='Property Analysis Dashboard')

@st.cache_data
def load_data():
    try:
        excel_file = 'Suburbtrends SA2 Excel March 2025.xlsx'
        if not Path(excel_file).exists():
            st.error(f"Could not find {excel_file}. Please ensure it's in the same directory as the app.")
            return None
        df = pd.read_excel(excel_file, sheet_name='House SA2', header=2)
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Sidebar: Feature and SA2 selection
    st.sidebar.title('Property Analysis Filters')
    # Get available numeric columns for feature selection
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    feature_options = [col for col in numeric_columns if 'Score' in col or 'Growth' in col or 'Index' in col]
    feature = st.sidebar.selectbox('Select Feature', feature_options, index=0 if feature_options else None)
    sa2_options = df['SA2'].dropna().unique()
    selected_sa2 = st.sidebar.selectbox('Select SA2', sa2_options)

    st.title('Property Market Analysis Dashboard')
    st.write('You can now filter by SA2 and see its details and location on the map.')

    # Show details for selected SA2
    sa2_row = df[df['SA2'] == selected_sa2]
    st.subheader(f'Details for SA2: {selected_sa2}')
    if not sa2_row.empty:
        st.table(sa2_row.T)
    else:
        st.warning('No data found for selected SA2.')

    # Map: Socio-economic score for all, highlight selected SA2
    st.subheader('Socio-economic Map (SA2)')
    if all(col in df.columns for col in ['Lat', 'Long', 'Socio economics']):
        fig_map = px.scatter_mapbox(
            df,
            lat='Lat',
            lon='Long',
            color='Socio economics',
            size=np.where(df['SA2'] == selected_sa2, 20, 10),
            color_continuous_scale=['red', 'yellow', 'green'],
            mapbox_style='carto-positron',
            zoom=4,
            title='Socio-economic Score by SA2',
            hover_name='SA2',
            hover_data={'Lat': True, 'Long': True, 'Socio economics': True}
        )
        # Highlight selected SA2 with a marker
        if not sa2_row.empty:
            fig_map.add_trace(
                go.Scattermapbox(
                    lat=sa2_row['Lat'],
                    lon=sa2_row['Long'],
                    mode='markers+text',
                    marker=dict(size=30, color='blue', opacity=0.7),
                    text=[selected_sa2],
                    textposition='top right',
                    name='Selected SA2'
                )
            )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning('Missing latitude/longitude or socio-economic data for map visualization.')

    # Feature Distribution
    st.subheader(f'{feature} Distribution Analysis')
    fig_hist = px.histogram(
        df,
        x=feature,
        nbins=30,
        title=f'Distribution of {feature}'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Price Change Analysis
    st.subheader('Price Change Analysis')
    price_columns = [col for col in df.columns if 'Sale_Median' in col]
    if len(price_columns) >= 3:
        price_changes = pd.DataFrame({
            'SA2': df['SA2'],
            '2M Price Change': ((df['Sale_Median_Now'] - df['Sale_Median_3m Ago']) / df['Sale_Median_3m Ago'] * 100).clip(-50, 50),
            '12M Price Change': ((df['Sale_Median_Now'] - df['Sale_Median_12m Ago']) / df['Sale_Median_12m Ago'] * 100).clip(-50, 50)
        }).melt(id_vars=['SA2'], var_name='Period', value_name='Price Change %')
        fig_heatmap = px.density_heatmap(
            price_changes,
            x='Period',
            y='Price Change %',
            title='Price Changes Distribution'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.warning('Insufficient price data for change analysis.')

    # Socio-economic Analysis
    if 'Socio economics' in df.columns:
        st.subheader('Socio-economic Analysis')
        fig_scatter = px.scatter(
            df,
            x='Socio economics',
            y=feature,
            title=f'Socio-economic Score vs {feature}',
            trendline='ols'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning('Socio-economic data not available.')

    # Summary Statistics
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
    correlation_features = [col for col in feature_options if col in df.columns][:8]
    if correlation_features:
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
        fig_corr.update_layout(title='Feature Correlation Matrix', height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning('Insufficient features for correlation analysis.')
else:
    st.error('Failed to load data. Please check the data file and try again.')
