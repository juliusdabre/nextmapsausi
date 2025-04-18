
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
    st.title('Property Market Analysis Dashboard')
    
    # Create a container for all filters
    with st.container():
        st.subheader('Filter Properties by Scores')
        
        # Get numeric columns for filtering
        score_columns = [col for col in df.columns if 'Score' in col or 'Index' in col]
        selected_features = {}
        
        # Create three columns for sliders
        cols = st.columns(3)
        
        # Add sliders for each feature in columns
        for idx, feature in enumerate(score_columns):
            col_idx = idx % 3
            with cols[col_idx]:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                selected_features[feature] = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    help=f'Filter {feature} between selected values'
                )

        # Add price change filters
        st.subheader('Filter by Price Changes')
        price_cols = st.columns(2)
        
        with price_cols[0]:
            price_2m = st.slider(
                '2M Price Change (%)',
                min_value=float(((df['Sale_Median_Now'] - df['Sale_Median_3m Ago']) / df['Sale_Median_3m Ago'] * 100).clip(-50, 50).min()),
                max_value=float(((df['Sale_Median_Now'] - df['Sale_Median_3m Ago']) / df['Sale_Median_3m Ago'] * 100).clip(-50, 50).max()),
                value=(-50.0, 50.0)
            )
        
        with price_cols[1]:
            price_12m = st.slider(
                '12M Price Change (%)',
                min_value=float(((df['Sale_Median_Now'] - df['Sale_Median_12m Ago']) / df['Sale_Median_12m Ago'] * 100).clip(-50, 50).min()),
                max_value=float(((df['Sale_Median_Now'] - df['Sale_Median_12m Ago']) / df['Sale_Median_12m Ago'] * 100).clip(-50, 50).max()),
                value=(-50.0, 50.0)
            )

    # Filter the dataframe based on selected ranges
    mask = pd.Series(True, index=df.index)
    
    for feature, (min_val, max_val) in selected_features.items():
        mask = mask & (df[feature].between(min_val, max_val))
    
    # Add price change filters to mask
    price_2m_calc = ((df['Sale_Median_Now'] - df['Sale_Median_3m Ago']) / df['Sale_Median_3m Ago'] * 100).clip(-50, 50)
    price_12m_calc = ((df['Sale_Median_Now'] - df['Sale_Median_12m Ago']) / df['Sale_Median_12m Ago'] * 100).clip(-50, 50)
    
    mask = mask & (price_2m_calc.between(price_2m[0], price_2m[1]))
    mask = mask & (price_12m_calc.between(price_12m[0], price_12m[1]))
    
    filtered_df = df[mask]
    
    # Show number of matching SA2s
    st.metric('Matching SA2 Regions', len(filtered_df))
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(['Map View', 'Analysis', 'Details'])
    
    with tab1:
        # Map visualization
        if all(col in filtered_df.columns for col in ['Lat', 'Long', 'Socio economics']):
            st.subheader('Socio-economic Map of Filtered SA2 Regions')
            fig_map = px.scatter_mapbox(
                filtered_df,
                lat='Lat',
                lon='Long',
                color='Socio economics',
                size=np.ones(len(filtered_df)) * 15,
                color_continuous_scale=['red', 'yellow', 'green'],
                mapbox_style='carto-positron',
                zoom=4,
                title='Filtered SA2 Regions',
                hover_name='SA2',
                hover_data={
                    'Lat': False,
                    'Long': False,
                    'Socio economics': True,
                    'Sale_Median_Now': True,
                    'Growth_Score': True
                }
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning('Missing required columns for map visualization')

    with tab2:
        # Analysis visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of socio-economic scores
            st.subheader('Socio-economic Score Distribution')
            fig_socio = px.histogram(
                filtered_df,
                x='Socio economics',
                nbins=30,
                title='Distribution of Socio-economic Scores'
            )
            st.plotly_chart(fig_socio, use_container_width=True)
            
            # Price changes heatmap
            st.subheader('Price Changes Distribution')
            price_changes = pd.DataFrame({
                'SA2': filtered_df['SA2'],
                '2M Price Change': price_2m_calc[mask],
                '12M Price Change': price_12m_calc[mask]
            }).melt(id_vars=['SA2'], var_name='Period', value_name='Price Change %')
            
            fig_heatmap = px.density_heatmap(
                price_changes,
                x='Period',
                y='Price Change %',
                title='Price Changes Distribution'
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        with col2:
            # Correlation matrix
            st.subheader('Feature Correlations')
            correlation_features = score_columns[:8]  # Limit to 8 features
            if correlation_features:
                correlation_matrix = filtered_df[correlation_features].corr()
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

    with tab3:
        # Detailed data view
        st.subheader('Filtered SA2 Details')
        
        # Add a search box for SA2
        search_term = st.text_input('Search SA2 by name')
        if search_term:
            display_df = filtered_df[filtered_df['SA2'].str.contains(search_term, case=False, na=False)]
        else:
            display_df = filtered_df
            
        # Show the data
        st.dataframe(
            display_df.style.highlight_max(axis=0, subset=score_columns),
            height=400
        )
        
        # Download button for filtered data
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="filtered_sa2_data.csv",
            mime="text/csv"
        )

else:
    st.error('Failed to load data. Please check the data file and try again.')
