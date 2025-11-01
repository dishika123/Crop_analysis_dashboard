import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Project Samarth",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
        .main {
            padding: 1.5rem 2rem;
            max-width: 100%;
        }
        
        /* Button Styling */
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: white;
            color: #333;
            border: 1px solid #ddd;
            font-weight: 500;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #f8f9fa;
            border-color: #1E88E5;
            color: #1E88E5;
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Spacing for Select Boxes */
        .stSelectbox, .stMultiSelect {
            margin: 1.5rem 0;
            padding: 0.5rem 0;
        }
        
        /* Container backgrounds */
        .reportview-container {
            background: #f0f2f6;
            padding: 1rem;
        }
        .sidebar .sidebar-content {
            background: #f8f9fa;
            padding: 1.5rem 1rem;
        }
        .sidebar .sidebar-content > div {
            margin-bottom: 2rem;
        }
        
        /* Header Styling */
        h1 {
            color: #1E88E5;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            padding: 0.5rem 0;
        }
        h2 {
            color: #333;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 500;
        }
        h3 {
            color: #666;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stPlotlyChart {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 1rem;
            margin-bottom: 2rem;
            transition: transform 0.3s ease;
        }
        .stPlotlyChart:hover {
            transform: translateY(-5px);
        }
        .section-header {
            background-color: #1E88E5;
            padding: 1rem;
            border-radius: 5px;
            color: white;
            margin: 1rem 0;
        }
        /* Metric Card Styling */
        .metric-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
            margin: 1rem 0;
            border: 1px solid #eee;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border-color: #1E88E5;
        }
        .metric-card h4 {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        .metric-card h2 {
            color: #333;
            font-size: 1.8rem;
            margin: 0;
            font-weight: 600;
        }
        /* Custom styling for tabs */
        /* Tab Styling */
        .stTabs {
            margin: 2rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
            padding: 0.5rem;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #ffffff;
            border-radius: 8px;
            padding: 0px 24px;
            transition: all 0.3s ease;
            border: 1px solid #eee;
            margin: 0.25rem 0;
            font-weight: 500;
            color: #333;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #f8f9fa;
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-color: #1E88E5;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #1E88E5 !important;
            border-radius: 8px;
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(30,136,229,0.2);
            border: 2px solid #1E88E5 !important;
            font-weight: 600;
        }
        /* Add space between tab panels */
        .stTabs [data-baseweb="tab-panel"] {
            padding: 1.5rem 0;
        }
        /* Animation for tab content */
        .element-container {
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            0% {
                opacity: 0;
                transform: translateY(10px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
        /* Hover effects for sidebar elements */
        .stSelectbox:hover, .stMultiSelect:hover {
            transform: translateY(-2px);
            transition: transform 0.3s ease;
        }
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb {
            background: #1E88E5;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #1976D2;
        }
        
        /* Conclusion Card Styling */
        .conclusion-card {
            background-color: #f8f9fa;
            padding: 1rem 1.5rem;
            border-left: 4px solid #1E88E5;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0 2rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .conclusion-card h4 {
            color: #1E88E5;
            margin-bottom: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
        }
        .conclusion-card ul {
            margin: 0;
            padding-left: 1.2rem;
            color: #555;
        }
        .conclusion-card li {
            margin: 0.3rem 0;
            line-height: 1.4;
        }
    </style>
""", unsafe_allow_html=True)

# Load the datasets
@st.cache_data
def load_data():
    merge_df = pd.read_csv('merge.csv')
    season_pattern_df = pd.read_csv('season_pattern.csv')
    # Convert string representations of lists to actual lists
    season_pattern_df['Sowing Period'] = season_pattern_df['Sowing Period'].apply(eval)
    season_pattern_df['Harvesting Period'] = season_pattern_df['Harvesting Period'].apply(eval)
    return merge_df, season_pattern_df


merge_df, season_pattern_df = load_data()

# Header with custom styling
st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: #1E88E5; margin-bottom: 0.5rem;'>🌾 Project Samarth</h1>
        <p style='font-size: 1.2rem; color: #666;'>Data-driven insights into India's Agricultural–Climate Nexus</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with custom styling
st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 1rem;'>
        <h2 style='color: #1E88E5;'>Dashboard Controls</h2>
    </div>
""", unsafe_allow_html=True)

# Global Filters with better styling
st.sidebar.markdown("""
    <div style='padding: 1rem; background-color: white; border-radius: 5px; margin-bottom: 1rem;'>
        <h3 style='color: #666; margin-bottom: 1rem;'>Global Filters</h3>
    </div>
""", unsafe_allow_html=True)

# State filter
states = sorted(merge_df['State'].unique())
state_options = ['All'] + states
selected_states = st.sidebar.multiselect("Select States", state_options, default=['All'])
selected_states = states if 'All' in selected_states else selected_states

# Crop filter
crops = sorted(merge_df['Crop'].unique())
crop_options = ['All'] + crops
selected_crops = st.sidebar.multiselect("Select Crops", crop_options, default=['All'])
selected_crops = crops if 'All' in selected_crops else selected_crops

# Year range filter
year_min = int(merge_df['Year'].min())
year_max = int(merge_df['Year'].max())
selected_years = st.sidebar.slider("Select Year Range", year_min, year_max, (year_min, year_max))

# Season filter
seasons = sorted(season_pattern_df['Season'].unique())
selected_season = st.sidebar.selectbox("Select Season", seasons)

# Reset filters button
if st.sidebar.button("Reset Filters"):
    selected_states = states[0]
    selected_crops = crops[:5]  # Reset to first 5 crops
    selected_years = (year_min, year_max)
    selected_season = seasons[0]

# Filter the dataframe based on selections
filtered_df = merge_df[
    (merge_df['State'].isin(selected_states)) &
    (merge_df['Crop'].isin(selected_crops)) &
    (merge_df['Year'].between(selected_years[0], selected_years[1]))
]

# Add metrics overview after filtering the data
st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
col_metrics = st.columns(4)

# Only show metrics if there's data
if not filtered_df.empty:
    with col_metrics[0]:
        total_states = len(filtered_df['State'].unique())
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #666;'>Total States</h4>
                <h2 style='color: #1E88E5;'>{}</h2>
            </div>
        """.format(total_states), unsafe_allow_html=True)

    with col_metrics[1]:
        total_crops = len(filtered_df['Crop'].unique())
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #666;'>Unique Crops</h4>
                <h2 style='color: #1E88E5;'>{}</h2>
            </div>
        """.format(total_crops), unsafe_allow_html=True)

    with col_metrics[2]:
        avg_rainfall = filtered_df['ANNUAL'].mean()
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #666;'>Avg. Rainfall</h4>
                <h2 style='color: #1E88E5;'>{:.0f} mm</h2>
            </div>
        """.format(avg_rainfall), unsafe_allow_html=True)

    with col_metrics[3]:
        total_production = filtered_df['Production'].sum()
        st.markdown("""
            <div class='metric-card'>
                <h4 style='color: #666;'>Total Production</h4>
                <h2 style='color: #1E88E5;'>{:.1f}M tonnes</h2>
            </div>
        """.format(total_production/1000000), unsafe_allow_html=True)
else:
    st.warning("No data available for the selected filters. Please adjust your selection.")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["☁️ Climate Analysis", "🌾 Production Analysis", "🔄 Climate-Agriculture Relationship"])

# Tab 1: Climate Analysis
with tab1:
    # State-wise Annual Rainfall Trend (full width)
    st.subheader("State-wise Annual Rainfall Trend")
    fig_rainfall_trend = px.line(
        filtered_df.groupby(['Year', 'State'])['ANNUAL'].mean().reset_index(),
        x='Year',
        y='ANNUAL',
        color='State',
        title='Annual Rainfall Trend by State',
        labels={'ANNUAL': 'Average Annual Rainfall (mm)'}
    )
    st.plotly_chart(fig_rainfall_trend, use_container_width=True, key="rainfall_trend")
    
    col1, col2 = st.columns(2)

with col1:
    # Decadal Rainfall Variability Index
    st.subheader("Decadal Rainfall Variability")
    # Add decade column
    decade_df = filtered_df.copy()
    decade_df['Decade'] = (decade_df['Year'] // 10) * 10
    decade_df['Decade'] = decade_df['Decade'].astype(str) + 's'
    
    fig_decadal = px.box(
        decade_df,
        x='Decade',
        y='ANNUAL',
        title='Decadal Rainfall Variability',
        labels={'ANNUAL': 'Rainfall (mm)', 'Decade': 'Decade'}
    )
    st.plotly_chart(fig_decadal, use_container_width=True, key="decadal_rainfall")

with col2:
    # Rainfall Seasonality Breakdown
    st.subheader("Rainfall Seasonality Breakdown")
    # Calculate seasonal rainfall by summing relevant months for each season
    def get_seasonal_rainfall(df):
        # Define months for each season
        season_months = {
            'Kharif': ['JUN', 'JUL', 'AUG', 'SEP'],
            'Rabi': ['OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR'],
            'Summer': ['APR', 'MAY'],
            'Whole Year': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        }
        
        seasonal_data = []
        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            for season, months in season_months.items():
                rainfall = year_data[months].mean().sum()
                seasonal_data.append({'Year': year, 'Season': season, 'Rainfall': rainfall})
        
        return pd.DataFrame(seasonal_data)

    seasonal_rainfall = get_seasonal_rainfall(filtered_df)
    fig_seasonal = px.area(
        seasonal_rainfall,
        x='Year',
        y='Rainfall',
        color='Season',
        title='Seasonal Rainfall Distribution',
        labels={'Rainfall': 'Rainfall (mm)'}
    )
    st.plotly_chart(fig_seasonal, use_container_width=True, key="seasonal_rainfall")

# Tab 2: Production Analysis
with tab2:
    # Top Crops by Production
    st.subheader("Top Crops by Production")
    top_crops = filtered_df.groupby('Crop')['Production'].sum().sort_values(ascending=True).tail(10)
    fig_top_crops = px.bar(
        top_crops,
        orientation='h',
        title='Top 10 Crops by Production',
        labels={'value': 'Production (tonnes)', 'Crop': 'Crop Name'}
    )
    st.plotly_chart(fig_top_crops, use_container_width=True, key="top_crops")

    # Crop Production Trend
    st.subheader("Crop Production Trend")
    crop_trend = filtered_df.groupby(['Year', 'Crop'])['Production'].sum().reset_index()
    fig_crop_trend = px.line(
        crop_trend,
        x='Year',
        y='Production',
        color='Crop',
        title='Crop Production Trends Over Time',
        labels={'Production': 'Production (tonnes)'}
    )
    st.plotly_chart(fig_crop_trend, use_container_width=True, key="crop_trend")

    # Top vs Bottom Performing Districts
    st.subheader("Top vs Bottom Performing Districts")
    district_performance = filtered_df.groupby(['State', 'District'])['Production'].sum().reset_index()
    
    # Get number of districts to show (minimum between 5 and half of available districts)
    n_districts = min(5, len(district_performance) // 2)
    
    if n_districts > 0:
        top_districts = district_performance.nlargest(n_districts, 'Production')
        bottom_districts = district_performance.nsmallest(n_districts, 'Production')
        
        # Add category before concatenation
        top_districts['Category'] = f'Top {n_districts}'
        bottom_districts['Category'] = f'Bottom {n_districts}'
        
        performance_comparison = pd.concat([top_districts, bottom_districts])
        
        fig_district_comparison = px.bar(
            performance_comparison,
            x='District',
            y='Production',
            color='Category',
            title=f'Top {n_districts} vs Bottom {n_districts} Districts by Production',
            labels={'Production': 'Production (tonnes)'},
            barmode='group'
        )
        fig_district_comparison.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_district_comparison, use_container_width=True, key="district_comparison")
    else:
        st.warning("Not enough district data available for comparison with current filters.")
    st.plotly_chart(fig_district_comparison, use_container_width=True)

    # District-wise Production Heatmap (full width)
    st.subheader("District-wise Production Heatmap")
    district_prod = filtered_df.groupby(['State', 'District'])['Production'].sum().reset_index()
    fig_district_heatmap = px.treemap(
        district_prod,
        path=[px.Constant("India"), 'State', 'District'],
        values='Production',
        title='District-wise Production Distribution',
        labels={'Production': 'Production (tonnes)'}
    )
    st.plotly_chart(fig_district_heatmap, use_container_width=True)

    # Crop Productivity (Yield per hectare) - full width
    st.subheader("Crop Productivity (Yield per hectare)")
    # Calculate yield (Production/Area) and handle invalid values
    productivity_df = filtered_df.copy()
    # Remove rows where Area or Production is 0 or negative
    productivity_df = productivity_df[
        (productivity_df['Area'] > 0) & 
        (productivity_df['Production'] >= 0)
    ]
    productivity_df['Yield'] = productivity_df['Production'] / productivity_df['Area']
    # Remove any remaining invalid values
    productivity_df = productivity_df.dropna(subset=['Yield'])
    
    # Normalize yield for bubble sizes to be more visible
    max_yield = productivity_df['Yield'].max()
    min_yield = productivity_df['Yield'].min()
    productivity_df['Bubble_Size'] = ((productivity_df['Yield'] - min_yield) / (max_yield - min_yield) * 40) + 10
    
    fig_bubble = px.scatter(
        productivity_df,
        x='Area',
        y='Production',
        size='Bubble_Size',
        color='Crop',
        hover_data=['State', 'District', 'Yield'],
        title='Crop Productivity Analysis',
        labels={
            'Area': 'Area (hectares)',
            'Production': 'Production (tonnes)',
            'Yield': 'Yield (tonnes/ha)'
        }
    )
    # Update layout for better visibility
    fig_bubble.update_traces(marker=dict(sizemode='area'))
    st.plotly_chart(fig_bubble, use_container_width=True)

# Tab 3: Climate-Agriculture Relationship
with tab3:
    # Rainfall vs. Crop Production Correlation (full width)
    st.subheader("Rainfall vs. Production Correlation")
    fig_correlation = px.scatter(
        filtered_df,
        x='ANNUAL',
        y='Production',
        color='State',
        trendline="ols",
        title='Rainfall vs. Production Correlation',
        labels={'ANNUAL': 'Annual Rainfall (mm)', 'Production': 'Production (tonnes)'}
    )
    st.plotly_chart(fig_correlation, use_container_width=True, key="rainfall_correlation")
    
    # Calculate correlation coefficient (optional)
    corr = filtered_df['ANNUAL'].corr(filtered_df['Production'])

    # Rainfall-Production Correlation Matrix (full width)
    st.subheader("Rainfall-Production Correlation Matrix")
    # Calculate correlation coefficients for each crop-state combination
    correlation_data = []
    for state in filtered_df['State'].unique():
        state_data = filtered_df[filtered_df['State'] == state]
        for crop in state_data['Crop'].unique():
            crop_data = state_data[state_data['Crop'] == crop]
            if len(crop_data) > 1:  # Need at least 2 points for correlation
                corr = crop_data['ANNUAL'].corr(crop_data['Production'])
                correlation_data.append({
                    'State': state,
                    'Crop': crop,
                    'Correlation': corr if not pd.isna(corr) else 0
                })
    
    corr_df = pd.DataFrame(correlation_data)
    fig_corr_matrix = px.density_heatmap(
        corr_df,
        x='Crop',
        y='State',
        z='Correlation',
        title='Rainfall-Production Correlation by State and Crop',
        labels={'Correlation': 'Correlation Coefficient'},
        color_continuous_scale='RdBu',
        range_color=[-1, 1]
    )
    fig_corr_matrix.update_layout(
        xaxis_tickangle=-45,
        height=600
    )
    st.plotly_chart(fig_corr_matrix, use_container_width=True, key="correlation_matrix")

    # Drought vs. Crop Shift Simulation (kept, full width)
    st.subheader("Drought vs. Crop Shift Simulation")
    # Calculate median rainfall for identifying drought years
    median_rainfall = filtered_df['ANNUAL'].median()
    drought_df = filtered_df.copy()
    drought_df['Drought_Year'] = drought_df['ANNUAL'] < median_rainfall
    
    # Compare water-intensive vs drought-resistant crops
    # Assuming crops with higher correlation with rainfall are more water-intensive
    crop_water_dependency = corr_df.groupby('Crop')['Correlation'].mean()
    water_intensive_crops = crop_water_dependency.nlargest(3).index
    drought_resistant_crops = crop_water_dependency.nsmallest(3).index
    
    drought_analysis = filtered_df[
        (filtered_df['Crop'].isin(water_intensive_crops) | 
         filtered_df['Crop'].isin(drought_resistant_crops))
    ].copy()
    
    drought_analysis['Crop_Type'] = drought_analysis['Crop'].apply(
        lambda x: 'Water Intensive' if x in water_intensive_crops else 'Drought Resistant'
    )
    
    fig_drought = px.line(
        drought_analysis.groupby(['Year', 'Crop_Type'])['Production'].mean().reset_index(),
        x='Year',
        y='Production',
        color='Crop_Type',
        title='Production Trends: Water Intensive vs Drought Resistant Crops',
        labels={'Production': 'Average Production (tonnes)'}
    )
    st.plotly_chart(fig_drought, use_container_width=True, key="drought_simulation")

    # Seasonal Crop Pattern Overview (full width)
    st.subheader("Seasonal Crop Pattern Overview")
    seasonal_crops = filtered_df.groupby(['Season', 'Crop']).size().reset_index(name='count')
    fig_seasonal_pattern = px.bar(
        seasonal_crops,
        x='Season',
        y='count',
        color='Crop',
        title='Seasonal Crop Distribution',
        labels={'count': 'Number of Records'}
    )
    st.plotly_chart(fig_seasonal_pattern, use_container_width=True, key="seasonal_pattern")

# Footer with enhanced styling
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 10px; margin-top: 3rem;'>
        <h3 style='color: #1E88E5; margin-bottom: 1rem;'>Data Sources</h3>
        <ul style='color: #666; margin-bottom: 2rem;'>
            <li>IMD Sub-Divisional Rainfall Dataset (data.gov.in)</li>
            <li>District-wise Season-wise Crop Production (data.gov.in)</li>
        </ul>
        <div style='text-align: center; padding-top: 1rem; border-top: 1px solid #ddd;'>
            <p style='color: #666;'>Generated with Project Samarth Prototype – Version 1.0</p>
            <p style='color: #999; font-size: 0.8rem;'>Built with Python + Streamlit + DuckDB</p>
        </div>
    </div>
""", unsafe_allow_html=True)
