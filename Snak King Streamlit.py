import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Page config
st.set_page_config(
    page_title="Snak King Product Analyzer",
    page_icon="👑",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .product-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('Salty Snack Market Data Fall 2025-1 (2).xlsx')
    df = df[df['Product Level'] == 'UPC'].copy()
    return df

# Reclassify flavors based on description
@st.cache_data
def reclassify_flavors(df):
    """Reclassify flavors based on obvious keywords in description"""
    
    def classify_flavor(row):
        desc = str(row['Description']).lower()
        current_flavor = str(row['FLAVOR'])
        
        # If flavor is already known and specific, keep it
        if current_flavor not in ['UNKNOWN', 'OTHER', 'nan']:
            return current_flavor
        
        # Butter flavors
        if any(word in desc for word in ['butter', 'movie theater']):
            return 'BUTTER'
        
        # Cheese flavors
        if any(word in desc for word in ['cheddar', 'cheese', 'cheesy', 'parmesan']):
            return 'CHEESE'
        
        # Sweet flavors
        if any(word in desc for word in ['caramel', 'kettle corn', 'sweet', 'chocolate']):
            return 'CARAMEL'
        
        # Spicy flavors
        if any(word in desc for word in ['jalapeno', 'habanero', 'sriracha', 'flamin', 'hot', 'spicy', 'chile', 'chili']):
            return 'HOT / SPICY / CHILE'
        
        # Ranch
        if 'ranch' in desc:
            return 'RANCH'
        
        # BBQ
        if any(word in desc for word in ['bbq', 'barbeque', 'barbecue']):
            return 'BBQ'
        
        # Sea salt / plain
        if any(word in desc for word in ['sea salt', 'salted', 'original', 'classic', 'plain']):
            return 'SALTED / SEA SALT'
        
        # Keep original if no match
        return current_flavor
    
    df['FLAVOR'] = df.apply(classify_flavor, axis=1)
    return df

# Function to create opportunity matrix
def create_opportunity_matrix(data, title_suffix=""):
    subcategory_analysis = data.groupby('Subcategory').agg({
        'Dollars': 'sum',
        'Dollars, Yago': 'sum',
        'Dollars/TDP': 'mean'
    }).reset_index()
    
    subcategory_analysis['YoY_%'] = (
        (subcategory_analysis['Dollars'] - subcategory_analysis['Dollars, Yago']) / 
        subcategory_analysis['Dollars, Yago'] * 100
    )
    
    subcategory_analysis['Velocity_K'] = subcategory_analysis['Dollars/TDP'] / 1000
    subcategory_analysis['Revenue_M'] = subcategory_analysis['Dollars'] / 1_000_000
    
    subcategory_analysis = subcategory_analysis.sort_values('Dollars', ascending=False)
    
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in subcategory_analysis['YoY_%']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=subcategory_analysis['Velocity_K'],
        y=subcategory_analysis['YoY_%'],
        mode='markers+text',
        marker=dict(
            size=subcategory_analysis['Revenue_M'],
            sizemode='diameter',
            sizeref=max(subcategory_analysis['Revenue_M']) / 100,
            color=colors,
            line=dict(width=2, color='white'),
            opacity=0.7
        ),
        text=subcategory_analysis['Subcategory'].str.replace('SS ', ''),
        textposition='top center',
        textfont=dict(size=10, color='#2C3E50', family='Arial Black'),
        hovertemplate='<b>%{text}</b><br>' +
                      'Velocity: $%{x:.1f}k/TDP<br>' +
                      'YoY Growth: %{y:.1f}%<br>' +
                      'Revenue: $%{marker.size:.0f}M<br>' +
                      '<extra></extra>',
        name=''
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=subcategory_analysis['Velocity_K'].median(), line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title={
            'text': f'Snack Category Opportunity Matrix{title_suffix}<br><sub>Bubble size = Market size (revenue)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        xaxis_title='Velocity ($/TDP in thousands) - How Fast It Sells',
        yaxis_title='Year-over-Year Growth (%) - Momentum',
        template='plotly_white',
        height=600,
        font=dict(size=12),
        hovermode='closest'
    )
    
    return fig

# Slide 3: Brand Performance
def create_brand_performance(data, product_name):
    # Aggregate by brand
    brand_analysis = data.groupby('Brand').agg({
        'Dollars': 'sum',
        'Dollars/TDP': 'mean'
    }).reset_index()
    
    brand_analysis['Revenue_M'] = brand_analysis['Dollars'] / 1_000_000
    brand_analysis['Velocity_K'] = brand_analysis['Dollars/TDP'] / 1000
    
    # Get top 10 brands by revenue
    brand_analysis = brand_analysis.nlargest(10, 'Revenue_M')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=brand_analysis['Velocity_K'],
        y=brand_analysis['Revenue_M'],
        mode='markers+text',
        marker=dict(
            size=brand_analysis['Revenue_M'],
            sizemode='diameter',
            sizeref=max(brand_analysis['Revenue_M']) / 50,
            color='#3498db',
            line=dict(width=2, color='white'),
            opacity=0.7
        ),
        text=brand_analysis['Brand'],
        textposition='top center',
        textfont=dict(size=9, color='#2C3E50'),
        hovertemplate='<b>%{text}</b><br>' +
                      'Velocity: $%{x:.1f}k/TDP<br>' +
                      'Revenue: $%{y:.1f}M<br>' +
                      '<extra></extra>',
        name=''
    ))
    
    # Add quadrant lines
    fig.add_hline(y=brand_analysis['Revenue_M'].median(), line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=brand_analysis['Velocity_K'].median(), line_dash="dash", line_color="gray", line_width=1)
    
    # Add quadrant label
    fig.add_annotation(
        x=brand_analysis['Velocity_K'].max() * 0.9,
        y=brand_analysis['Revenue_M'].max() * 0.9,
        text="HIGH VELOCITY<br>HIGH VOLUME<br>RETAILER'S DREAM",
        showarrow=False,
        font=dict(size=11, color='darkgreen', family='Arial Black'),
        bgcolor='rgba(46, 204, 113, 0.2)',
        borderpad=8
    )
    
    fig.update_layout(
        title={
            'text': f'{product_name} Brand Performance: Velocity vs Market Size',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        xaxis_title='Velocity (Dollars/TDP) - How Fast Products Sell',
        yaxis_title='Market Size (Revenue in Millions $)',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig

# Slide 4: Flavor Performance Matrix - FIXED VERSION
def create_flavor_performance(data, product_name):
    # Aggregate by flavor
    flavor_analysis = data.groupby('FLAVOR').agg({
        'Dollars': 'sum',
        'Dollars, Yago': 'sum',
        'Dollars/TDP': 'mean',
        'TDP': 'sum'
    }).reset_index()
    
    # Calculate YoY growth
    flavor_analysis['YoY_%'] = (
        (flavor_analysis['Dollars'] - flavor_analysis['Dollars, Yago']) / 
        flavor_analysis['Dollars, Yago'] * 100
    )
    
    # Calculate distribution percentage
    total_tdp = data['TDP'].sum()
    flavor_analysis['Distribution_%'] = (flavor_analysis['TDP'] / total_tdp * 100).round(0).astype(int)
    
    # Replace infinity with NaN, then drop NaN
    flavor_analysis['YoY_%'] = flavor_analysis['YoY_%'].replace([float('inf'), float('-inf')], float('nan'))
    flavor_analysis = flavor_analysis.dropna(subset=['YoY_%'])
    
    # Filter out very small flavors AFTER calculating growth
    flavor_analysis = flavor_analysis[flavor_analysis['Dollars'] > 500000]  # >$500k revenue
    
    # Velocity is already in $/TDP - DO NOT divide by 1000
    flavor_analysis['Velocity_Display'] = flavor_analysis['Dollars/TDP']
    flavor_analysis['Revenue_M'] = flavor_analysis['Dollars'] / 1_000_000
    
    # Take top 12 by revenue to avoid outliers
    flavor_analysis = flavor_analysis.nlargest(12, 'Revenue_M')
    
    # Sort by revenue for consistent display
    flavor_analysis = flavor_analysis.sort_values('Revenue_M', ascending=False)
    
    # Create separate traces for growing vs declining
    growing = flavor_analysis[flavor_analysis['YoY_%'] >= 0]
    declining = flavor_analysis[flavor_analysis['YoY_%'] < 0]
    
    fig = go.Figure()
    
    # Growing flavors - FIXED
    if len(growing) > 0:
        fig.add_trace(go.Scatter(
            x=growing['Velocity_Display'],
            y=growing['YoY_%'],
            mode='markers+text',
            marker=dict(
                size=growing['Revenue_M'],
                sizemode='diameter',
                sizeref=max(flavor_analysis['Revenue_M']) / 60,
                color='#27AE60',
                line=dict(width=2, color='white'),
                opacity=0.7
            ),
            text=growing['FLAVOR'],
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>' +
                          'Velocity: $%{x:,.0f}/TDP<br>' +
                          'Growth: %{y:.1f}%<br>' +
                          'Revenue: $' + growing['Revenue_M'].apply(lambda x: f'{x:.1f}M').values + '<br>' +
                          'Distribution: ' + growing['Distribution_%'].astype(str).values + '% stores<br>' +
                          '<extra></extra>',
            name='Growing',
            showlegend=False
        ))
    
    # Declining flavors
    if len(declining) > 0:
        fig.add_trace(go.Scatter(
            x=declining['Velocity_Display'],
            y=declining['YoY_%'],
            mode='markers+text',
            marker=dict(
                size=declining['Revenue_M'],
                sizemode='diameter',
                sizeref=max(flavor_analysis['Revenue_M']) / 60,
                color='#E74C3C',
                line=dict(width=2, color='white'),
                opacity=0.7
            ),
            text=declining['FLAVOR'],
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>' +
                          'Velocity: $%{x:,.0f}/TDP<br>' +
                          'Growth: %{y:.1f}%<br>' +
                          'Revenue: $' + declining['Revenue_M'].apply(lambda x: f'{x:.1f}M').values + '<br>' +
                          'Distribution: ' + declining['Distribution_%'].astype(str).values + '% stores<br>' +
                          '<extra></extra>',
            name='Declining',
            showlegend=False
        ))
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=2, opacity=0.5)
    fig.add_vline(x=flavor_analysis['Velocity_Display'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    if len(growing) > 0:
        max_vel = flavor_analysis['Velocity_Display'].max()
        max_growth = flavor_analysis['YoY_%'].max()
        median_vel = flavor_analysis['Velocity_Display'].median()
        
        # Top right - best performers
        fig.add_annotation(
            x=max_vel * 0.85,
            y=max_growth * 0.85,
            text="⭐ HIGH VELOCITY<br>HIGH GROWTH<br>(Retailer's Dream)",
            showarrow=False,
            font=dict(size=10, color='darkgreen', family='Arial Black'),
            bgcolor='rgba(144, 238, 144, 0.3)',
            borderpad=5
        )
        
        # Top left - slower but growing
        if median_vel > flavor_analysis['Velocity_Display'].min():
            fig.add_annotation(
                x=flavor_analysis['Velocity_Display'].min() * 1.2,
                y=max_growth * 0.85,
                text="GROWING<br>SLOWER MOVERS",
                showarrow=False,
                font=dict(size=9, color='darkgreen'),
                bgcolor='rgba(173, 216, 230, 0.3)',
                borderpad=5
            )
    
    fig.update_layout(
        title={
            'text': f'{product_name} Flavor Performance Matrix<br><sub>Bubble size = Revenue (market size)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        xaxis_title='Velocity (Dollars/TDP) - How Fast It Sells',
        yaxis_title='Year-over-Year Growth (%) - Momentum',
        template='plotly_white',
        height=600,
        showlegend=False,
        hovermode='closest',
        font=dict(size=11)
    )
    
    return fig

# Slide 5: Price Tier Analysis
def create_price_tier_analysis(data, product_name):
    # Calculate price per unit
    data_copy = data.copy()
    data_copy['Price_Per_Unit'] = data_copy['Dollars'] / data_copy['Units']
    
    # Define premium (top 25%) and value (bottom 25%) by price
    price_75 = data_copy['Price_Per_Unit'].quantile(0.75)
    price_25 = data_copy['Price_Per_Unit'].quantile(0.25)
    
    data_copy['Tier'] = 'Mid-Tier'
    data_copy.loc[data_copy['Price_Per_Unit'] >= price_75, 'Tier'] = 'Premium (Top 25% Price)'
    data_copy.loc[data_copy['Price_Per_Unit'] <= price_25, 'Tier'] = 'Value (Bottom 25% Price)'
    
    # Aggregate by tier
    tier_analysis = data_copy.groupby('Tier').agg({
        'Dollars': 'sum',
        'Dollars, Yago': 'sum',
        'Dollars/TDP': 'mean',
        'Price_Per_Unit': 'mean'
    }).reset_index()
    
    tier_analysis['YoY_%'] = (
        (tier_analysis['Dollars'] - tier_analysis['Dollars, Yago']) / 
        tier_analysis['Dollars, Yago'] * 100
    )
    tier_analysis['Revenue_M'] = tier_analysis['Dollars'] / 1_000_000
    tier_analysis['Velocity_K'] = tier_analysis['Dollars/TDP'] / 1000
    
    # Filter to just Premium and Value
    tier_analysis = tier_analysis[tier_analysis['Tier'].isin(['Premium (Top 25% Price)', 'Value (Bottom 25% Price)'])]
    
    # Create grouped bar chart
    metrics = ['Revenue ($M)', 'Growth (%)', 'Velocity ($K/TDP)', 'Avg Price ($/unit)']
    
    fig = go.Figure()
    
    premium_data = tier_analysis[tier_analysis['Tier'] == 'Premium (Top 25% Price)'].iloc[0]
    value_data = tier_analysis[tier_analysis['Tier'] == 'Value (Bottom 25% Price)'].iloc[0]
    
    premium_values = [premium_data['Revenue_M'], premium_data['YoY_%'], 
                     premium_data['Velocity_K'], premium_data['Price_Per_Unit']]
    value_values = [value_data['Revenue_M'], value_data['YoY_%'], 
                   value_data['Velocity_K'], value_data['Price_Per_Unit']]
    
    fig.add_trace(go.Bar(
        name='Premium (Top 25% Price)',
        x=metrics,
        y=premium_values,
        marker_color='#9B59B6',
        text=[f"${x:.1f}M" if i==0 else f"{x:+.1f}%" if i==1 else f"${x:.1f}k" if i==2 else f"${x:.2f}" 
              for i, x in enumerate(premium_values)],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Value (Bottom 25% Price)',
        x=metrics,
        y=value_values,
        marker_color='#3498DB',
        text=[f"${x:.1f}M" if i==0 else f"{x:+.1f}%" if i==1 else f"${x:.1f}k" if i==2 else f"${x:.2f}" 
              for i, x in enumerate(value_values)],
        textposition='outside'
    ))
    
    # Add annotation about premium performance
    if premium_data['YoY_%'] > value_data['YoY_%']:
        annotation_text = f"Premium {product_name.lower()} is outperforming:<br>" \
                         f"✓ Growing {premium_data['YoY_%']:.1f}% vs {value_data['YoY_%']:.1f}%<br>" \
                         f"✓ Commands ${premium_data['Price_Per_Unit']:.2f} vs ${value_data['Price_Per_Unit']:.2f} per unit<br>" \
                         f"✓ Higher velocity despite premium pricing"
        
        fig.add_annotation(
            x=2, y=max(premium_values + value_values) * 0.7,
            text=annotation_text,
            showarrow=False,
            font=dict(size=10, color='#9B59B6'),
            bgcolor='rgba(155, 89, 182, 0.1)',
            bordercolor='#9B59B6',
            borderwidth=2,
            borderpad=10,
            align='left'
        )
    
    fig.update_layout(
        title={
            'text': f'Premium vs Value {product_name} Performance',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        yaxis_title='',
        template='plotly_white',
        height=600,
        barmode='group',
        showlegend=True,
        legend=dict(x=0.7, y=1.0)
    )
    
    return fig

# Slide 6: Package Size Performance - FIXED VERSION
def create_size_performance(data, product_name):
    # Create size groups
    def group_size(size):
        if pd.isna(size):
            return 'UNKNOWN'
        size = float(size)
        if size < 2:
            return '<2 oz'
        elif size <= 5:
            return '2-5 oz'
        elif size <= 8:
            return '5-8 oz'
        elif size <= 12:
            return '8-12 oz'
        else:
            return '12+ oz'
    
    # Add size group column
    data_copy = data.copy()
    data_copy['Size_Group'] = data_copy['SIZE'].apply(group_size)
    
    # Aggregate by size group
    size_analysis = data_copy.groupby('Size_Group').agg({
        'Dollars': 'sum',
        'Dollars, Yago': 'sum',
        'Dollars/TDP': 'mean',
        'TDP': 'sum'
    }).reset_index()
    
    # Calculate YoY growth
    size_analysis['YoY_%'] = (
        (size_analysis['Dollars'] - size_analysis['Dollars, Yago']) / 
        size_analysis['Dollars, Yago'] * 100
    )
    
    # Calculate distribution percentage
    total_tdp = data_copy['TDP'].sum()
    size_analysis['Distribution_%'] = (size_analysis['TDP'] / total_tdp * 100).round(0).astype(int)
    
    # Replace infinity with NaN, then drop NaN
    size_analysis['YoY_%'] = size_analysis['YoY_%'].replace([float('inf'), float('-inf')], float('nan'))
    size_analysis = size_analysis.dropna(subset=['YoY_%'])
    
    # Filter out very small sizes AFTER calculating growth
    size_analysis = size_analysis[size_analysis['Dollars'] > 2000000]  # >$2M revenue
    
    # Remove UNKNOWN size group if it exists
    size_analysis = size_analysis[size_analysis['Size_Group'] != 'UNKNOWN']
    
    # Velocity is already in $/TDP - DO NOT divide by 1000
    size_analysis['Velocity_Display'] = size_analysis['Dollars/TDP']
    size_analysis['Revenue_M'] = size_analysis['Dollars'] / 1_000_000
    
    # Sort by size group order
    size_order = ['<2 oz', '2-5 oz', '5-8 oz', '8-12 oz', '12+ oz']
    size_analysis['Size_Order'] = size_analysis['Size_Group'].apply(
        lambda x: size_order.index(x) if x in size_order else 999
    )
    size_analysis = size_analysis.sort_values('Size_Order')
    
    # Create separate traces for growing vs declining
    growing = size_analysis[size_analysis['YoY_%'] >= 0]
    declining = size_analysis[size_analysis['YoY_%'] < 0]
    
    fig = go.Figure()
    
    # Growing sizes
    if len(growing) > 0:
        fig.add_trace(go.Scatter(
            x=growing['Velocity_Display'],
            y=growing['YoY_%'],
            mode='markers+text',
            marker=dict(
                size=growing['Revenue_M'],
                sizemode='diameter',
                sizeref=max(size_analysis['Revenue_M']) / 45,
                color='#27AE60',
                line=dict(width=2, color='white'),
                opacity=0.7
            ),
            text=growing['Size_Group'],
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            hovertemplate='<b>%{text}</b><br>' +
                          'Velocity: $%{x:,.0f}/TDP<br>' +
                          'Growth: %{y:.1f}%<br>' +
                          'Revenue: $' + growing['Revenue_M'].apply(lambda x: f'{x:.0f}M').values + '<br>' +
                          'Distribution: ' + growing['Distribution_%'].astype(str).values + '% stores<br>' +
                          '<extra></extra>',
            name='Growing',
            showlegend=True
        ))
    
    # Declining sizes
    if len(declining) > 0:
        fig.add_trace(go.Scatter(
            x=declining['Velocity_Display'],
            y=declining['YoY_%'],
            mode='markers+text',
            marker=dict(
                size=declining['Revenue_M'],
                sizemode='diameter',
                sizeref=max(size_analysis['Revenue_M']) / 45,
                color='#E74C3C',
                line=dict(width=2, color='white'),
                opacity=0.7
            ),
            text=declining['Size_Group'],
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            hovertemplate='<b>%{text}</b><br>' +
                          'Velocity: $%{x:,.0f}/TDP<br>' +
                          'Growth: %{y:.1f}%<br>' +
                          'Revenue: $' + declining['Revenue_M'].apply(lambda x: f'{x:.0f}M').values + '<br>' +
                          'Distribution: ' + declining['Distribution_%'].astype(str).values + '% stores<br>' +
                          '<extra></extra>',
            name='Declining',
            showlegend=True
        ))
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=2, opacity=0.5)
    fig.add_vline(x=size_analysis['Velocity_Display'].median(), line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels for high performers
    if len(growing) > 0:
        max_vel = size_analysis['Velocity_Display'].max()
        max_growth = size_analysis['YoY_%'].max()
        
        fig.add_annotation(
            x=max_vel * 0.85,
            y=max_growth * 0.85,
            text="⭐ HIGH VELOCITY<br>HIGH GROWTH<br>(Retailer's Dream)",
            showarrow=False,
            font=dict(size=10, color='darkgreen', family='Arial Black'),
            bgcolor='rgba(144, 238, 144, 0.3)',
            borderpad=5
        )
    
    # Add legend annotation - moved to top right
    fig.add_annotation(
        x=0.98, y=0.98,
        xref='paper', yref='paper',
        text='<b>Bubble Size = Revenue</b><br>(Market opportunity)<br><br>🟢 Growing<br>🔴 Declining',
        showarrow=False,
        font=dict(size=11),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=2,
        borderpad=10,
        align='left',
        xanchor='right',
        yanchor='top'
    )
    
    fig.update_layout(
        title={
            'text': f'Package Size Performance Matrix<br><sub>Bubble size = Market size (revenue)</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        xaxis_title='Velocity ($/TDP) - How Fast It Sells',
        yaxis_title='Year-over-Year Growth (%) - Momentum',
        template='plotly_white',
        height=600,
        showlegend=False,
        hovermode='closest',
        font=dict(size=11)
    )
    
    return fig

# Slide 7: Store Performance
def create_store_performance(data, product_name):
    # Get top 6 retailers by revenue
    store_analysis = data.groupby('Retail Account').agg({
        'Dollars': 'sum',
        'Dollars, Yago': 'sum'
    }).reset_index()
    
    store_analysis['YoY_%'] = (
        (store_analysis['Dollars'] - store_analysis['Dollars, Yago']) / 
        store_analysis['Dollars, Yago'] * 100
    )
    store_analysis['Revenue_M'] = store_analysis['Dollars'] / 1_000_000
    
    # Get top 6
    store_analysis = store_analysis.nlargest(6, 'Revenue_M')
    store_analysis = store_analysis.sort_values('Revenue_M', ascending=True)
    
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in store_analysis['YoY_%']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=store_analysis['Retail Account'],
        x=store_analysis['Revenue_M'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"${x:.0f}M<br>{y:+.1f}%" for x, y in zip(store_analysis['Revenue_M'], store_analysis['YoY_%'])],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                      'Revenue: $%{x:.0f}M<br>' +
                      '<extra></extra>'
    ))
    
    # Add growth % annotation
    for i, (store, growth) in enumerate(zip(store_analysis['Retail Account'], store_analysis['YoY_%'])):
        fig.add_annotation(
            x=store_analysis['Revenue_M'].iloc[i] / 2,
            y=i,
            text=f"{growth:+.1f}%",
            showarrow=False,
            font=dict(size=12, color='white', family='Arial Black')
        )
    
    fig.update_layout(
        title={
            'text': f'{product_name} Market by Retailer<br><sub>Green = Growing | Red = Declining | Snak King has zero presence</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        xaxis_title='Market Size (Millions $)',
        yaxis_title='',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig

# Load and process data
df = load_data()
df = reclassify_flavors(df)

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "🍿 Popcorn", "🌽 Tortilla Chips", "📦 Variety Snack Packs"]
)

# HOME PAGE
if page == "🏠 Home":
    # Title
    st.markdown('<div class="main-header"><h1 style="color: white; margin: 0;">🎯 Snak King Product Opportunity Analyzer</h1><p style="color: white; margin: 10px 0 0 0;">Data-Driven Insights for Product Development</p></div>', unsafe_allow_html=True)
    
    # Single opportunity matrix
    st.plotly_chart(create_opportunity_matrix(df, " - OVERALL MARKET"), use_container_width=True)
    
    # Market Opportunity Equation
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;'>
        <h4 style='color: #2C3E50; margin-bottom: 10px;'>Market Opportunity Framework</h4>
        <p style='font-size: 18px; color: #34495E; font-family: monospace;'>
            <b>Opportunity = Growth × Velocity × Market Size</b>
        </p>
        <p style='font-size: 14px; color: #7f8c8d; margin-top: 10px;'>
            Where: <b>Growth</b> = YoY % | <b>Velocity</b> = $/TDP | <b>Market Size</b> = Revenue $
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional Visualizations Section
    st.markdown("### 📊 Additional Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; text-align: center;'>
            <h4 style='color: #2C3E50; margin-bottom: 15px;'>Interactive Market Analysis</h4>
            <a href='https://flourish-user-preview.com/26318359/cacs0E7-Y07zdRmWo8cMDlizk5uERwfH7-At41IGoTJOBwzq4kxPjBEhAMYc-trN/' target='_blank' style='text-decoration: none;'>
                <button style='background-color: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold;'>
                    View Visualization 1 →
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; text-align: center;'>
            <h4 style='color: #2C3E50; margin-bottom: 15px;'>Market Trends Dashboard</h4>
            <a href='https://public.flourish.studio/visualisation/26298160/' target='_blank' style='text-decoration: none;'>
                <button style='background-color: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold;'>
                    View Visualization 2 →
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

# PRODUCT PAGES
else:
    # Determine which product and filter data
    if page == "🍿 Popcorn":
        category = "SS POPCORN"
        product_name = "Popcorn"
        emoji = "🍿"
    elif page == "🌽 Tortilla Chips":
        category = "SS TORTILLA & CORN CHIPS"
        product_name = "Tortilla Chips"
        emoji = "🌽"
    else:  # Variety Snack Packs
        category = "SS SNACKS VARIETY PACKS"
        product_name = "Variety Snack Packs"
        emoji = "📦"
    
    # Filter data
    product_data = df[df['Subcategory'] == category].copy()
    
    # Page title
    st.title(f"{emoji} {product_name} Market Analysis")
    st.markdown("---")
    
    # Slide 3: Brand Performance
    st.markdown("### 🏷️ Brand Performance Analysis")
    st.plotly_chart(create_brand_performance(product_data, product_name), use_container_width=True)
    
    st.markdown("---")
    
    # Slide 4: Flavor Performance
    st.markdown("### 🎨 Flavor Opportunity Matrix")
    st.plotly_chart(create_flavor_performance(product_data, product_name), use_container_width=True)
    
    st.markdown("---")
    
    # Slide 5: Price Tier Analysis
    st.markdown("### 💰 Premium vs Value Performance")
    st.plotly_chart(create_price_tier_analysis(product_data, product_name), use_container_width=True)
    
    st.markdown("---")
    
    # Slide 6: Package Size Performance
    st.markdown("### 📦 Package Size Opportunity")
    st.plotly_chart(create_size_performance(product_data, product_name), use_container_width=True)
    
    st.markdown("---")
    
    # Slide 7: Store Performance
    st.markdown("### 🏪 Retailer Performance")
    st.plotly_chart(create_store_performance(product_data, product_name), use_container_width=True)
    
    st.markdown("---")
    
    # Product Recommendation Card
    st.markdown("## 🎯 Recommended Product Specifications")
    
    st.markdown(f"""
    <div class="product-card">
        <h3>{emoji} Snak King {product_name}</h3>
        <br>
        <h4>Product Attributes:</h4>
        <ul>
            <li>📦 <b>Package Size:</b> TBD (Based on size performance analysis)</li>
            <li>🎨 <b>Flavor Profile:</b> TBD (Based on flavor opportunity matrix)</li>
            <li>💰 <b>Price Point:</b> TBD (Premium or Value positioning)</li>
            <li>🏷️ <b>Key Features:</b> TBD (Non-GMO, Gluten-Free, etc.)</li>
        </ul>
        <br>
        <h4>Target Retailer:</h4>
        <p><b>TBD</b> (Based on retailer growth and opportunity analysis)</p>
        <br>
        <h4>Why This Works:</h4>
        <ul>
            <li>✅ Aligns with high-growth flavor trends</li>
            <li>✅ Targets optimal price tier based on market dynamics</li>
            <li>✅ Positioned for retailer success (high velocity potential)</li>
            <li>✅ Fills identified market white space</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
