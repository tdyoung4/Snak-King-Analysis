import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Load data
print("Loading data...")
df = pd.read_excel('Salty Snack Market Data Fall 2025-1 (2).xlsx')
df = df[df['Product Level'] == 'UPC'].copy()

print(f"Total records: {len(df)}")
print(f"\nUnique brands: {df['Brand'].nunique()}")

# Filter for Snak King brand
snak_king_data = df[df['Brand'].str.contains('SNAK KING', case=False, na=False)].copy()

print(f"\n{'='*80}")
print(f"SNAK KING ANALYSIS")
print(f"{'='*80}")
print(f"Total Snak King products: {len(snak_king_data)}")

if len(snak_king_data) == 0:
    print("\n⚠️  WARNING: No Snak King products found in dataset!")
    print("\nSearching for similar brand names...")
    
    # Search for variations
    similar_brands = df[df['Brand'].str.contains('KING|SNACK', case=False, na=False)]['Brand'].unique()
    print(f"\nBrands containing 'KING' or 'SNACK': {len(similar_brands)}")
    for brand in sorted(similar_brands)[:20]:
        print(f"  - {brand}")
    
    print("\n" + "="*80)
    print("ALTERNATIVE ANALYSIS: TOP BRANDS FOR COMPARISON")
    print("="*80)
    
    # Analyze top brands instead
    brand_summary = df.groupby('Brand').agg({
        'Dollars': 'sum',
        'Dollars, Yago': 'sum',
        'Dollars/TDP': 'mean',
        'Units': 'sum',
        'Subcategory': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    brand_summary['Revenue_M'] = brand_summary['Dollars'] / 1_000_000
    brand_summary['YoY_%'] = ((brand_summary['Dollars'] - brand_summary['Dollars, Yago']) / 
                               brand_summary['Dollars, Yago'] * 100)
    brand_summary = brand_summary.sort_values('Revenue_M', ascending=False).head(20)
    
    print("\nTop 20 Brands by Revenue:")
    print(brand_summary[['Brand', 'Revenue_M', 'YoY_%', 'Subcategory']].to_string(index=False))
    
    # Create visualization of top brands
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=brand_summary['Brand'][:10],
        y=brand_summary['Revenue_M'][:10],
        marker_color=['#2ECC71' if x > 0 else '#E74C3C' for x in brand_summary['YoY_%'][:10]],
        text=[f"${x:.1f}M<br>{y:+.1f}%" for x, y in zip(brand_summary['Revenue_M'][:10], brand_summary['YoY_%'][:10])],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:.1f}M<extra></extra>'
    ))
    
    fig.update_layout(
        title='Top 10 Salty Snack Brands by Revenue<br><sub>Green = Growing | Red = Declining</sub>',
        xaxis_title='Brand',
        yaxis_title='Revenue (Millions $)',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    fig.write_html('/mnt/user-data/outputs/top_brands_comparison.html')
    print(f"\n✅ Saved: top_brands_comparison.html")

else:
    print(f"\n✅ Found {len(snak_king_data)} Snak King products!")
    
    # Product breakdown by subcategory
    print("\n" + "="*80)
    print("SNAK KING PRODUCT BREAKDOWN")
    print("="*80)
    
    subcategory_breakdown = snak_king_data.groupby('Subcategory').agg({
        'Dollars': 'sum',
        'Dollars, Yago': 'sum',
        'Units': 'sum',
        'TDP': 'sum',
        'Dollars/TDP': 'mean'
    }).reset_index()
    
    subcategory_breakdown['Revenue_M'] = subcategory_breakdown['Dollars'] / 1_000_000
    subcategory_breakdown['YoY_%'] = ((subcategory_breakdown['Dollars'] - subcategory_breakdown['Dollars, Yago']) / 
                                      subcategory_breakdown['Dollars, Yago'] * 100)
    subcategory_breakdown['Velocity_K'] = subcategory_breakdown['Dollars/TDP'] / 1000
    
    subcategory_breakdown = subcategory_breakdown.sort_values('Revenue_M', ascending=False)
    
    print("\nSnak King by Category:")
    print(subcategory_breakdown[['Subcategory', 'Revenue_M', 'YoY_%', 'Velocity_K']].to_string(index=False))
    
    # Graph 1: Revenue by Category
    fig1 = go.Figure()
    
    fig1.add_trace(go.Bar(
        x=subcategory_breakdown['Subcategory'].str.replace('SS ', ''),
        y=subcategory_breakdown['Revenue_M'],
        marker_color='#FF6B6B',
        text=[f"${x:.2f}M" for x in subcategory_breakdown['Revenue_M']],
        textposition='outside'
    ))
    
    fig1.update_layout(
        title='Snak King Revenue by Product Category',
        xaxis_title='Category',
        yaxis_title='Revenue (Millions $)',
        template='plotly_white',
        height=500
    )
    
    fig1.write_html('/mnt/user-data/outputs/snak_king_revenue_by_category.html')
    print(f"\n✅ Saved: snak_king_revenue_by_category.html")
    
    # Graph 2: Growth Rate by Category
    fig2 = go.Figure()
    
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in subcategory_breakdown['YoY_%']]
    
    fig2.add_trace(go.Bar(
        x=subcategory_breakdown['Subcategory'].str.replace('SS ', ''),
        y=subcategory_breakdown['YoY_%'],
        marker_color=colors,
        text=[f"{x:+.1f}%" for x in subcategory_breakdown['YoY_%']],
        textposition='outside'
    ))
    
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", line_width=2)
    
    fig2.update_layout(
        title='Snak King Year-over-Year Growth by Category<br><sub>Green = Growing | Red = Declining</sub>',
        xaxis_title='Category',
        yaxis_title='YoY Growth (%)',
        template='plotly_white',
        height=500
    )
    
    fig2.write_html('/mnt/user-data/outputs/snak_king_growth_by_category.html')
    print(f"✅ Saved: snak_king_growth_by_category.html")
    
    # Graph 3: Velocity vs Growth Matrix
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=subcategory_breakdown['Velocity_K'],
        y=subcategory_breakdown['YoY_%'],
        mode='markers+text',
        marker=dict(
            size=subcategory_breakdown['Revenue_M'] * 50,
            color=colors,
            line=dict(width=2, color='white'),
            opacity=0.7
        ),
        text=subcategory_breakdown['Subcategory'].str.replace('SS ', ''),
        textposition='top center',
        textfont=dict(size=10, color='#2C3E50', family='Arial Black'),
        hovertemplate='<b>%{text}</b><br>Velocity: $%{x:.1f}k/TDP<br>Growth: %{y:.1f}%<extra></extra>'
    ))
    
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig3.add_vline(x=subcategory_breakdown['Velocity_K'].median(), line_dash="dash", line_color="gray", line_width=1)
    
    fig3.update_layout(
        title='Snak King Performance Matrix<br><sub>Bubble size = Revenue | Green = Growing | Red = Declining</sub>',
        xaxis_title='Velocity ($/TDP in thousands)',
        yaxis_title='Year-over-Year Growth (%)',
        template='plotly_white',
        height=600
    )
    
    fig3.write_html('/mnt/user-data/outputs/snak_king_performance_matrix.html')
    print(f"✅ Saved: snak_king_performance_matrix.html")
    
    # Graph 4: Flavor Analysis (if available)
    if 'FLAVOR' in snak_king_data.columns:
        print("\n" + "="*80)
        print("SNAK KING FLAVOR ANALYSIS")
        print("="*80)
        
        flavor_analysis = snak_king_data.groupby('FLAVOR').agg({
            'Dollars': 'sum',
            'Dollars, Yago': 'sum',
            'Units': 'sum'
        }).reset_index()
        
        flavor_analysis['Revenue_M'] = flavor_analysis['Dollars'] / 1_000_000
        flavor_analysis['YoY_%'] = ((flavor_analysis['Dollars'] - flavor_analysis['Dollars, Yago']) / 
                                    flavor_analysis['Dollars, Yago'] * 100)
        flavor_analysis = flavor_analysis.sort_values('Revenue_M', ascending=False)
        
        print("\nTop Flavors:")
        print(flavor_analysis[['FLAVOR', 'Revenue_M', 'YoY_%']].head(10).to_string(index=False))
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Pie(
            labels=flavor_analysis['FLAVOR'][:8],
            values=flavor_analysis['Revenue_M'][:8],
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        ))
        
        fig4.update_layout(
            title='Snak King Revenue by Flavor (Top 8)',
            template='plotly_white',
            height=500
        )
        
        fig4.write_html('/mnt/user-data/outputs/snak_king_flavor_breakdown.html')
        print(f"\n✅ Saved: snak_king_flavor_breakdown.html")
    
    # Graph 5: Package Size Analysis (if available)
    if 'SIZE' in snak_king_data.columns:
        print("\n" + "="*80)
        print("SNAK KING PACKAGE SIZE ANALYSIS")
        print("="*80)
        
        # Create size groups
        def group_size(size):
            if pd.isna(size):
                return 'Unknown'
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
        
        snak_king_data['Size_Group'] = snak_king_data['SIZE'].apply(group_size)
        
        size_analysis = snak_king_data.groupby('Size_Group').agg({
            'Dollars': 'sum',
            'Dollars, Yago': 'sum',
            'Units': 'sum'
        }).reset_index()
        
        size_analysis['Revenue_M'] = size_analysis['Dollars'] / 1_000_000
        size_analysis['YoY_%'] = ((size_analysis['Dollars'] - size_analysis['Dollars, Yago']) / 
                                  size_analysis['Dollars, Yago'] * 100)
        
        # Sort by size order
        size_order = ['<2 oz', '2-5 oz', '5-8 oz', '8-12 oz', '12+ oz', 'Unknown']
        size_analysis['Size_Order'] = size_analysis['Size_Group'].apply(
            lambda x: size_order.index(x) if x in size_order else 999
        )
        size_analysis = size_analysis.sort_values('Size_Order')
        
        print("\nBy Package Size:")
        print(size_analysis[['Size_Group', 'Revenue_M', 'YoY_%']].to_string(index=False))
        
        fig5 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Revenue by Size', 'Growth by Size'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig5.add_trace(
            go.Bar(x=size_analysis['Size_Group'], y=size_analysis['Revenue_M'], 
                   marker_color='#4ECDC4', name='Revenue'),
            row=1, col=1
        )
        
        colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in size_analysis['YoY_%']]
        fig5.add_trace(
            go.Bar(x=size_analysis['Size_Group'], y=size_analysis['YoY_%'], 
                   marker_color=colors, name='Growth'),
            row=1, col=2
        )
        
        fig5.update_xaxes(title_text="Package Size", row=1, col=1)
        fig5.update_xaxes(title_text="Package Size", row=1, col=2)
        fig5.update_yaxes(title_text="Revenue ($M)", row=1, col=1)
        fig5.update_yaxes(title_text="YoY Growth (%)", row=1, col=2)
        
        fig5.update_layout(
            title_text='Snak King Package Size Analysis',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        fig5.write_html('/mnt/user-data/outputs/snak_king_size_analysis.html')
        print(f"\n✅ Saved: snak_king_size_analysis.html")
    
    # Summary Statistics
    print("\n" + "="*80)
    print("SNAK KING SUMMARY STATISTICS")
    print("="*80)
    
    total_revenue = snak_king_data['Dollars'].sum() / 1_000_000
    total_revenue_yago = snak_king_data['Dollars, Yago'].sum() / 1_000_000
    total_growth = ((total_revenue - total_revenue_yago) / total_revenue_yago * 100)
    avg_velocity = snak_king_data['Dollars/TDP'].mean() / 1000
    total_tdp = snak_king_data['TDP'].sum()
    
    print(f"\nTotal Revenue: ${total_revenue:.2f}M")
    print(f"Previous Year: ${total_revenue_yago:.2f}M")
    print(f"YoY Growth: {total_growth:+.1f}%")
    print(f"Average Velocity: ${avg_velocity:.1f}k/TDP")
    print(f"Total Distribution Points: {total_tdp:,.0f}")
    print(f"Number of Products: {len(snak_king_data)}")
    print(f"Categories Present: {snak_king_data['Subcategory'].nunique()}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files saved to: /mnt/user-data/outputs/")
print("\nFiles created:")
if len(snak_king_data) == 0:
    print("  - top_brands_comparison.html")
else:
    print("  - snak_king_revenue_by_category.html")
    print("  - snak_king_growth_by_category.html")
    print("  - snak_king_performance_matrix.html")
    if 'FLAVOR' in snak_king_data.columns:
        print("  - snak_king_flavor_breakdown.html")
    if 'SIZE' in snak_king_data.columns:
        print("  - snak_king_size_analysis.html")
