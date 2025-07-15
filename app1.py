# Save this as app1.py and run with: streamlit run app1.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Antenna Pattern Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI and zoom functionality
st.markdown("""
<style>
    /* Enable page zoom */
    html {
        zoom: 1;
    }
    
    /* Zoom controls styling */
    .zoom-control-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        z-index: 1000;
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
    .step-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Make plots more interactive */
    .js-plotly-plot {
        cursor: move;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with zoom controls and plot instructions
with st.sidebar:
    st.markdown("### 🔍 Page Zoom Controls")
    
    # Page zoom slider
    zoom_level = st.slider(
        "Page Zoom Level",
        min_value=50,
        max_value=200,
        value=100,
        step=10,
        format="%d%%"
    )
    
    # Apply zoom with JavaScript
    st.markdown(f"""
    <script>
        document.body.style.zoom = "{zoom_level}%";
    </script>
    """, unsafe_allow_html=True)
    
    # Reset zoom button
    if st.button("Reset Zoom to 100%"):
        st.markdown("""
        <script>
            document.body.style.zoom = "100%";
        </script>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Plot interaction instructions
    st.markdown("### 📊 Plot Controls")
    st.info("""
    **Interactive Features:**
    - 🔍 **Zoom**: Scroll or box select
    - 🤚 **Pan**: Click and drag
    - 🏠 **Reset**: Double-click
    - 📸 **Save**: Camera icon
    - ✏️ **Draw**: Use toolbar
    """)
    
    st.markdown("---")
    
    # Keyboard shortcuts
    st.markdown("### ⌨️ Shortcuts")
    st.code("""
    Ctrl/Cmd + : Zoom in
    Ctrl/Cmd - : Zoom out
    Ctrl/Cmd 0 : Reset zoom
    """)

# Helper function to create download buttons for plots
def create_plot_downloads(fig, filename_prefix):
    """Create download buttons for multiple formats"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # PNG download (high resolution)
        img_bytes = pio.to_image(fig, format='png', width=1920, height=1080, scale=2)
        st.download_button(
            label="📥 PNG",
            data=img_bytes,
            file_name=f"{filename_prefix}.png",
            mime="image/png"
        )
    
    with col2:
        # SVG download (vector format)
        img_svg = pio.to_image(fig, format='svg')
        st.download_button(
            label="📥 SVG",
            data=img_svg,
            file_name=f"{filename_prefix}.svg",
            mime="image/svg+xml"
        )
    
    with col3:
        # PDF download
        try:
            img_pdf = pio.to_image(fig, format='pdf')
            st.download_button(
                label="📥 PDF",
                data=img_pdf,
                file_name=f"{filename_prefix}.pdf",
                mime="application/pdf"
            )
        except:
            st.caption("PDF export needs kaleido")
    
    with col4:
        # Interactive HTML
        html_str = pio.to_html(fig, include_plotlyjs='cdn')
        st.download_button(
            label="📥 HTML",
            data=html_str,
            file_name=f"{filename_prefix}.html",
            mime="text/html"
        )

# Enhanced plotly config for better interactivity
plotly_config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath',
        'drawclosedpath',
        'drawcircle',
        'drawrect',
        'eraseshape'
    ],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'antenna_pattern',
        'height': 1080,
        'width': 1920,
        'scale': 2
    }
}
# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Header
st.markdown("<h1 style='text-align: center; color: #2E86AB;'>📡 Antenna Radiation Pattern Analyzer from WATS 2002 dat File</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload → Process → Visualize</p>", unsafe_allow_html=True)
st.markdown("---")

# Progress bar
progress = st.session_state.step / 3
st.progress(progress)

# Main container
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Step 1: File Upload
    if st.session_state.step >= 0:
        st.markdown("<div class='step-header'><h2>Step 1: Upload Data File 📁</h2></div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a .dat file containing antenna measurement data",
            type=['dat'],
            help="Binary file with float32 data"
        )
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size} bytes"
            }
            st.json(file_details)
            
            if st.button("✅ Confirm Upload", key="upload_btn"):
                st.session_state.data = uploaded_file.read()
                st.session_state.step = 1
                st.rerun()
    
    # Step 2: Process Data
    if st.session_state.step >= 1:
        st.markdown("---")
        st.markdown("<div class='step-header'><h2>Step 2: Process Data 🔄</h2></div>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info("📊 Data will be converted using:")
            st.code("""
• X values: 0.9° to 360° (0.9° steps)
• Z = 127.00417 × Y + 348.78785
• M = Normalized values (dB)
            """)
        
        with col_b:
            if st.button("🚀 Process Data", key="process_btn"):
                with st.spinner("Processing..."):
                    # Process the data
                    data = np.frombuffer(st.session_state.data, dtype=np.float32).copy()
                    data_reshaped = data.reshape(-1, 2)
                    
                    # Create DataFrame
                    df = pd.DataFrame(data_reshaped, columns=["X", "Y"])
                    
                    # Replace X values
                    new_x_values = np.arange(0.9, 360.0 + 0.9, 0.9, dtype=np.float32)
                    limit = min(len(new_x_values), len(df))
                    df.loc[:limit-1, 'X'] = new_x_values[:limit]
                    
                    # Add Z column (dBm)
                    a = 127.00416816938
                    b = 348.787854613837
                    df['Z'] = np.round(a * df['Y'] + b, 1)
                    
                    # Normalize (M column)
                    max_value = df['Z'].max()
                    df['M'] = df['Z'] - max_value
                    
                    st.session_state.df = df
                    st.session_state.processed = True
                    st.session_state.step = 2
                    
                st.success("✅ Data processed successfully!")
                st.rerun()
    
    # Step 3: View Converted Data
    if st.session_state.step >= 2:
        st.markdown("---")
        st.markdown("<div class='step-header'><h2>Step 3: View Converted Data 📋</h2></div>", unsafe_allow_html=True)
        
        # Display statistics
        col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)
        
        df = st.session_state.df
        
        with col1_stats:
            st.metric("Max Level", f"{df['Z'].max():.1f} dBm")
        with col2_stats:
            st.metric("Min Level", f"{df['Z'].min():.1f} dBm")
        with col3_stats:
            st.metric("Normalized Max", f"{df['M'].max():.1f} dB")
        with col4_stats:
            st.metric("Dynamic Range", f"{df['M'].max() - df['M'].min():.1f} dB")
        
        # Show data preview with tabs
        tab1, tab2 = st.tabs(["📊 Data Preview", "📥 Download Options"])
        
        with tab1:
            # Pagination for data display
            rows_per_page = st.select_slider("Rows per page", options=[10, 25, 50, 100, 400], value=25)
            total_pages = int(np.ceil(len(df) / rows_per_page))
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(df))
            
            st.dataframe(
                df[start_idx:end_idx].style.format({
                    'X': '{:.1f}°',
                    'Y': '{:.4f}',
                    'Z': '{:.1f} dBm',
                    'M': '{:.1f} dB'
                }),
                use_container_width=True,
                height=400
            )
        
        with tab2:
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                # Download as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="antenna_pattern_converted.csv",
                    mime="text/csv"
                )
            
            with col_dl2:
                # Download as Excel
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Antenna_Data', index=False)
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_data,
                        file_name="antenna_pattern_converted.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except:
                    st.info("Excel export requires xlsxwriter. Install with: pip install xlsxwriter")
        
        if st.button("📈 Continue to Visualization", key="viz_btn"):
            st.session_state.step = 3
            st.rerun()
    # Step 4: Interactive Plots
    if st.session_state.step >= 3:
        st.markdown("---")
        st.markdown("<div class='step-header'><h2>Step 4: Interactive Visualization 📈</h2></div>", unsafe_allow_html=True)
        
        # Plot customization options
        with st.expander("🎨 Customize Plot Appearance"):
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                plot_color = st.color_picker("Plot Color", "#FF0000")
            with col_opt2:
                line_width = st.slider("Line Width", 1, 5, 2)
            with col_opt3:
                show_fill = st.checkbox("Show Fill", True)
        
        # Create tabs for different plot types
        plot_tab1, plot_tab2, plot_tab3, plot_tab4 = st.tabs([
            "📊 2D Plots", "🌐 3D Visualization", "🎯 Pattern Analysis", "📐 Measurements"
        ])
        
        df = st.session_state.df
        
        with plot_tab1:
            # 2D Plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cartesian View (Original)', 'Cartesian View (Normalized)',
                               'Polar View (Original)', 'Polar View (Normalized)'),
                specs=[[{'type': 'xy'}, {'type': 'xy'}],
                       [{'type': 'polar'}, {'type': 'polar'}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.2
            )
            
            # Cartesian plots
            fig.add_trace(
                go.Scatter(
                    x=df['X'], 
                    y=df['Z'], 
                    mode='lines',
                    line=dict(color=plot_color, width=line_width),
                    name='Original (dBm)',
                    hovertemplate='Angle: %{x:.1f}°<br>Level: %{y:.1f} dBm<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['X'], 
                    y=df['M'], 
                    mode='lines',
                    line=dict(color=plot_color, width=line_width),
                    name='Normalized (dB)',
                    hovertemplate='Angle: %{x:.1f}°<br>Level: %{y:.1f} dB<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add -3dB line to normalized plot
            fig.add_hline(
                y=-3, 
                line_dash="dash", 
                line_color="green",
                annotation_text="-3dB", 
                annotation_position="right",
                row=1, col=2
            )
            
            # Polar plots
            fig.add_trace(
                go.Scatterpolar(
                    r=df['Z'], 
                    theta=df['X'], 
                    mode='lines',
                    line=dict(color=plot_color, width=line_width),
                    fill='toself' if show_fill else None,
                    fillcolor=f'rgba(255,0,0,0.2)' if show_fill else None,
                    name='Original Polar',
                    hovertemplate='Angle: %{theta}°<br>Level: %{r:.1f} dBm<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatterpolar(
                    r=df['M'], 
                    theta=df['X'], 
                    mode='lines',
                    line=dict(color=plot_color, width=line_width),
                    fill='toself' if show_fill else None,
                    fillcolor=f'rgba(255,0,0,0.2)' if show_fill else None,
                    name='Normalized Polar',
                    hovertemplate='Angle: %{theta}°<br>Level: %{r:.1f} dB<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Update Cartesian layouts
            fig.update_xaxes(
                title_text="Angle (degrees)", 
                range=[0, 360],
                dtick=45, 
                showgrid=True, 
                gridwidth=1, 
                gridcolor='LightGray',
                row=1, col=1
            )
            fig.update_xaxes(
                title_text="Angle (degrees)", 
                range=[0, 360],
                dtick=45, 
                showgrid=True, 
                gridwidth=1, 
                gridcolor='LightGray',
                row=1, col=2
            )
            fig.update_yaxes(
                title_text="Level (dBm)", 
                showgrid=True, 
                gridwidth=1, 
                gridcolor='LightGray',
                row=1, col=1
            )
            fig.update_yaxes(
                title_text="Level (dB)", 
                range=[-25, 5],
                showgrid=True, 
                gridwidth=1, 
                gridcolor='LightGray',
                row=1, col=2
            )
            
            # Update polar layouts
            fig.update_polars(
                radialaxis=dict(
                    range=[df['Z'].min()-5, df['Z'].max()+5],
                    dtick=5,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90,
                    dtick=45,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                row=2, col=1
            )
            
            fig.update_polars(
                radialaxis=dict(
                    range=[-25, 5],
                    dtick=5,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90,
                    dtick=45,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800, 
                showlegend=False,
                title_text="Interactive Antenna Radiation Pattern Analysis",
                dragmode='zoom'
            )
            
            # Display plot with enhanced config
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
            
            # Download buttons for this plot
            st.markdown("**Download Plot:**")
            create_plot_downloads(fig, "antenna_pattern_2d")
        with plot_tab2:
            # 3D Visualization
            st.subheader("3D Radiation Pattern")
            
            # Create a mesh for 3D visualization
            angles_3d = np.linspace(0, 360, 361)
            radii_3d = np.linspace(0, 1, 50)
            angles_mesh, radii_mesh = np.meshgrid(angles_3d, radii_3d)
            
            # Interpolate pattern values
            try:
                from scipy.interpolate import interp1d
                interp_func = interp1d(df['X'], df['M'], kind='linear', fill_value='extrapolate')
                pattern_values = interp_func(angles_3d)
                
                # Create Z values for 3D surface
                z_mesh = np.zeros_like(angles_mesh)
                for i, angle in enumerate(angles_3d):
                    z_mesh[:, i] = pattern_values[i] * radii_mesh[:, i]
                
                # Convert to Cartesian coordinates
                x_mesh = radii_mesh * np.cos(np.deg2rad(angles_mesh))
                y_mesh = radii_mesh * np.sin(np.deg2rad(angles_mesh))
                
                # Create 3D surface plot
                fig_3d = go.Figure(data=[
                    go.Surface(
                        x=x_mesh,
                        y=y_mesh,
                        z=z_mesh,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Level (dB)"),
                        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Level: %{z:.1f} dB<extra></extra>'
                    )
                ])
                
                fig_3d.update_layout(
                    title="3D Radiation Pattern Visualization",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Level (dB)",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                        aspectmode='cube'
                    ),
                    height=700,
                    dragmode='orbit'
                )
                
                st.plotly_chart(fig_3d, use_container_width=True, config=plotly_config)
                
                # Download buttons for 3D plot
                st.markdown("**Download 3D Plot:**")
                create_plot_downloads(fig_3d, "antenna_pattern_3d")
                
            except ImportError:
                st.error("3D visualization requires scipy. Install with: pip install scipy")
        
        with plot_tab3:
            # Pattern Analysis
            st.subheader("Pattern Analysis")
            
            # Find -3dB beamwidth
            above_3db = df[df['M'] >= -3]
            if len(above_3db) > 0:
                st.write("**-3dB Beamwidth Analysis:**")
                
                # Find continuous regions above -3dB
                beamwidth_angles = []
                diff = above_3db.index.to_series().diff()
                groups = (diff != 1).cumsum()
                
                for _, group in above_3db.groupby(groups):
                    beamwidth_angles.append({
                        'start': group['X'].iloc[0],
                        'end': group['X'].iloc[-1],
                        'width': group['X'].iloc[-1] - group['X'].iloc[0]
                    })
                
                # Display beamwidth info
                for i, bw in enumerate(beamwidth_angles):
                    st.metric(
                        f"Beamwidth Region {i+1}", 
                        f"{bw['width']:.1f}°",
                        f"({bw['start']:.1f}° - {bw['end']:.1f}°)"
                    )
            
            # Front-to-back ratio
            front_value = df.loc[df['X'] == df['X'].min(), 'M'].values[0] if len(df) > 0 else 0
            back_idx = len(df) // 2
            back_value = df.iloc[back_idx]['M'] if back_idx < len(df) else 0
            fb_ratio = front_value - back_value
            
            col1_analysis, col2_analysis = st.columns(2)
            with col1_analysis:
                st.metric("Front Value", f"{front_value:.1f} dB")
            with col2_analysis:
                st.metric("Approximate F/B Ratio", f"{abs(fb_ratio):.1f} dB")
            
            # Create analysis plot
            fig_analysis = go.Figure()
            
            # Add pattern
            fig_analysis.add_trace(go.Scatter(
                x=df['X'],
                y=df['M'],
                mode='lines',
                name='Pattern',
                line=dict(color='blue', width=2)
            ))
            
            # Add -3dB reference line
            fig_analysis.add_hline(
                y=-3,
                line_dash="dash",
                line_color="green",
                annotation_text="-3dB"
            )
            
            # Highlight beamwidth regions
            for i, bw in enumerate(beamwidth_angles):
                fig_analysis.add_vrect(
                    x0=bw['start'],
                    x1=bw['end'],
                    fillcolor="LightGreen",
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    annotation_text=f"BW {i+1}"
                )
            
            fig_analysis.update_layout(
                title="Pattern Analysis with Beamwidth Regions",
                xaxis_title="Angle (degrees)",
                yaxis_title="Normalized Level (dB)",
                height=500,
                xaxis=dict(range=[0, 360], dtick=45),
                yaxis=dict(range=[-25, 5])
            )
            
            st.plotly_chart(fig_analysis, use_container_width=True, config=plotly_config)
            
            # Download analysis plot
            st.markdown("**Download Analysis Plot:**")
            create_plot_downloads(fig_analysis, "antenna_pattern_analysis")
            
            # Statistics table
            st.markdown("---")
            stats_data = {
                'Parameter': ['Max Level', 'Min Level', 'Average Level', 'Dynamic Range'],
                'Original (dBm)': [
                    f"{df['Z'].max():.1f}",
                    f"{df['Z'].min():.1f}", 
                    f"{df['Z'].mean():.1f}",
                    f"{df['Z'].max() - df['Z'].min():.1f}"
                ],
                'Normalized (dB)': [
                    f"{df['M'].max():.1f}",
                    f"{df['M'].min():.1f}", 
                    f"{df['M'].mean():.1f}",
                    f"{df['M'].max() - df['M'].min():.1f}"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        with plot_tab4:
            # Measurements
            st.subheader("📐 Interactive Measurements")
            
            # Angle selector
            selected_angle = st.slider(
                "Select Angle (degrees)", 
                min_value=0.0, 
                max_value=360.0, 
                value=0.0, 
                step=0.9
            )
            
            # Find closest angle in data
            idx = (df['X'] - selected_angle).abs().idxmin()
            actual_angle = df.loc[idx, 'X']
            original_value = df.loc[idx, 'Z']
            normalized_value = df.loc[idx, 'M']
            
            # Display values at selected angle
            col1_meas, col2_meas, col3_meas = st.columns(3)
            with col1_meas:
                st.metric("Angle", f"{actual_angle:.1f}°")
            with col2_meas:
                st.metric("Original Level", f"{original_value:.1f} dBm")
            with col3_meas:
                st.metric("Normalized Level", f"{normalized_value:.1f} dB")
            
            # Interactive plot with marker
            fig_interactive = go.Figure()
            
            # Add the pattern
            fig_interactive.add_trace(go.Scatter(
                x=df['X'], 
                y=df['M'],
                mode='lines',
                name='Pattern',
                line=dict(color='blue', width=2),
                hovertemplate='Angle: %{x:.1f}°<br>Level: %{y:.1f} dB<extra></extra>'
            ))
            
            # Add marker at selected angle
            fig_interactive.add_trace(go.Scatter(
                x=[actual_angle],
                y=[normalized_value],
                mode='markers+text',
                name='Selected Point',
                marker=dict(size=12, color='red'),
                text=[f'{normalized_value:.1f} dB'],
                textposition='top center'
            ))
            
            # Add -3dB reference line
            fig_interactive.add_hline(
                y=-3, 
                line_dash="dash", 
                line_color="green",
                annotation_text="-3dB"
            )
            
            fig_interactive.update_layout(
                title="Interactive Pattern Measurement",
                xaxis_title="Angle (degrees)",
                yaxis_title="Normalized Level (dB)",
                height=500,
                showlegend=False,
                xaxis=dict(range=[0, 360], dtick=45),
                yaxis=dict(range=[-25, 5]),
                dragmode='pan'
            )
            
            st.plotly_chart(fig_interactive, use_container_width=True, config=plotly_config)
            
            # Download interactive plot
            st.markdown("**Download Interactive Plot:**")
            create_plot_downloads(fig_interactive, "antenna_pattern_interactive")
            
            # Comparison angle selector
            st.markdown("---")
            st.markdown("### 📊 Compare Two Angles")
            
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                angle1 = st.number_input("Angle 1 (degrees)", min_value=0.0, max_value=360.0, value=0.0, step=0.9)
            with col_comp2:
                angle2 = st.number_input("Angle 2 (degrees)", min_value=0.0, max_value=360.0, value=180.0, step=0.9)
            
            # Find values at both angles
            idx1 = (df['X'] - angle1).abs().idxmin()
            idx2 = (df['X'] - angle2).abs().idxmin()
            
            val1 = df.loc[idx1, 'M']
            val2 = df.loc[idx2, 'M']
            diff = abs(val1 - val2)
            
            # Display comparison
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric(f"Level at {df.loc[idx1, 'X']:.1f}°", f"{val1:.1f} dB")
            with col_res2:
                st.metric(f"Level at {df.loc[idx2, 'X']:.1f}°", f"{val2:.1f} dB")
            with col_res3:
                st.metric("Difference", f"{diff:.1f} dB")
            
            # Export complete analysis report
            st.markdown("---")
            if st.button("📄 Generate Complete Report"):
                report = f"""
ANTENNA RADIATION PATTERN ANALYSIS REPORT
========================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
------------------
Maximum Level: {df['Z'].max():.1f} dBm (Normalized: {df['M'].max():.1f} dB)
Minimum Level: {df['Z'].min():.1f} dBm (Normalized: {df['M'].min():.1f} dB)
Average Level: {df['Z'].mean():.1f} dBm (Normalized: {df['M'].mean():.1f} dB)
Dynamic Range: {df['Z'].max() - df['Z'].min():.1f} dB

ANGULAR INFORMATION:
-------------------
Maximum at: {df.loc[df['Z'].idxmax(), 'X']:.1f}°
Minimum at: {df.loc[df['Z'].idxmin(), 'X']:.1f}°

BEAMWIDTH ANALYSIS:
------------------
"""
                if len(beamwidth_angles) > 0:
                    for i, bw in enumerate(beamwidth_angles):
                        report += f"Region {i+1}: {bw['width']:.1f}° ({bw['start']:.1f}° - {bw['end']:.1f}°)\n"
                else:
                    report += "No -3dB beamwidth regions found\n"
                
                report += f"""
MEASUREMENT POINTS:
------------------
Selected Angle: {actual_angle:.1f}°
Level at Selected Angle: {original_value:.1f} dBm ({normalized_value:.1f} dB)

COMPARISON:
-----------
Angle 1: {df.loc[idx1, 'X']:.1f}° - Level: {val1:.1f} dB
Angle 2: {df.loc[idx2, 'X']:.1f}° - Level: {val2:.1f} dB
Difference: {diff:.1f} dB

DATA POINTS: {len(df)}
                """
                
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="antenna_pattern_report.txt",
                    mime="text/plain"
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <p style='color: #666; margin: 2px; font-size: 14px;'>📡 Antenna Pattern Analyzer v1.0</p>
        <p style='color: #888; font-size: 13px; margin: 2px;'>Developed by <strong>Md. Nafiul Hasnat</strong></p>
        <p style='color: #999; font-size: 11px; margin: 2px;'>© 2025 All rights reserved | Powered by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)