from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
from folium import plugins
import osmnx as ox
import networkx as nx
from datetime import datetime
import json
import matplotlib.pyplot as plt
import plotly.express as px
import os
import time

app = Flask(__name__)
CORS(app)

# Create temp directory for files
os.makedirs('temp', exist_ok=True)

# Load and prepare the store data
stores_df = pd.read_csv('dataset of 50 stores.csv')

# Cleanup function for temporary files
@app.before_request
def cleanup_temp_files():
    temp_dir = 'temp'
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.isfile(file_path) and file.endswith('.html'):
                    # Delete files older than 1 hour
                    if os.path.getmtime(file_path) < time.time() - 3600:
                        os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up temp files: {e}")

class StoreLocator:
    def __init__(self, stores_dataframe):
        self.stores_df = stores_dataframe
        self.network_graph = None
        
    def initialize_graph(self, center_point, dist=20000):
        """Initialize road network graph for a given center point"""
        try:
            self.network_graph = ox.graph_from_point(center_point, dist=dist, network_type="drive")
            self.network_graph = ox.add_edge_speeds(self.network_graph)
            self.network_graph = ox.add_edge_travel_times(self.network_graph)
            return True
        except Exception as e:
            print(f"Error initializing graph: {str(e)}")
            return False

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate direct distance between two points"""
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers

    def estimate_delivery_time(self, distance, current_time=None):
        """Estimate delivery time based on distance and current time"""
        if current_time is None:
            current_time = datetime.now()

        # Base time: 5 mins base + 2 mins per km
        base_minutes = 5 + (distance * 2)
        
        # Apply traffic multiplier based on time of day
        hour = current_time.hour
        if hour in [8, 9, 10, 17, 18, 19]:  # Peak hours
            multiplier = 1.5
        elif hour in [23, 0, 1, 2, 3, 4]:   # Off-peak hours
            multiplier = 0.8
        else:                                # Normal hours
            multiplier = 1.0
            
        return round(base_minutes * multiplier)

    def find_nearby_stores(self, lat, lon, radius=5):
        """Find stores within specified radius"""
        nearby_stores = []
        
        for _, store in self.stores_df.iterrows():
            distance = self.calculate_distance(lat, lon, 
                                            store['Latitude'], 
                                            store['Longitude'])
            
            if distance <= radius:
                delivery_time = self.estimate_delivery_time(distance)
                nearby_stores.append({
                    'store_name': store['Store Name'],
                    'address': store['Address'],
                    'contact': store['Contact Number'],
                    'distance': round(distance, 2),
                    'estimated_delivery_time': delivery_time,
                    'product_categories': store['Product Categories'],
                    'location': {
                        'lat': store['Latitude'],
                        'lon': store['Longitude']
                    }
                })
        
        return sorted(nearby_stores, key=lambda x: x['distance'])

    def create_store_map(self, center_lat, center_lon, radius=5):
        """Create an interactive map with store locations"""
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=13,
                      tiles="cartodbpositron")
        
        # Add stores to map
        nearby_stores = self.find_nearby_stores(center_lat, center_lon, radius)
        
        for store in nearby_stores:
            # Prepare popup content
            popup_content = f"""
            <div style='width: 200px'>
                <b>{store['store_name']}</b><br>
                Address: {store['address']}<br>
                Distance: {store['distance']} km<br>
                Est. Delivery: {store['estimated_delivery_time']} mins<br>
                Categories: {store['product_categories']}
            </div>
            """
            
            # Add store marker
            folium.Marker(
                location=[store['location']['lat'], store['location']['lon']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add line to show distance from center
            folium.PolyLine(
                locations=[[center_lat, center_lon], 
                          [store['location']['lat'], store['location']['lon']]],
                weight=2,
                color='blue',
                opacity=0.3
            ).add_to(m)
        
        # Add current location marker
        folium.Marker(
            location=[center_lat, center_lon],
            popup='Your Location',
            icon=folium.Icon(color='green', icon='home')
        ).add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m

# Initialize store locator
store_locator = StoreLocator(stores_df)

def create_animated_route(G, path, color, weight=3):
    """Create an animated route visualization"""
    features = []
    timestamps = []
        
    # Convert path nodes to coordinates
    route_coords = [
        (G.nodes[node]['y'], G.nodes[node]['x']) 
        for node in path
    ]
        
    # Create features for each segment of the route
    for i in range(len(route_coords) - 1):
        segment = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [
                        [route_coords[i][1], route_coords[i][0]],
                        [route_coords[i+1][1], route_coords[i+1][0]]
                    ]
                },
                'properties': {
                    'times': [datetime.now().isoformat()],
                    'style': {
                        'color': color,
                        'weight': weight,
                        'opacity': 0.8
                    }
                }
            }
        features.append(segment)
        timestamps.append(datetime.now().isoformat())
        
    return features

def create_route_animation_data(G, path_time, path_length):
    """Create animation data for route visualization"""
    lst_start, lst_end = [], []
    start_x, start_y = [], []
    end_x, end_y = [], []
    lst_length, lst_time = [], []
    
    # Process time-based path
    for a, b in zip(path_time[:-1], path_time[1:]):
        lst_start.append(a)
        lst_end.append(b)
        lst_length.append(round(G.edges[(a,b,0)]['length']))
        lst_time.append(round(G.edges[(a,b,0)]['travel_time']))
        start_x.append(G.nodes[a]['x'])
        start_y.append(G.nodes[a]['y'])
        end_x.append(G.nodes[b]['x'])
        end_y.append(G.nodes[b]['y'])
    
    # Create DataFrame
    df = pd.DataFrame(
        list(zip(lst_start, lst_end, start_x, start_y, end_x, end_y, 
                 lst_length, lst_time)),
        columns=["start", "end", "start_x", "start_y", "end_x", "end_y",
                "length", "travel_time"]
    ).reset_index().rename(columns={"index": "id"})
    
    return df

@app.route('/api/stores/nearby', methods=['GET'])
def get_nearby_stores():
    """Get nearby stores based on user location"""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius = float(request.args.get('radius', 5))
        
        nearby_stores = store_locator.find_nearby_stores(lat, lon, radius)
        
        return jsonify({
            'status': 'success',
            'stores': nearby_stores
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/stores/map', methods=['GET'])
def get_stores_map():
    """Get HTML map with store locations"""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius = float(request.args.get('radius', 5))
        
        store_map = store_locator.create_store_map(lat, lon, radius)
        
        # Create complete HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
            <title>Stores Map</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    width: 100vw;
                    height: 100vh;
                    overflow: hidden;
                }}
                #map {{
                    width: 100%;
                    height: 100%;
                }}
            </style>
        </head>
        <body>
            {store_map.get_root().render()}
            <script>
                window.onload = function() {{
                    setTimeout(function() {{
                        window.dispatchEvent(new Event('resize'));
                    }}, 1000);
                }};
            </script>
        </body>
        </html>
        """
        
        # Save the HTML to a file
        file_path = 'temp/stores_map.html'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Return the file
        return send_file(file_path, mimetype='text/html')
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    
@app.route('/api/stores/route', methods=['GET'])
def get_store_route():
    try:
        user_lat = float(request.args.get('user_lat'))
        user_lon = float(request.args.get('user_lon'))
        store_lat = float(request.args.get('store_lat'))
        store_lon = float(request.args.get('store_lon'))
        
        # Initialize graph if not already initialized
        if store_locator.network_graph is None:
            store_locator.initialize_graph((user_lat, user_lon))
        
        # Get nearest nodes
        start_node = ox.distance.nearest_nodes(
            store_locator.network_graph, user_lon, user_lat)
        end_node = ox.distance.nearest_nodes(
            store_locator.network_graph, store_lon, store_lat)
        
        try:
            # Calculate paths
            path_time = nx.shortest_path(
                store_locator.network_graph, 
                start_node, 
                end_node, 
                weight='travel_time'
            )
            
            # Create animation data
            lst_start, lst_end = [], []
            start_x, start_y = [], []
            end_x, end_y = [], []
            lst_length, lst_time = [], []

            for a, b in zip(path_time[:-1], path_time[1:]):
                lst_start.append(a)
                lst_end.append(b)
                lst_length.append(round(store_locator.network_graph.edges[(a,b,0)]['length']))
                lst_time.append(round(store_locator.network_graph.edges[(a,b,0)]['travel_time']))
                start_x.append(store_locator.network_graph.nodes[a]['x'])
                start_y.append(store_locator.network_graph.nodes[a]['y'])
                end_x.append(store_locator.network_graph.nodes[b]['x'])
                end_y.append(store_locator.network_graph.nodes[b]['y'])

            df = pd.DataFrame(
                list(zip(lst_start, lst_end, start_x, start_y, end_x, end_y, 
                        lst_length, lst_time)),
                columns=["start", "end", "start_x", "start_y",
                        "end_x", "end_y", "length", "travel_time"]
            ).reset_index().rename(columns={"index": "id"})

            # Create animation using plotly
            df_start = df[df["start"] == start_node]
            df_end = df[df["end"] == end_node]

            fig = px.scatter_mapbox(
                data_frame=df, 
                lon="start_x", 
                lat="start_y",
                zoom=15, 
                width=1000, 
                height=800,
                animation_frame="id",
                mapbox_style="carto-positron"
            )

            # Add driver marker
            fig.data[0].marker = {"size": 12}

            # Add start point
            fig.add_trace(
                px.scatter_mapbox(
                    data_frame=df_start, 
                    lon="start_x", 
                    lat="start_y"
                ).data[0]
            )
            fig.data[1].marker = {"size": 15, "color": "red"}

            # Add end point
            fig.add_trace(
                px.scatter_mapbox(
                    data_frame=df_end, 
                    lon="start_x", 
                    lat="start_y"
                ).data[0]
            )
            fig.data[2].marker = {"size": 15, "color": "green"}

            # Add route
            fig.add_trace(
                px.line_mapbox(
                    data_frame=df, 
                    lon="start_x", 
                    lat="start_y"
                ).data[0]
            )

            # Update layout with slower animation settings
            fig.update_layout(
                showlegend=False,
                margin={"r":0,"t":0,"l":0,"b":0},
                autosize=True,
                height=None,
                updatemenus=[{
                    "type": "buttons",
                    "showactive": False,
                    "y": 0,
                    "x": 0,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 1000, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 800}
                                }
                            ]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                }
                            ]
                        }
                    ]
                }],
                sliders=[{
                    "currentvalue": {"prefix": "Step: "},
                    "pad": {"t": 20},
                    "len": 0.9,
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 0.02,
                    "yanchor": "bottom",
                    "steps": [
                        {
                            "args": [
                                [k],
                                {
                                    "frame": {"duration": 1000, "redraw": True},
                                    "transition": {"duration": 500},
                                    "mode": "immediate"
                                }
                            ],
                            "label": str(k),
                            "method": "animate"
                        } 
                        for k in range(len(df))
                    ]
                }]
            )

            # Create complete HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
                <title>Route Map</title>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        width: 100vw;
                        height: 100vh;
                        overflow: hidden;
                    }}
                    #map-container {{
                        width: 100%;
                        height: 100%;
                    }}
                </style>
            </head>
            <body>
                <div id="map-container">
                    {fig.to_html(include_plotlyjs=True, full_html=False)}
                </div>
                <script>
                    window.onload = function() {{
                        setTimeout(function() {{
                            window.dispatchEvent(new Event('resize'));
                        }}, 1000);
                    }};
                </script>
            </body>
            </html>
            """
            
            # Save the HTML to a file
            file_path = 'temp/route_map.html'
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Return the file
            return send_file(file_path, mimetype='text/html')

        except nx.NetworkXNoPath:
            return jsonify({
                'status': 'error',
                'message': 'No route found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    
@app.route('/api/stores/locations', methods=['GET'])
def get_all_store_locations():
    """Get a map showing all stores in the given radius with colors based on distance"""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        radius = float(request.args.get('radius', 10))
        
        # Get nearby stores
        nearby_stores = store_locator.find_nearby_stores(lat, lon, radius)
        
        # Create base map centered on user location
        m = folium.Map(
            location=[lat, lon],
            zoom_start=12,
            tiles="cartodbpositron"
        )
        
        # Add user location marker
        folium.Marker(
            [lat, lon],
            popup='Your Location',
            icon=folium.Icon(color='green', icon='home')
        ).add_to(m)
        
        # Add markers for each store with color coding based on distance
        for store in nearby_stores:
            # Color code based on distance
            if store['distance'] <= 2:
                color = 'red'  # Very close
            elif store['distance'] <= 5:
                color = 'orange'  # Moderate distance
            else:
                color = 'blue'  # Further away
                
            # Create detailed popup content with mobile-friendly styling
            popup_content = f"""
            <div style='width: 200px; font-size: 14px;'>
                <h4 style='color: {color}; margin: 0 0 8px 0;'>{store['store_name']}</h4>
                <b>Address:</b> {store['address']}<br>
                <b>Distance:</b> {store['distance']} km<br>
                <b>Est. Delivery:</b> {store['estimated_delivery_time']} mins<br>
                <b>Contact:</b> {store['contact']}<br>
                <b>Categories:</b> {store['product_categories']}<br>
                <button onclick="window.location.href='/api/stores/route?user_lat={lat}&user_lon={lon}&store_lat={store['location']['lat']}&store_lon={store['location']['lon']}'" 
                        style='margin-top: 8px; padding: 8px; width: 100%; background-color: #007bff; color: white; border: none; border-radius: 4px;'>
                    Get Route
                </button>
            </div>
            """
            
            # Add store marker
            folium.Marker(
                location=[store['location']['lat'], store['location']['lon']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=color, icon='info-sign'),
                tooltip=f"{store['store_name']} ({store['distance']} km)"
            ).add_to(m)
            
            # Add circle to show distance
            folium.Circle(
                location=[store['location']['lat'], store['location']['lon']],
                radius=store['distance'] * 100,
                color=color,
                fill=True,
                opacity=0.1
            ).add_to(m)
        
        # Add distance circles from user location
        for radius, color in [(2000, 'red'), (5000, 'orange'), (radius * 1000, 'blue')]:
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=False,
                weight=1,
                dash_array='5, 5'
            ).add_to(m)
        
        # Create mobile-friendly HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
            <title>Nearby Stores</title>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    width: 100vw;
                    height: 100vh;
                    overflow: hidden;
                }}
                #map {{
                    width: 100%;
                    height: 100%;
                }}
                .legend {{
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.2);
                    font-size: 12px;
                    z-index: 1000;
                }}
                .info-box {{
                    position: fixed;
                    top: 20px;
                    left: 20px;
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.2);
                    font-size: 12px;
                    z-index: 1000;
                }}
            </style>
        </head>
        <body>
            {m.get_root().render()}
            <div class="legend">
                <b>Distance Zones</b><br>
                <span style="color: red;">●</span> &lt; 2 km<br>
                <span style="color: orange;">●</span> 2-5 km<br>
                <span style="color: blue;">●</span> &gt; 5 km
            </div>
            <div class="info-box">
                <b>Search Radius:</b> {radius} km<br>
                <b>Stores Found:</b> {len(nearby_stores)}
            </div>
            <script>
                window.onload = function() {{
                    setTimeout(function() {{
                        window.dispatchEvent(new Event('resize'));
                    }}, 1000);
                }};
            </script>
        </body>
        </html>
        """
        
        # Save and return the file
        file_path = 'temp/locations_map.html'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return send_file(file_path, mimetype='text/html')
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
