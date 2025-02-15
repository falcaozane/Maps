from flask import Flask, request, jsonify
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

app = Flask(__name__)
CORS(app)

# Load and prepare the store data
stores_df = pd.read_csv('dataset of 50 stores.csv')

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
        
        return store_map.get_root().render()
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/stores/route', methods=['GET'])
def get_store_route():
    """Get route to a specific store with visualization"""
    try:
        user_lat = float(request.args.get('user_lat'))
        user_lon = float(request.args.get('user_lon'))
        store_lat = float(request.args.get('store_lat'))
        store_lon = float(request.args.get('store_lon'))
        visualization_type = request.args.get('viz_type', 'simple')  # simple or animated
        
        # Initialize graph if not already initialized
        if store_locator.network_graph is None:
            store_locator.initialize_graph((user_lat, user_lon))
        
        # Get nearest nodes
        start_node = ox.distance.nearest_nodes(
            store_locator.network_graph, user_lon, user_lat)
        end_node = ox.distance.nearest_nodes(
            store_locator.network_graph, store_lon, store_lat)
        
        # Calculate both time and distance based paths
        try:
            path_time = nx.shortest_path(
                store_locator.network_graph, 
                start_node, 
                end_node, 
                weight='travel_time'
            )
            
            path_length = nx.shortest_path(
                store_locator.network_graph, 
                start_node, 
                end_node, 
                weight='length'
            )
            
            # Create map
            m = folium.Map(
                location=[user_lat, user_lon],
                zoom_start=13,
                tiles="cartodbpositron"
            )
            
            # Add time-based route (blue)
            route_coords_time = [
                (store_locator.network_graph.nodes[node]['y'],
                 store_locator.network_graph.nodes[node]['x']) 
                for node in path_time
            ]
            
            folium.PolyLine(
                locations=route_coords_time,
                color='blue',
                weight=3,
                opacity=0.8,
                popup='Fastest Route'
            ).add_to(m)
            
            # Add distance-based route (red)
            route_coords_length = [
                (store_locator.network_graph.nodes[node]['y'],
                 store_locator.network_graph.nodes[node]['x']) 
                for node in path_length
            ]
            
            folium.PolyLine(
                locations=route_coords_length,
                color='red',
                weight=3,
                opacity=0.8,
                popup='Shortest Route'
            ).add_to(m)
            
            # Add markers for start and end points
            folium.Marker(
                [user_lat, user_lon],
                popup='Your Location',
                icon=folium.Icon(color='green', icon='home')
            ).add_to(m)
            
            folium.Marker(
                [store_lat, store_lon],
                popup='Store Location',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add layer control and fullscreen option
            for tile, attribution in [
                ("cartodbpositron", "© OpenStreetMap contributors, © CartoDB"),
                ("openstreetmap", "© OpenStreetMap contributors"),
                ("Stamen Terrain", "Map tiles by Stamen Design, under CC BY 3.0"),
                ("Stamen Water Color", "Map tiles by Stamen Design, under CC BY 3.0"),
                ("Stamen Toner", "Map tiles by Stamen Design, under CC BY 3.0"),
                ("cartodbdark_matter", "© OpenStreetMap contributors, © CartoDB")
            ]:
                folium.TileLayer(tile, attr=attribution).add_to(m)
            
            folium.LayerControl(position='bottomright').add_to(m)
            plugins.Fullscreen().add_to(m)
            
            if visualization_type == 'animated':
                # Create animated visualization using plotly
                df = create_route_animation_data(
                    store_locator.network_graph, 
                    path_time, 
                    path_length
                )
                
                return jsonify({
                    'status': 'success',
                    'static_map': m.get_root().render(),
                    'animation_data': df.to_dict('records')
                })
            else:
                return m.get_root().render()
                
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
                
            # Create detailed popup content
            popup_content = f"""
            <div style='width: 200px'>
                <h4 style='color: {color}'>{store['store_name']}</h4>
                <b>Address:</b> {store['address']}<br>
                <b>Distance:</b> {store['distance']} km<br>
                <b>Est. Delivery:</b> {store['estimated_delivery_time']} mins<br>
                <b>Contact:</b> {store['contact']}<br>
                <b>Categories:</b> {store['product_categories']}<br>
                <a href='/api/stores/route?user_lat={lat}&user_lon={lon}&store_lat={store['location']['lat']}&store_lon={store['location']['lon']}' target='_blank'>Get Route</a>
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
                radius=store['distance'] * 100,  # Visual radius for the circle
                color=color,
                fill=True,
                opacity=0.1
            ).add_to(m)
        
        # Add distance circles from user location
        folium.Circle(
            location=[lat, lon],
            radius=2000,  # 2km radius
            color='red',
            fill=False,
            weight=1,
            dash_array='5, 5'
        ).add_to(m)
        
        folium.Circle(
            location=[lat, lon],
            radius=5000,  # 5km radius
            color='orange',
            fill=False,
            weight=1,
            dash_array='5, 5'
        ).add_to(m)
        
        folium.Circle(
            location=[lat, lon],
            radius=radius * 1000,  # User-specified radius
            color='blue',
            fill=False,
            weight=1,
            dash_array='5, 5'
        ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: 130px; 
                    border:2px solid grey; z-index:9999; background-color:white;
                    padding: 10px;
                    font-size: 14px;
                    border-radius: 5px;">
            <p style="margin-bottom: 5px"><b>Distance Zones</b></p>
            <p style="margin: 5px">
                <i class="fa fa-circle" style="color:red"></i> &lt; 2 km
            </p>
            <p style="margin: 5px">
                <i class="fa fa-circle" style="color:orange"></i> 2-5 km
            </p>
            <p style="margin: 5px">
                <i class="fa fa-circle" style="color:blue"></i> &gt; 5 km
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer controls and fullscreen option
        folium.LayerControl(position='bottomleft').add_to(m)
        plugins.Fullscreen(
            position='topright',
            title='Expand map',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(m)
        
        # Add search radius info
        radius_info = f'''
        <div style="position: fixed; 
                    top: 50px; left: 50px; 
                    border:2px solid grey; z-index:9999; background-color:white;
                    padding: 10px;
                    font-size: 14px;
                    border-radius: 5px;">
            <b>Search Radius:</b> {radius} km<br>
            <b>Stores Found:</b> {len(nearby_stores)}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(radius_info))
        
        return m.get_root().render()
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
