Based on your uploaded dataset and deployed API, here are example API calls for your Kotlin client:

1. **Find Nearby Stores (JSON Response)**
```
https://maps-yiv5.onrender.com/api/stores/nearby?lat=18.9695&lon=72.8320&radius=1
```
Use this to get store details near Market Road area within 1km

2. **View Basic Store Map**
```
https://maps-yiv5.onrender.com/api/stores/map?lat=18.9701&lon=72.8330&radius=0.5
```
Shows map centered at Main Street with 500m radius

3. **View All Store Locations with Color Coding**
```
https://maps-yiv5.onrender.com/api/stores/locations?lat=18.9685&lon=72.8325&radius=2
```
Shows detailed map with color-coded stores within 2km

4. **Get Route Between Points**
Example routes:
```
# Route from Park Avenue to Hill Road stores
https://maps-yiv5.onrender.com/api/stores/route?user_lat=18.9710&user_lon=72.8335&store_lat=18.9705&store_lon=72.8345&viz_type=simple

# Route from Main Street to Market Road stores
https://maps-yiv5.onrender.com/api/stores/route?user_lat=18.9701&user_lon=72.8330&store_lat=18.9695&store_lon=72.8320&viz_type=simple
```

Key Location Points in Dataset:
- Main Street Area: 18.9701, 72.8330
- Park Avenue: 18.9710, 72.8335
- Market Road: 18.9695, 72.8320
- Shopping Center: 18.9670, 72.8300
- Commercial Street: 18.9690, 72.8340