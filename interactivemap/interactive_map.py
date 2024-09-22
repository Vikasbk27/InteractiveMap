import os
import requests
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import math
import pandas as pd

# Load the map image
map_image = cv2.imread('data/world-political-map-2020.jpg')

# Load country data from GeoJSON
gdf = gpd.read_file('data/countries.geo.json')

# Path to the India map image
india_map_image_path = 'data/india-map-2019.jpg'


# Path to the population data CSV
population_csv_path = 'data/world_population.csv'


# Function to convert latitude/longitude to image coordinates
def latlon_to_image_coords(lat, lon, img_width, img_height):
    x = int((lon + 180) * (img_width / 360))
    y = int((90 - lat) * (img_height / 180))
    return (x, y)


# Function to calculate the great-circle distance using the Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon1 - lon2)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


# Function to fetch current weather data for a given latitude and longitude
def fetch_weather_data(lat, lon):
    api_key = '18b83575f644151d64d8b8b07f2957d9'  # Your actual API key
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data['weather'][0]['description'].capitalize()
        temperature = data['main']['temp']
        return f"{weather_description}, {temperature}°C"
    else:
        return "Weather data unavailable"


# Load population and area data from CSV
population_df = pd.read_csv(population_csv_path, encoding='utf-8')
# Select relevant columns
columns_to_select = [
    'Country/Territory', 'Capital', 'Continent', '2022 Population', '2020 Population',
    '2015 Population', '2010 Population', '2000 Population', '1990 Population',
    '1980 Population', '1970 Population', 'Area (km²)', 'Density (per km²)',
    'Growth Rate', 'World Population Percentage'
]
population_df = population_df[columns_to_select]
# Rename columns for consistency
population_df.columns = [
    'country_name', 'capital', 'continent', 'population_2022', 'population_2020',
    'population_2015', 'population_2010', 'population_2000', 'population_1990',
    'population_1980', 'population_1970', 'area', 'density',
    'growth_rate', 'world_population_percentage'
]

# Clean data to remove special characters or handle encoding issues
population_df = population_df.applymap(lambda x: x.replace('�', '') if isinstance(x, str) else x)

# Merge population and area data with GeoDataFrame
gdf = gdf.merge(population_df, left_on='name', right_on='country_name', how='left')

# Prepare country data with image coordinates, polygons, and additional information
country_data = {}
img_height, img_width, _ = map_image.shape

for idx, row in gdf.iterrows():
    country_name = row['name']
    centroid = row['geometry'].centroid
    coords = latlon_to_image_coords(centroid.y, centroid.x, img_width, img_height)
    geometry = row['geometry']
    polygons = []
    if isinstance(geometry, Polygon):
        polygons.append(geometry)
    elif isinstance(geometry, MultiPolygon):
        polygons.extend(geometry.geoms)
    polygon_coords_list = []
    for polygon in polygons:
        polygon_coords = [latlon_to_image_coords(lat, lon, img_width, img_height) for lon, lat in
                          polygon.exterior.coords]
        polygon_coords_list.append(polygon_coords)
    country_data[country_name] = {
        'coords': coords, 'centroid': centroid, 'weather': None,
        'capital': row['capital'], 'continent': row['continent'],
        'population_2022': row['population_2022'], 'population_2020': row['population_2020'],
        'population_2015': row['population_2015'], 'population_2010': row['population_2010'],
        'population_2000': row['population_2000'], 'population_1990': row['population_1990'],
        'population_1980': row['population_1980'], 'population_1970': row['population_1970'],
        'area': row['area'], 'density': row['density'],
        'growth_rate': row['growth_rate'], 'world_population_percentage': row['world_population_percentage'],
        'polygons': polygon_coords_list
    }

selected_countries = []
highlight_color = (0, 255, 0)
original_map_image = map_image.copy()


def get_country_by_coords(x, y):
    for country, data in country_data.items():
        cx, cy = data['coords']
        if abs(x - cx) < 20 and abs(y - cy) < 20:
            return country
    return None


def update_country_data(country):
    lat = country_data[country]['centroid'].y
    lon = country_data[country]['centroid'].x
    country_data[country]['weather'] = fetch_weather_data(lat, lon)


def highlight_country(map_img, country):
    if country in country_data:
        for polygon in country_data[country]['polygons']:
            cv2.polylines(map_img, [np.array(polygon, dtype=np.int32)], isClosed=True, color=highlight_color,
                          thickness=2)


def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def display_country_info(country):
    info_image = np.zeros((600, 800, 3), dtype=np.uint8)  # Increased height to 600
    info_image[:] = (255, 255, 255)  # Set background to white
    weather = country_data[country]['weather']
    capital = country_data[country]['capital']
    continent = country_data[country]['continent']
    population_2022 = country_data[country]['population_2022']
    population_2020 = country_data[country]['population_2020']
    population_2015 = country_data[country]['population_2015']
    population_2010 = country_data[country]['population_2010']
    population_2000 = country_data[country]['population_2000']
    population_1990 = country_data[country]['population_1990']
    population_1980 = country_data[country]['population_1980']
    population_1970 = country_data[country]['population_1970']
    area = country_data[country]['area']
    density = country_data[country]['density']
    growth_rate = country_data[country]['growth_rate']
    world_population_percentage = country_data[country]['world_population_percentage']

    cv2.putText(info_image, f"Country: {country}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info_image, f"Capital: {capital}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(info_image, f"Continent: {continent}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(info_image, f"Weather: {weather}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(info_image, f"Population (2022): {population_2022}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.putText(info_image, f"Population (2020): {population_2020}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.putText(info_image, f"Population (2015): {population_2015}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.putText(info_image, f"Population (2010): {population_2010}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.putText(info_image, f"Population (2000): {population_2000}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.putText(info_image, f"Population (1990): {population_1990}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.putText(info_image, f"Population (1980): {population_1980}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.putText(info_image, f"Population (1970): {population_1970}", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.putText(info_image, f"Area: {area} km²", (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(info_image, f"Density: {density} per km²", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(info_image, f"Growth Rate: {growth_rate}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(info_image, f"World Population %: {world_population_percentage}", (10, 490), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)

    cv2.imshow('Country Information', info_image)


def mouse_callback(event, x, y, flags, param):
    global map_image, text_image, display_image
    if event == cv2.EVENT_MOUSEMOVE:
        map_copy = original_map_image.copy()
        country = get_country_by_coords(x, y)
        if country:
            highlight_country(map_copy, country)
            cv2.putText(map_copy, country, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Interactive Map', map_copy)
        else:
            cv2.imshow('Interactive Map', display_image)
    elif event == cv2.EVENT_LBUTTONDOWN:
        country = get_country_by_coords(x, y)
        if country:
            update_country_data(country)
            selected_countries.append(country)
            if len(selected_countries) == 2:
                country1 = selected_countries[0]
                country2 = selected_countries[1]
                centroid1 = country_data[country1]['centroid']
                centroid2 = country_data[country2]['centroid']
                distance = haversine_distance(centroid1.y, centroid1.x, centroid2.y, centroid2.x)
                display_image = original_map_image.copy()
                cv2.putText(display_image, f"Flight distance between {country1} and {country2} is {distance:.2f} km", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Interactive Map', display_image)
                selected_countries.clear()
            else:
                display_country_info(country)
    elif event == cv2.EVENT_RBUTTONDOWN:
        country = get_country_by_coords(x, y)
        if country == "India":
            india_map = cv2.imread(india_map_image_path)
            if india_map is not None:
                india_map_resized = resize_image(india_map, height=600)  # Resize to fit within a window of height 600 pixels
                cv2.imshow('India Map', india_map_resized)

display_image = original_map_image.copy()
cv2.imshow('Interactive Map', display_image)
cv2.setMouseCallback('Interactive Map', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
