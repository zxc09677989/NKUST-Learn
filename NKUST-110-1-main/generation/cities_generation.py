import folium
import pandas as pd
from folium.plugins import HeatMap


def html_generation_function(city_name):
    cdata = pd.read_csv('../data/NPA_TD1.csv')
    city_data = cdata.loc[cdata['CityName'] == city_name]
    incidents = folium.map.FeatureGroup()
    latitudes = list(city_data['Latitude'])
    longitudes = list(city_data['Longitude'])
    labels = list(city_data['Address'])
    city_latitude = latitudes[0]
    city_longitude = longitudes[0]
    city_map = folium.Map(location=[city_latitude, city_longitude], zoom_start=9)
    for lat, lng, label in zip(latitudes, longitudes, labels):  # 標籤圖
        iframe = folium.IFrame(f'{label}', height=50, width=len(f'{label}') * 15)
        popup = folium.Popup(iframe, max_width=len(f'{label}') * 15)
        folium.Marker([lat, lng], popup=popup).add_to(city_map)

    heat_data = city_data[['Latitude', 'Longitude']].values.tolist()  # 熱度圖
    HeatMap(heat_data).add_to(city_map)
    city_map.add_child(incidents)
    city_map.save(f'../html/{city_name}.html')


def main():
    cities = ['臺北市', '新北市', '桃園市', '臺中市', '臺南市', '高雄市',
              '新竹縣', '苗栗縣', '彰化縣', '南投縣', '雲林縣', '嘉義縣',
              '屏東縣', '宜蘭縣', '花蓮縣', '臺東縣', '澎湖縣', '金門縣',
              '連江縣', '基隆市', '新竹市', '嘉義市']
    for city in cities:
        html_generation_function(city)


if __name__ == '__main__':
    main()
