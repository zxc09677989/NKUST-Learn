import branca
import random
import folium
import json

color_scale = branca.colormap.linear.YlOrRd_09.scale(0, 22)  # color scale
taiwan_map = folium.Map(location=[23.0431, 121.539723], zoom_start=7)
taiwan_map.save('../html/台灣.html')
