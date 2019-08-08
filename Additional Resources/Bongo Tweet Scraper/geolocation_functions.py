import pandas as pd
import geopy
from geopy import geocoders
from tweet_config import get_app_config
from geopy.geocoders import Nominatim
from uszipcode import SearchEngine
from uszipcode import Zipcode
import math
import re

app_config = get_app_config()

def location_2_coordinates(location, geolocator):
    """Uses gelocator to take user location and get latitude and longitude coordinates"""
    _location = geolocator.geocode(location)
    if _location is not None:
        return {"latitude":_location.latitude, "longitude":_location.longitude}
    else:
        return {"latitude": None, "longitude": None}

def coordinates_2_zip(search, zip_fip_mapping, latitude, longitude, radius=5, returns=3):
    """Uses coordinates to get information on zipcode, and address from USPS"""
    result = search.by_coordinates(latitude, longitude, radius=radius, returns=returns)
    i = 0
    if i < returns:
        for zip_dict in result:
            zipcode=float(zip_dict.to_dict()["zipcode"])
            city_state = zip_dict.to_dict()['post_office_city']
            print(city_state)
            city = re.search("(?P<city>.*), (?P<state>.*)", city_state)['city']
            state = re.search("(?P<city>.*), (?P<state>.*)", city_state)['state']
            if zipcode in list(zip_fip_mapping["zip"]):
                county= zip_fip_mapping.loc[zip_fip_mapping["zip"]==zipcode].iloc[0]["county"]
                return {"zipcode": zipcode, "county": county, "city": city, "state": state, "city_state": city+" "+state}
            else:
                if i == returns-1:
                    zipcode = result[0].to_dict()["zipcode"]
                    city_state = zip_dict.to_dict()['post_office_city']
                    city = re.search("(?P<city>.*), (?P<state>.*)", city_state)['city']
                    state = re.search("(?P<city>.*), (?P<state>.*)", city_state)['state']
                    county = None
                    return {"zipcode": zipcode, "county": county, "city": city, "state": state, "city_state": city+" "+state}
            i=i+1

def validate_tweet_location(location_string):
    print("Encoding String...")
    location_string = str(location_string).encode("ascii", errors="ignore").decode()
    if location_string is None:
        location_string = " "
    location_string=location_string.lower().strip()
    pattern = "(?P<city>.*),[ ]*(?P<state>.*)"
    match = re.search(pattern, location_string)
    return match

def get_city_state(match, state_lookup):
    if match["state"] in list(state_lookup["State"]):
        state = state_lookup[state_lookup["State"]==match["state"]].iloc[0]["Abbreviation"]
        return {"city": match["city"], "state": state}
    elif match["state"] in list(state_lookup["Abbreviation"]):
        return {"city": match["city"], "state": match["state"]}
    else: 
        return None

def location_2_everything(search, city, state, zip_fip_mapping):
    result = search.by_city_and_state(city, state)
    if len(result) == 0:
        return {"zipcode": None, "county": None, "latitude": None, "longitude": None}
    else:
        result = result[0].to_dict()
        zipcode = float(result["zipcode"])
        lat = result["lat"]
        lng = result["lng"]
        if zipcode in list(zip_fip_mapping["zip"]):
            county= zip_fip_mapping.loc[zip_fip_mapping["zip"]==zipcode].iloc[0]["county"]
        else:
            county= None
        return {"zipcode": zipcode, "county": county, "latitude": lat, "longitude": lng}