"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast, Dict, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from PyPDF2 import PdfWriter, PdfReader
import os
from src.react_agent.configuration import Configuration
from pdf2image import convert_from_path
import pytesseract
from langdetect import detect
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Iterator
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import AnyMessage, HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel, Field, validator

# LangChain Community
from langchain_community.document_loaders import NeedleLoader
from langchain_community.retrievers import NeedleRetriever
from datetime import datetime

# from crewai_tools import BaseTool
from amadeus import Client, ResponseError
from typing import Optional
from os import environ
from langchain.tools import BaseTool, Tool
import requests
from pydantic import Field, BaseModel  
import logging
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()
      

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#----------------------------------------------------------- Tools i have created -----------------------------------------------------------
# Tavily Search: Performs web searches using the Tavily search engine, providing accurate and trusted results for general queries.
# Amadeus Flight Search: Searches for flight availability and prices using the Amadeus API.
# Amadeus Hotel Search: Searches for hotel availability and prices using the Amadeus API.
# Geoapify Places Search: Searches for points of interest in a location using the Geoapify Places API.
# Weather Search: Provides weather information for a given location and date using the OpenWeatherMap API.
# Route Finder: Finds the optimal route between locations using the Google Maps API.
# Flight Search: Provides flight information between two locations, including airlines, prices, departure/arrival times, and more.
# Google Scholar Search: Provides academic research results from Google Scholar, including titles, links, snippets, publication info, citations, and versions.
# Booking Scraper: Scrapes hotel data from Booking.com based on destination, check-in/check-out dates, and other parameters.
# Address Validation: Uses Google Maps Address Validation API to validate and refine addresses.
# Google Maps Static Map: Generates a static map image using the Google Maps Static API.
# Google Maps Roads API: Calls the Google Maps Roads API for snap-to-roads, nearest-roads, and speed limits.
# Google Maps Time Zone: Calls the Google Maps Time Zone API to get time zone information for a location.
# Google Maps Places API: Calls the Google Maps Places API for text search and nearby search.
# Google Maps Find Place API: Calls the Google Maps Find Place API to find places by text query.
# Google Maps Place Details API: Calls the Google Maps Place Details API to get detailed information about a place.
# Google Maps Geolocation: Calls the Google Maps Geolocation API to estimate the device's location.
# Google Maps Geocoding: Calls the Google Maps Geocoding API to convert addresses to geographic coordinates.
# Google Maps Elevation: Calls the Google Maps Elevation API to get elevation data for locations.
# Google Maps Distance Matrix: Calls the Google Maps Distance Matrix API to get travel distance and time data.
# Google Maps Directions: Calls the Google Maps Directions API to get travel directions.
# Yelp Business Search: Calls the Yelp Business Search endpoint to find businesses.
# Yelp Phone Search: Calls the Yelp Phone Search endpoint to find businesses by phone number.
# Yelp Business Details: Calls the Yelp Business Details endpoint to get information about a business.
# Yelp Business Reviews: Calls the Yelp Business Reviews endpoint to get reviews for a business.
# Yelp Events Search: Calls the Yelp Events search endpoint to find local events.
# Yelp GraphQL: Calls the Yelp GraphQL endpoint with a user-provided query.
# YouTube Search: Calls the YouTube Data API's 'search' endpoint to find videos.
# YouTube Videos: Calls the YouTube Data API's 'videos' endpoint to get video details.
# YouTube CommentThreads: Calls the YouTube Data API's 'commentThreads' endpoint to get video comments.
# YouTube PlaylistItems: Calls the YouTube Data API's 'playlistItems' endpoint to get playlist items.
# docling_text_extractor: Extracts text from PDFs with OCR fallback.
# docling_table_extractor: Extracts structured tables from PDF documents.
# docling_full_processor: Comprehensive PDF processing with text, OCR, and table extraction.
# add_file_to_collection: Add a file to the Needle collection.
# search_collection: Search the Needle collection using a retrieval chain.




#----------------------------------------------------------------------------------------------------------------------------
# Define Input Schema# Define Input Schema
class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up.")
    max_results: Optional[int] = Field(default=10, description="The maximum number of search results to return.")

# Define the Tool
class TavilySearchTool:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform a web search using the Tavily search engine.
        """
        try:
            # Initialize the Tavily search tool with the configured max_results
            search_tool = TavilySearchResults(max_results=self.max_results)

            # Perform the search
            result = search_tool.invoke({"query": query})

            # Return the search results
            return result
        except Exception as e:
            return {"error": str(e)}

# Create the LangChain Tool
tavily_search_tool = Tool(
    name="Tavily Search",
    func=TavilySearchTool().search,
    description="Performs web searches using the Tavily search engine, providing accurate and trusted results for general queries.",
    args_schema=SearchToolInput
)

    
    
#----------------------------------------------------------------------------------------------------------------------------   
# Define Input Schema
class AmadeusFlightSearchInput(BaseModel):
    origin: str = Field(..., description="The origin airport code (e.g., 'JFK').")
    destination: str = Field(..., description="The destination airport code (e.g., 'LAX').")
    departure_date: str = Field(..., description="The departure date in YYYY-MM-DD format.")
    adults: int = Field(default=1, description="The number of adult passengers.")
    max_price: Optional[int] = Field(default=None, description="The maximum price for the flight.")

# Define the Tool
class AmadeusFlightSearchTool:
    def __init__(self):
        self.client_id = os.getenv('AMADEUS_CLIENT_ID')
        self.client_secret = os.getenv('AMADEUS_CLIENT_SECRET')
        if not self.client_id or not self.client_secret:
            raise ValueError("AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET environment variables are required.")

    def _get_access_token(self):
        """
        Fetches an access token from the Amadeus API.
        """
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()  # Raise an error for bad status codes
            token_data = response.json()
            return token_data["access_token"]
        except Exception as e:
            raise Exception(f"Failed to obtain access token: {str(e)}")

    def search_flights(self, input: AmadeusFlightSearchInput) -> str:
        """
        Searches for flights using the Amadeus API.
        """
        try:
            # Get the access token
            access_token = self._get_access_token()
            print(f"Access Token: {access_token}")  # Debugging: Print the access token

            # Prepare the API request
            url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
            params = {
                "originLocationCode": input.origin,
                "destinationLocationCode": input.destination,
                "departureDate": input.departure_date,
                "adults": input.adults,
                "max": 5  # Limit to 5 results for brevity
            }
            if input.max_price:
                params["maxPrice"] = input.max_price

            headers = {"Authorization": f"Bearer {access_token}"}
            print(f"Request Headers: {headers}")  # Debugging: Print the headers

            # Make the API request
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an error for bad status codes

            # Parse the response
            data = response.json()["data"]
            results = []
            for offer in data:
                itinerary = offer['itineraries'][0]
                price = offer['price']
                result = f"Flight: {itinerary['segments'][0]['carrierCode']} {itinerary['segments'][0]['number']}\n"
                result += f"Departure: {itinerary['segments'][0]['departure']['iataCode']} at {itinerary['segments'][0]['departure']['at']}\n"
                result += f"Arrival: {itinerary['segments'][-1]['arrival']['iataCode']} at {itinerary['segments'][-1]['arrival']['at']}\n"
                result += f"Price: {price['total']} {price['currency']}\n"
                result += "---\n"
                results.append(result)
            return "\n".join(results) if results else "No flights found matching the criteria."
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (e.g., 401 Unauthorized)
            error_message = f"HTTP Error: {e.response.status_code} - {e.response.text}"
            print(error_message)  # Debugging: Print the error message
            return error_message
        except Exception as e:
            # Handle other exceptions
            error_message = f"An error occurred while searching for flights: {str(e)}"
            print(error_message)  # Debugging: Print the error message
            return error_message

# # Create the LangChain Tool
amadeus_tool = Tool(
    name="Amadeus Flight Search",
    func=AmadeusFlightSearchTool().search_flights,
    description="Searches for flight availability and prices using the Amadeus API.",
    args_schema=AmadeusFlightSearchInput
)


# # Example Usage
# print(amadeus_tool.run({
#     "origin": "JFK",
#     "destination": "LAX",
#     "departure_date": "2023-12-15",
#     "adults": 1
# }))

# class AmadeusFlightSearchTool(BaseTool):
#     name: str = "Amadeus Flight Search Tool"
#     description: str = "Searches for flight availability and prices using the Amadeus API."
#     client_id: str = Field(default_factory=lambda: environ.get('AMADEUS_CLIENT_ID'))
#     client_secret: str = Field(default_factory=lambda: environ.get('AMADEUS_CLIENT_SECRET'))

#     def __init__(self, **data):
#         super().__init__(**data)

#     def _get_access_token(self):
#         url = "https://test.api.amadeus.com/v1/security/oauth2/token"
#         headers = {"Content-Type": "application/x-www-form-urlencoded"}
#         data = {
#             "grant_type": "client_credentials",
#             "client_id": self.client_id,
#             "client_secret": self.client_secret
#         }
        
#         try:
#             response = requests.post(url, headers=headers, data=data)
#             response.raise_for_status()
#             return response.json()["access_token"]
#         except Exception as e:
#             logger.error(f"Failed to obtain access token: {str(e)}")
#             raise

#     def _run(self, origin: str, destination: str, departure_date: str, adults: int = 1, max_price: Optional[int] = None) -> str:
#         access_token = self._get_access_token()

#         url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
#         params = {
#             "originLocationCode": origin,
#             "destinationLocationCode": destination,
#             "departureDate": departure_date,
#             "adults": adults,
#             "max": 5  # Limit to 5 results for brevity
#         }
#         if max_price:
#             params["maxPrice"] = max_price

#         headers = {"Authorization": f"Bearer {access_token}"}

#         try:
#             response = requests.get(url, headers=headers, params=params)
#             response.raise_for_status()
#             data = response.json()["data"]

#             results = []
#             for offer in data:
#                 itinerary = offer['itineraries'][0]
#                 price = offer['price']
#                 result = f"Flight: {itinerary['segments'][0]['carrierCode']} {itinerary['segments'][0]['number']}\n"
#                 result += f"Departure: {itinerary['segments'][0]['departure']['iataCode']} at {itinerary['segments'][0]['departure']['at']}\n"
#                 result += f"Arrival: {itinerary['segments'][-1]['arrival']['iataCode']} at {itinerary['segments'][-1]['arrival']['at']}\n"
#                 result += f"Price: {price['total']} {price['currency']}\n"
#                 result += "---\n"
#                 results.append(result)

#             return "\n".join(results) if results else "No flights found matching the criteria."

#         except Exception as e:
#             logger.error(f"Error in flight search: {str(e)}")
#             return f"An error occurred while searching for flights: {str(e)}"

#     async def _arun(self, origin: str, destination: str, departure_date: str, adults: int = 1, max_price: Optional[int] = None) -> str:
#         return self._run(origin, destination, departure_date, adults, max_price)

# Usage example:
# tool = AmadeusFlightSearchTool(client_id='YOUR_API_KEY', client_secret='YOUR_API_SECRET')
# result = tool._run(origin='PAR', max_price=200)
# print(result)

#----------------------------------------------------------------------------------------------------------------------------
class AmadeusHotelSearchInput(BaseModel):
    city_code: str = Field(..., description="The city code for the hotel search (e.g., 'NYC').")
    check_in_date: str = Field(..., description="The check-in date in YYYY-MM-DD format.")
    check_out_date: str = Field(..., description="The check-out date in YYYY-MM-DD format.")
    adults: int = Field(default=2, description="The number of adult guests.")
    children: Optional[int] = Field(default=None, description="The number of child guests.")

# Define the Tool
class AmadeusHotelSearchTool:
    def __init__(self):
        self.client_id = os.getenv('AMADEUS_CLIENT_ID')
        self.client_secret = os.getenv('AMADEUS_CLIENT_SECRET')
        if not self.client_id or not self.client_secret:
            raise ValueError("Amadeus API credentials are missing. Please set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET environment variables.")

    def _get_access_token(self):
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            return response.json()["access_token"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Authentication failed. Please check your Amadeus API credentials.")
            else:
                raise Exception(f"HTTP error occurred: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while obtaining access token: {str(e)}")

    def search_hotels(self, city_code: str, check_in_date: str, check_out_date: str, adults: int = 2, children: Optional[int] = None) -> str:
        try:
            access_token = self._get_access_token()
        except ValueError as e:
            return str(e)
        url = "https://test.api.amadeus.com/v2/shopping/hotel-offers"
        params = {
            "cityCode": city_code,
            "checkInDate": check_in_date,
            "checkOutDate": check_out_date,
            "adults": adults
        }
        if children:
            params["children"] = children
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()["data"]
            results = []
            for offer in data:
                hotel = offer['hotel']
                price = offer['offers'][0]['price']
                result = f"Hotel: {hotel['name']}\n"
                result += f"Rating: {hotel.get('rating', 'N/A')}\n"
                result += f"Price: {price['total']} {price['currency']}\n"
                result += f"Available: {offer['offers'][0].get('available', 'N/A')}\n"
                result += "---\n"
                results.append(result)
            return "\n".join(results) if results else "No hotels found matching the criteria."
        except requests.exceptions.HTTPError as e:
            return f"An error occurred while searching for hotels: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred while searching for hotels: {str(e)}"

amadeus_hotel_tool = Tool(
    name="Amadeus Hotel Search",
    func=AmadeusHotelSearchTool().search_hotels,
    description="Searches for hotel availability and prices using the Amadeus API.",
    args_schema=AmadeusHotelSearchInput
)

# print(amadeus_hotel_tool.run({
#     "city_code": "NYC",
#     "check_in_date": "2023-12-15",
#     "check_out_date": "2023-12-20",
#     "adults": 2
# }))

# class AmadeusHotelSearchTool(BaseTool):
#     name: str = "Amadeus Hotel Search Tool"
#     description: str = "Searches for hotel availability and prices using the Amadeus API."
#     client_id: str = Field(default_factory=lambda: environ.get('AMADEUS_CLIENT_ID'))
#     client_secret: str = Field(default_factory=lambda: environ.get('AMADEUS_CLIENT_SECRET'))

#     def __init__(self, **data):
#         super().__init__(**data)
#         if not self.client_id or not self.client_secret:
#             logger.error("Amadeus API credentials are missing. Please set AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET environment variables.")

#     def _get_access_token(self):
#         url = "https://test.api.amadeus.com/v1/security/oauth2/token"
#         headers = {"Content-Type": "application/x-www-form-urlencoded"}
#         data = {
#             "grant_type": "client_credentials",
#             "client_id": self.client_id,
#             "client_secret": self.client_secret
#         }
        
#         try:
#             response = requests.post(url, headers=headers, data=data)
#             response.raise_for_status()
#             return response.json()["access_token"]
#         except requests.exceptions.HTTPError as e:
#             if e.response.status_code == 401:
#                 logger.error(f"Authentication failed. Please check your Amadeus API credentials. Error: {e.response.text}")
#                 raise ValueError("Authentication failed. Please check your Amadeus API credentials.")
#             else:
#                 logger.error(f"HTTP error occurred: {e}")
#                 raise
#         except Exception as e:
#             logger.error(f"An unexpected error occurred while obtaining access token: {str(e)}")
#             raise

#     def _run(self, city_code: str, check_in_date: str, check_out_date: str, adults: int = 2, children: Optional[int] = None) -> str:
#         try:
#             access_token = self._get_access_token()
#         except ValueError as e:
#             return str(e)

#         url = "https://test.api.amadeus.com/v2/shopping/hotel-offers"
#         params = {
#             "cityCode": city_code,
#             "checkInDate": check_in_date,
#             "checkOutDate": check_out_date,
#             "adults": adults
#         }
#         if children:
#             params["children"] = children

#         headers = {"Authorization": f"Bearer {access_token}"}

#         try:
#             response = requests.get(url, headers=headers, params=params)
#             response.raise_for_status()
#             data = response.json()["data"]

#             results = []
#             for offer in data:
#                 hotel = offer['hotel']
#                 price = offer['offers'][0]['price']
#                 result = f"Hotel: {hotel['name']}\n"
#                 result += f"Rating: {hotel.get('rating', 'N/A')}\n"
#                 result += f"Price: {price['total']} {price['currency']}\n"
#                 result += f"Available: {offer['offers'][0].get('available', 'N/A')}\n"
#                 result += "---\n"
#                 results.append(result)

#             return "\n".join(results) if results else "No hotels found matching the criteria."

#         except requests.exceptions.HTTPError as e:
#             logger.error(f"HTTP error occurred: {e}")
#             return f"An error occurred while searching for hotels: {str(e)}"
#         except Exception as e:
#             logger.error(f"An unexpected error occurred: {str(e)}")
#             return f"An unexpected error occurred while searching for hotels: {str(e)}"

#     async def _arun(self, city_code: str, check_in_date: str, check_out_date: str, adults: int = 2, children: Optional[int] = None) -> str:
#         return self._run(city_code, check_in_date, check_out_date, adults, children)

#----------------------------------------------------------------------------------------------------------------------------

class GeoapifyPlacesSearchInput(BaseModel):
    location: str = Field(..., description="The location to search for points of interest (e.g., 'New York').")
    categories: Optional[str] = Field(default=None, description="The categories of places to search for (e.g., 'accommodation').")
    limit: int = Field(default=20, description="The maximum number of results to return.")

# Define the Tool
class GeoapifyPlacesSearchTool:
    def __init__(self):
        self.api_key = os.getenv("GEOAPIFY_API_KEY")
        self.base_url = "https://api.geoapify.com/v1"
        if not self.api_key:
            raise ValueError("Geoapify API key is missing. Please set the GEOAPIFY_API_KEY environment variable.")

    def search_places(self, location: str, categories: Optional[str] = None, limit: int = 20) -> str:
        try:
            # Geocode the location to get coordinates
            geocoding_url = f"{self.base_url}/geocode/search"
            geocoding_params = {
                "text": location,
                "format": "json",
                "apiKey": self.api_key
            }
            geocoding_response = requests.get(geocoding_url, params=geocoding_params)
            geocoding_data = geocoding_response.json()
            
            if not geocoding_data.get('features'):
                return f"Could not find coordinates for the location: {location}"
            
            lat = geocoding_data['features'][0]['properties']['lat']
            lon = geocoding_data['features'][0]['properties']['lon']
            
            # Search for places
            places_url = f"{self.base_url}/places"
            places_params = {
                "filter": f"circle:{lon},{lat},5000",
                "bias": f"proximity:{lon},{lat}",
                "limit": limit,
                "apiKey": self.api_key
            }
            
            if categories:
                places_params["categories"] = categories
            places_response = requests.get(places_url, params=places_params)
            places_data = places_response.json()
            
            # Format the response
            results = []
            for place in places_data.get('features', []):
                props = place['properties']
                result = f"Name: {props.get('name', 'N/A')}, "
                result += f"Category: {props.get('categories', ['N/A'])[0]}, "
                result += f"Address: {props.get('formatted', 'N/A')}"
                results.append(result)
            return "\n".join(results) if results else "No places found matching the criteria."
        except Exception as e:
            return f"An error occurred: {str(e)}"


geoapify_tool = Tool(
    name="Geoapify Places Search",
    func=GeoapifyPlacesSearchTool().search_places,
    description="Searches for points of interest in a location using the Geoapify Places API.",
    args_schema=GeoapifyPlacesSearchInput
)

# # Example Usage
# print(geoapify_tool.run({
#     "location": "New York",
#     "categories": "accommodation",
#     "limit": 5
# }))


# class GeoapifyPlacesSearchTool(BaseTool):
#     name: str = "Geoapify Places Search Tool"
#     description: str = "Searches for points of interest in a location using the Geoapify Places API."
#     api_key: str = Field(default_factory=lambda: environ.get("GEOAPIFY_API_KEY"))
#     base_url: str = "https://api.geoapify.com/v1"

#     def _run(self, location: str, categories: Optional[str] = None, limit: int = 20) -> str:
#         try:
#             # Geocode the location to get coordinates
#             geocoding_url = f"{self.base_url}/geocode/search"
#             geocoding_params = {
#                 "text": location,
#                 "format": "json",
#                 "apiKey": self.api_key
#             }
#             geocoding_response = requests.get(geocoding_url, params=geocoding_params)
#             geocoding_data = geocoding_response.json()
            
#             if not geocoding_data.get('features'):
#                 return f"Could not find coordinates for the location: {location}"
            
#             lat = geocoding_data['features'][0]['properties']['lat']
#             lon = geocoding_data['features'][0]['properties']['lon']

#             # Search for places
#             places_url = f"{self.base_url}/places"
#             places_params = {
#                 "filter": f"circle:{lon},{lat},5000",
#                 "bias": f"proximity:{lon},{lat}",
#                 "limit": limit,
#                 "apiKey": self.api_key
#             }
            
#             if categories:
#                 places_params["categories"] = categories

#             places_response = requests.get(places_url, params=places_params)
#             places_data = places_response.json()
            
#             # Format the response
#             results = []
#             for place in places_data.get('features', []):
#                 props = place['properties']
#                 result = f"Name: {props.get('name', 'N/A')}, "
#                 result += f"Category: {props.get('categories', ['N/A'])[0]}, "
#                 result += f"Address: {props.get('formatted', 'N/A')}"
#                 results.append(result)

#             return "\n".join(results) if results else "No places found matching the criteria."

#         except Exception as e:
#             return f"An error occurred: {str(e)}"

# Usage example:
# tool = GeoapifyPlacesSearchTool(api_key='YOUR_API_KEY')
# result = tool._run(location="New York City", categories="catering.restaurant", limit=5)
# print(result)



#----------------------------------------------------------------------------------------------------------------------------

# Define Input Schema
class WeatherSearchInput(BaseModel):
    location: str = Field(..., description="The location to get weather information for (e.g., 'New York').")
    date: Optional[str] = Field(None, description="The date for the weather forecast in YYYY-MM-DD format.")

# Define the WeatherSearchTool class
class WeatherSearchTool:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5"
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is missing. Please set the OPENWEATHERMAP_API_KEY environment variable.")

    def get_weather(self, input: WeatherSearchInput) -> str:
        try:
            # Step 1: Get current weather or forecast for the specified location
            if input.date:
                # Use the forecast endpoint for future dates
                forecast_url = f"{self.base_url}/forecast"
                forecast_params = {
                    "q": input.location,
                    "appid": self.api_key,
                    "units": "metric"  # Use metric units (Celsius)
                }
                forecast_response = requests.get(forecast_url, params=forecast_params)
                forecast_data = forecast_response.json()

                if forecast_response.status_code != 200:
                    return f"Could not fetch weather data. Error: {forecast_data.get('message', 'Unknown error')}"

                # Find the weather for the specified date
                target_date = datetime.strptime(input.date, "%Y-%m-%d").date()
                for forecast in forecast_data.get('list', []):
                    forecast_date = datetime.fromtimestamp(forecast['dt']).date()
                    if forecast_date == target_date:
                        weather = forecast['weather'][0]['description']
                        temp_min = forecast['main']['temp_min']
                        temp_max = forecast['main']['temp_max']
                        humidity = forecast['main']['humidity']
                        return (
                            f"Weather in {input.location} on {input.date}:\n"
                            f"Description: {weather}\n"
                            f"Temperature: {temp_min}°C to {temp_max}°C\n"
                            f"Humidity: {humidity}%"
                        )
                return f"No weather data found for {input.location} on {input.date}."
            else:
                # Use the weather endpoint for current weather
                weather_url = f"{self.base_url}/weather"
                weather_params = {
                    "q": input.location,
                    "appid": self.api_key,
                    "units": "metric"  # Use metric units (Celsius)
                }
                weather_response = requests.get(weather_url, params=weather_params)
                weather_data = weather_response.json()

                if weather_response.status_code != 200:
                    return f"Could not fetch weather data. Error: {weather_data.get('message', 'Unknown error')}"

                weather = weather_data['weather'][0]['description']
                temp = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                return (
                    f"Current weather in {input.location}:\n"
                    f"Description: {weather}\n"
                    f"Temperature: {temp}°C\n"
                    f"Humidity: {humidity}%"
                )
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Create the LangChain Tool
weather_tool = Tool(
    name="Weather Search",
    func=WeatherSearchTool().get_weather,
    description="Provides weather information for a given location and date using the OpenWeatherMap API.",
    args_schema=WeatherSearchInput
)

# # Example Usage
# print(weather_tool.run({
#     "location": "New York",
#     "date": "2023-12-15"
# }))

#----------------------------------------------------------------------------------------------------------------------------

import googlemaps
from googlemaps.convert import decode_polyline
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import os

# Define Input Schema
class GoogleMapsRouteInput(BaseModel):
    start: str = Field(..., description="The starting address (e.g., 'San Francisco, CA').")
    end: str = Field(..., description="The destination address (e.g., 'Los Angeles, CA').")
    waypoints: List[str] = Field(..., description="A list of intermediate waypoints (e.g., ['Santa Cruz, CA', 'Monterey, CA']).")
    transit_type: str = Field(default="driving", description="The mode of transportation (e.g., 'driving', 'walking').")
    optimize_waypoints: bool = Field(default=True, description="Whether to optimize the order of waypoints.")

# Define the Tool
class GoogleMapsRouteTool:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("Google Maps API key is missing. Please set the GOOGLE_MAPS_API_KEY environment variable.")
        self.gmaps = googlemaps.Client(key=self.api_key)

    def get_route(self, start: str, end: str, waypoints: List[str], transit_type: str = "driving", optimize_waypoints: bool = True) -> Dict[str, Any]:
        """
        Generate a route between start and end points, including waypoints.
        Args:
            start: The starting address.
            end: The destination address.
            waypoints: A list of intermediate waypoints.
            transit_type: The mode of transportation (e.g., "driving", "walking").
            optimize_waypoints: Whether to optimize the order of waypoints.
        Returns:
            A dictionary containing the route details.
        """
        try:
            # Geocode the start, end, and waypoints
            start_location = self.gmaps.geocode(start)[0]
            end_location = self.gmaps.geocode(end)[0]
            waypoint_locations = [self.gmaps.geocode(wp)[0] for wp in waypoints]

            # Generate directions
            directions_result = self.gmaps.directions(
                origin=start_location["formatted_address"],
                destination=end_location["formatted_address"],
                waypoints=[wp["formatted_address"] for wp in waypoint_locations],
                mode=transit_type,
                optimize_waypoints=optimize_waypoints,
                departure_time=datetime.now(),
            )

            if not directions_result:
                return {"error": "No route found for the given addresses."}

            # Extract and decode the polyline
            polyline = directions_result[0]["overview_polyline"]["points"]
            coordinates = decode_polyline(polyline)

            return {
                "start": start_location["formatted_address"],
                "end": end_location["formatted_address"],
                "waypoints": [wp["formatted_address"] for wp in waypoint_locations],
                "route_coordinates": coordinates,
                "summary": directions_result[0]["summary"],
                "distance": directions_result[0]["legs"][0]["distance"]["text"],
                "duration": directions_result[0]["legs"][0]["duration"]["text"],
            }
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}


googlemaps_tool = Tool(
    name="Route Finder",
    func=GoogleMapsRouteTool().get_route,
    description="Finds the optimal route between locations using the Google Maps API.",
    args_schema=GoogleMapsRouteInput
)


# # Set the Google Maps API key as an environment variable
# import os
# os.environ["GOOGLE_MAPS_API_KEY"] = "your_google_maps_api_key"

# # Initialize the tool
# route_tool = GoogleMapsRouteTool()

# # Define inputs
# start = "San Francisco, CA"
# end = "Los Angeles, CA"
# waypoints = ["Santa Cruz, CA", "Monterey, CA"]

# # Generate a route
# route = route_tool.get_route(start, end, waypoints)

# # Print the results
# if "error" in route:
#     print("Error:", route["error"])
# else:
#     print("Route Summary:", route["summary"])
#     print("Distance:", route["distance"])
#     print("Duration:", route["duration"])
#     print("Route Coordinates:", route["route_coordinates"])

#------------------------------------------------------------------------------------------------------------------------------
@tool
def multiply_numbers(numbers: list) -> float:
    """
    Multiplies a list of numbers and returns the product.

    Args:
        numbers (list): A list of numbers to multiply.

    Returns:
        float: The product of the numbers.
    """
    product = 1
    for num in numbers:
        product *= num
    return product

# Step 2: Wrap the tool in a LangChain Tool object
multiply_tool = Tool(
    name="multiply_numbers",
    func=multiply_numbers,
    description="Multiplies a list of numbers and returns the product."
)


#----------------------------------------------------------------------------------------------------------------------------

import os
import requests
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Define Input Schema
class FlightSearchInput(BaseModel):
    departure_id: str = Field(..., description="The departure airport code or location kgmid.")
    arrival_id: str = Field(..., description="The arrival airport code or location kgmid.")
    outbound_date: str = Field(..., description="The outbound date in YYYY-MM-DD format.")
    return_date: str = Field(None, description="The return date in YYYY-MM-DD format (optional for one-way flights).")
    currency: str = Field(default="USD", description="The currency for the flight prices.")
    hl: str = Field(default="en", description="The language for the search results.")
    adults: int = Field(default=1, description="The number of adult passengers.")
    children: int = Field(default=0, description="The number of child passengers.")
    infants_in_seat: int = Field(default=0, description="The number of infants in seat.")
    infants_on_lap: int = Field(default=0, description="The number of infants on lap.")
    travel_class: int = Field(default=1, description="The travel class (1: Economy, 2: Premium Economy, 3: Business, 4: First).")
    sort_by: int = Field(default=1, description="The sorting order of the results (1: Top flights, 2: Price, etc.).")
    deep_search: bool = Field(default=False, description="Enable deep search for more precise results.")

# Define the Tool
# class GoogleFlightsSearchTool:
#     def __init__(self):
#         self.api_key = os.getenv("SERPAPI_API_KEY")
#         if not self.api_key:
#             raise ValueError("SerpApi API key is missing. Please set the SERPAPI_API_KEY environment variable.")
#         self.base_url = "https://serpapi.com/search.json"

#     def _extract_flight_details(self, flight: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Extract and structure detailed flight information from a flight result.
        
#         Args:
#             flight: A dictionary representing a flight result from the API.
        
#         Returns:
#             A dictionary containing structured flight details.
#         """
#         details = {
#             "airlines": [leg["airline"] for leg in flight.get("flights", [])],
#             "price": flight.get("price"),
#             "departure_airport": flight.get("flights", [{}])[0].get("departure_airport", {}).get("name"),
#             "arrival_airport": flight.get("flights", [{}])[-1].get("arrival_airport", {}).get("name"),
#             "departure_time": flight.get("flights", [{}])[0].get("departure_airport", {}).get("time"),
#             "arrival_time": flight.get("flights", [{}])[-1].get("arrival_airport", {}).get("time"),
#             "total_duration": flight.get("total_duration"),
#             "layovers": [
#                 {
#                     "duration": layover.get("duration"),
#                     "airport": layover.get("name"),
#                     "overnight": layover.get("overnight", False),
#                 }
#                 for layover in flight.get("layovers", [])
#             ],
#             "travel_class": flight.get("flights", [{}])[0].get("travel_class"),
#             "carbon_emissions": flight.get("carbon_emissions", {}).get("this_flight"),
#             "booking_token": flight.get("booking_token"),
#             "departure_token": flight.get("departure_token"),
#         }
#         return details

#     def search_flights(self, input: FlightSearchInput) -> Dict[str, Any]:
#         """
#         Search for flights using the Google Flights API via SerpApi.
        
#         Args:
#             input: The flight search parameters.
        
#         Returns:
#             A dictionary containing the flight search results, with detailed information for cheap and expensive flights.
#         """
#         params = {
#             "engine": "google_flights",
#             "departure_id": input.departure_id,
#             "arrival_id": input.arrival_id,
#             "outbound_date": input.outbound_date,
#             "currency": input.currency,
#             "hl": input.hl,
#             "adults": input.adults,
#             "children": input.children,
#             "infants_in_seat": input.infants_in_seat,
#             "infants_on_lap": input.infants_on_lap,
#             "travel_class": input.travel_class,
#             "sort_by": input.sort_by,
#             "deep_search": input.deep_search,
#             "api_key": self.api_key,
#         }
        
#         if input.return_date:
#             params["return_date"] = input.return_date

#         try:
#             response = requests.get(self.base_url, params=params)
#             response.raise_for_status()
#             results = response.json()

#             # Separate flights into cheap and expensive based on price insights
#             price_insights = results.get("price_insights", {})
#             typical_price_range = price_insights.get("typical_price_range", [0, float("inf")])

#             cheap_flights = []
#             expensive_flights = []

#             for flight in results.get("best_flights", []) + results.get("other_flights", []):
#                 flight_details = self._extract_flight_details(flight)
#                 if flight["price"] <= typical_price_range[1]:
#                     cheap_flights.append(flight_details)
#                 else:
#                     expensive_flights.append(flight_details)

#             return {
#                 "cheap_flights": cheap_flights,
#                 "expensive_flights": expensive_flights,
#                 "price_insights": price_insights,
#                 "search_metadata": results.get("search_metadata", {}),
#             }
#         except Exception as e:
#             return {"error": f"An error occurred: {str(e)}"}


# Define the Tool
class GoogleFlightsSearchTool:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SerpApi API key is missing. Please set the SERPAPI_API_KEY environment variable.")
        self.base_url = "https://serpapi.com/search.json"

    def _extract_flight_details(self, flight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and structure detailed flight information from a flight result.
        
        Args:
            flight: A dictionary representing a flight result from the API.
        
        Returns:
            A dictionary containing structured flight details.
        """
        details = {
            "airlines": [leg["airline"] for leg in flight.get("flights", [])],
            "price": flight.get("price"),
            "departure_airport": flight.get("flights", [{}])[0].get("departure_airport", {}).get("name"),
            "arrival_airport": flight.get("flights", [{}])[-1].get("arrival_airport", {}).get("name"),
            "departure_time": flight.get("flights", [{}])[0].get("departure_airport", {}).get("time"),
            "arrival_time": flight.get("flights", [{}])[-1].get("arrival_airport", {}).get("time"),
            "total_duration": flight.get("total_duration"),
            "layovers": [
                {
                    "duration": layover.get("duration"),
                    "airport": layover.get("name"),
                    "overnight": layover.get("overnight", False),
                }
                for layover in flight.get("layovers", [])
            ],
            "travel_class": flight.get("flights", [{}])[0].get("travel_class"),
            "carbon_emissions": flight.get("carbon_emissions", {}).get("this_flight"),
            "booking_token": flight.get("booking_token"),
            "departure_token": flight.get("departure_token"),
        }
        return details

    def search_flights(self, input: FlightSearchInput) -> Dict[str, Any]:
        """
        Search for flights using the Google Flights API via SerpApi.
        
        Args:
            input: The flight search parameters.
        
        Returns:
            A dictionary containing the flight search results, with detailed information for best and other flights.
        """
        params = {
            "engine": "google_flights",
            "departure_id": input.departure_id,
            "arrival_id": input.arrival_id,
            "outbound_date": input.outbound_date,
            "currency": input.currency,
            "hl": input.hl,
            "adults": input.adults,
            "children": input.children,
            "infants_in_seat": input.infants_in_seat,
            "infants_on_lap": input.infants_on_lap,
            "travel_class": input.travel_class,
            "sort_by": input.sort_by,
            "deep_search": input.deep_search,
            "api_key": self.api_key,
        }
        
        if input.return_date:
            params["return_date"] = input.return_date

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            results = response.json()

            # Extract best and other flights
            best_flights = [self._extract_flight_details(flight) for flight in results.get("best_flights", [])]
            other_flights = [self._extract_flight_details(flight) for flight in results.get("other_flights", [])]

            return {
                "best_flights": best_flights,
                "other_flights": other_flights,
                "search_metadata": results.get("search_metadata", {}),
                "search_parameters": results.get("search_parameters", {}),
            }
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}
        
        
flight_tool = Tool(
    name="Flight Search",
    func=GoogleFlightsSearchTool().search_flights,
    description="Provides flight information between two locations, including airlines, prices, departure/arrival times, and more.",
    args_schema=FlightSearchInput
)


# # Example Usage
# tool = GoogleFlightsSearchTool()
# input_data = FlightSearchInput(
#     departure_id="JFK",
#     arrival_id="LAX",
#     outbound_date="2025-01-22",
#     return_date="2025-01-28",
#     currency="USD",
#     hl="en",
#     adults=1,
#     children=0,
#     infants_in_seat=0,
#     infants_on_lap=0,
#     travel_class=1,
#     sort_by=2,
#     deep_search=True,
# )
# results = tool.search_flights(input_data)
# print(results)




#----------------------------------------------------------------------------------------------------------------------------

import os
import requests
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Define Input Schema
class GoogleScholarSearchInput(BaseModel):
    query: str = Field(..., description="The search query for Google Scholar (e.g., 'machine learning').")
    cites: Optional[str] = Field(None, description="The unique ID for an article to trigger 'Cited By' searches.")
    as_ylo: Optional[int] = Field(None, description="The year from which to include results (e.g., 2018).")
    as_yhi: Optional[int] = Field(None, description="The year until which to include results (e.g., 2023).")
    start: Optional[int] = Field(None, description="The result offset for pagination (e.g., 0 for the first page).")
    num: Optional[int] = Field(None, description="The maximum number of results to return (default is 10).")
    hl: Optional[str] = Field(None, description="The language for the search (e.g., 'en' for English).")

# Define the Tool
class GoogleScholarSearchTool:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        self.base_url = "https://serpapi.com/search"
        if not self.api_key:
            raise ValueError("SerpApi API key is missing. Please set the SERPAPI_API_KEY environment variable.")

    def search_scholar(self, query: str, cites: Optional[str] = None, as_ylo: Optional[int] = None, as_yhi: Optional[int] = None, start: Optional[int] = None, num: Optional[int] = None, hl: Optional[str] = None) -> Dict[str, Any]:
        """
        Search Google Scholar using the SerpApi service.
        Args:
            query: The search query.
            cites: The unique ID for an article to trigger 'Cited By' searches.
            as_ylo: The year from which to include results.
            as_yhi: The year until which to include results.
            start: The result offset for pagination.
            num: The maximum number of results to return.
            hl: The language for the search.
        Returns:
            A dictionary containing the search results.
        """
        try:
            # Prepare the parameters for the API request
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "output": "json"
            }

            # Add optional parameters if provided
            if cites:
                params["cites"] = cites
            if as_ylo:
                params["as_ylo"] = as_ylo
            if as_yhi:
                params["as_yhi"] = as_yhi
            if start:
                params["start"] = start
            if num:
                params["num"] = num
            if hl:
                params["hl"] = hl

            # Make the API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the JSON response
            results = response.json()

            # Extract relevant information from the results
            organic_results = results.get("organic_results", [])
            simplified_results = []

            for result in organic_results:
                simplified_results.append({
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "snippet": result.get("snippet"),
                    "publication_info": result.get("publication_info", {}).get("summary"),
                    "cited_by": result.get("inline_links", {}).get("cited_by", {}).get("total"),
                    "versions": result.get("inline_links", {}).get("versions", {}).get("total")
                })

            return {
                "search_metadata": results.get("search_metadata", {}),
                "organic_results": simplified_results,
                "related_searches": results.get("related_searches", []),
                "pagination": results.get("pagination", {})
            }

        except requests.exceptions.RequestException as e:
            return {"error": f"An error occurred while making the API request: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}


google_scholar_tool = Tool(
    name="Google Scholar Search",
    func=GoogleScholarSearchTool().search_scholar,
    description="Provides academic research results from Google Scholar, including titles, links, snippets, publication info, citations, and versions.",
    args_schema=GoogleScholarSearchInput
)

#----------------------------------------------------------------------------------------------------------------------------
# Define Input Schema
class BookingSearchInput(BaseModel):
    location: str = Field(..., description="The destination city or location (e.g., 'London').")
    checkin_date: str = Field(..., description="The check-in date in YYYY-MM-DD format.")
    checkout_date: str = Field(..., description="The check-out date in YYYY-MM-DD format.")
    adults: int = Field(default=2, description="The number of adult guests.")
    rooms: int = Field(default=1, description="The number of rooms.")
    currency: str = Field(default="USD", description="The currency for the prices.")

# Define the Booking Scraper Tool
class BookingScraperTool:
    def __init__(self):
        self.base_url = "https://www.booking.com/searchresults.html"
        self.session = requests.Session()

    def search(self, input: BookingSearchInput) -> List[Dict]:
        """
        Scrape hotel data from Booking.com based on the provided input parameters.
        Args:
            location: The destination city or location.
            checkin_date: The check-in date in YYYY-MM-DD format.
            checkout_date: The check-out date in YYYY-MM-DD format.
            adults: The number of adult guests.
            rooms: The number of rooms.
            currency: The currency for the prices.
        Returns:
            A list of dictionaries containing the scraped hotel data.
        """
        # Define search parameters
        params = {
            'ss': input.location,
            'dest_type': 'city',
            'checkin': input.checkin_date,
            'checkout': input.checkout_date,
            'group_adults': input.adults,
            'no_rooms': input.rooms,
            'selected_currency': input.currency
        }

        # Define headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

        # Send the request and parse the response
        response = self.session.get(self.base_url, params=params, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract hotel data
        results = []
        for card in soup.find_all('div', {'data-testid': 'property-card'}):
            try:
                # Extract hotel name
                name = card.find('div', {'data-testid': 'title'}).text.strip()

                # Try multiple price selectors
                price_elem = None
                selectors = [
                    {'class': 'prco-valign-middle-helper'},
                    {'data-testid': 'price-and-discounted-price'},
                    {'data-id': 'price-box'}
                ]

                for selector in selectors:
                    price_elem = card.find(['span', 'div'], selector)
                    if price_elem:
                        break

                price = price_elem.text.strip() if price_elem else 'N/A'

                # Extract rating
                rating = card.find('div', {'data-testid': 'review-score'})
                rating = rating.text.strip() if rating else 'N/A'

                # Extract link
                link_element = card.find('a', {'data-testid': 'title-link'})
                link = link_element['href'] if link_element else 'N/A'

                # Ensure the link is a full URL
                if link != 'N/A' and not link.startswith('http'):
                    link = f"https://www.booking.com{link}"

                results.append({
                    'name': name,
                    'price': price,
                    'rating': rating,
                    'link': link
                })

            except Exception as e:
                print(f'Error parsing card: {str(e)}')
                continue

        return results
    
# Create the LangChain Tool
booking_tool = Tool(
    name="Booking Scraper",
    func=BookingScraperTool().search,
    description="Scrapes hotel data from Booking.com based on destination, check-in/check-out dates, and other parameters.",
    args_schema=BookingSearchInput
)


#----------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------




# Validate and retrieve the NEEDLE_API_KEY and OPENAI_API_KEY from environment variables
def validate_api_keys() -> tuple[str, str]:
    """
    Validate and retrieve the NEEDLE_API_KEY and OPENAI_API_KEY from environment variables.
    
    Returns:
        tuple[str, str]: A tuple containing the NEEDLE_API_KEY and OPENAI_API_KEY.
    
    Raises:
        ValueError: If either API key is missing.
    """
    NEEDLE_API_KEY = os.environ.get("NEEDLE_API_KEY")
    COLLECTION_ID = os.environ.get("COLLECTION_ID")
    
    if not NEEDLE_API_KEY:
        raise ValueError(
            "Required environment variables NEEDLE_API_KEY must be set"
        )
    
    print("✅ API keys validated.")
    return NEEDLE_API_KEY, COLLECTION_ID


def needle_func() -> NeedleLoader:
    """
    Create a NeedleLoader instance using the validated API key.
    
    Returns:
        NeedleLoader: A NeedleLoader instance.
    """
    NEEDLE_API_KEY, COLLECTION_ID = validate_api_keys()
    needle_loader=  NeedleLoader(needle_api_key=NEEDLE_API_KEY, collection_id=COLLECTION_ID)
    retriever = NeedleRetriever(needle_api_key=NEEDLE_API_KEY, collection_id=COLLECTION_ID)
    
    return needle_loader, retriever

needle_loader, retriever = needle_func()



@tool
def add_file_to_collection() -> str:
    """
    Add a file to the Needle collection.
    Replace 'docs.needle-ai.com' and the URL with your own file or website.
    """
    print("\n📥 Adding documentation to collection...")
    files = {
        "docs.needle-ai.com": "https://docs.needle-ai.com"
    }
    needle_loader.add_files(files=files)
    print("✅ File added successfully")
    return "File added successfully"


@tool
def search_collection(query: str) -> str:
    """
    Search the Needle collection using a retrieval chain.
    Returns a direct answer from the retrieved context.
    """
    print(f"\n🔍 Searching collection for: '{query}'")

    llm = ChatOpenAI(temperature=0)

    system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know, say so concisely.\n\n{context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Create a chain that retrieves documents and then answers the question
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("⚡ Executing RAG chain...")
    response = rag_chain.invoke({"input": query})
    print("✨ Search complete")

    # Return the final answer
    return str(response.get('answer', response))


#------------------------------------------------------------
# List of available tools
TOOLS: List[Callable[..., Any]] = [tavily_search_tool]
#------------------------------------------------------------



#----------------------------------------------------Address Validation Tool----------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class GoogleMapsAddressValidationInput(BaseModel):
    """
    Input schema for Google Maps Address Validation API calls.
    """
    address_lines: List[str] = Field(
        ...,
        description="The address lines, e.g. ['1600 Amphitheatre Pkwy']."
    )
    region_code: Optional[str] = Field(
        None,
        description="The country/region code, e.g. 'US'."
    )
    locality: Optional[str] = Field(
        None,
        description="City/locality name to refine the address."
    )
    enable_usps_cass: Optional[bool] = Field(
        False,
        description="Enable USPS CASS for US addresses if true."
    )


class GoogleMapsAddressValidationTool:
    """
    A tool to call the Google Maps Address Validation API by direct POST request.
    (As the googlemaps library doesn't implement it directly.)
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")

        self.endpoint = "https://addressvalidation.googleapis.com/v1:validateAddress"

    def run_address_validation(
        self,
        address_lines: List[str],
        region_code: Optional[str] = None,
        locality: Optional[str] = None,
        enable_usps_cass: bool = False
    ) -> Dict[str, Any]:
        """
        POSTs to the Address Validation API with the given parameters.
        """
        try:
            payload = {
                "address": {
                    "addressLines": address_lines
                }
            }
            if region_code:
                payload["address"]["regionCode"] = region_code
            if locality:
                payload["address"]["locality"] = locality
            if enable_usps_cass:
                payload["enableUspsCass"] = True

            params = {
                "key": self.api_key
            }
            resp = requests.post(self.endpoint, params=params, json=payload)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


# Create the LangChain Tool
address_validation_tool = Tool(
    name="Address Validation",
    func=GoogleMapsAddressValidationTool().run_address_validation,
    description="Uses Google Maps Address Validation API to validate and refine addresses.",
    args_schema=GoogleMapsAddressValidationInput
)

#----------------------------------------------------Google Maps Static Tool----------------------------------------------------
import os
import googlemaps
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple

class GoogleMapsStaticMapInput(BaseModel):
    """
    Input schema for the Google Maps Static Maps API tool.
    """
    size: Tuple[int, int] = Field(
        ...,
        description="(width, height) in pixels, e.g. (400,400)."
    )
    center: Optional[str] = Field(
        None,
        description="Center of the map as lat/lng or address. Required if no markers."
    )
    zoom: Optional[int] = Field(
        None,
        description="Zoom level from 0 (world) to 21+ (building)."
    )
    scale: Optional[int] = Field(
        1,
        description="Scale factor: 1 or 2."
    )
    format: Optional[str] = Field(
        None,
        description="Image format: png, png8, png32, gif, jpg, jpg-baseline."
    )
    maptype: Optional[str] = Field(
        None,
        description="Map type: roadmap, satellite, terrain, hybrid."
    )
    language: Optional[str] = Field(
        None,
        description="Language for map tiles."
    )
    region: Optional[str] = Field(
        None,
        description="Region code for geocoding place names."
    )
    markers: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of marker definitions. Each marker can specify color, label, etc."
    )
    path: Optional[Dict[str, Any]] = Field(
        None,
        description="Path definition including points and styling."
    )
    visible: Optional[List[str]] = Field(
        None,
        description="Locations to keep visible on the map, e.g. addresses or lat/lng."
    )

class GoogleMapsStaticMapTool:
    """
    A tool to call googlemaps.Client.static_map(...) 
    returning an iterable of image bytes.
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")
        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_static_map(
        self,
        size: Tuple[int,int],
        center: Optional[str] = None,
        zoom: Optional[int] = None,
        scale: Optional[int] = 1,
        format: Optional[str] = None,
        maptype: Optional[str] = None,
        language: Optional[str] = None,
        region: Optional[str] = None,
        markers: Optional[List[Dict[str, Any]]] = None,
        path: Optional[Dict[str, Any]] = None,
        visible: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calls googlemaps.Client.static_map(...).
        Returns content in a stream (iterable) of bytes.
        """
        try:
            # googlemaps.Client.static_map returns a requests.Response in streaming mode
            response_stream = self.gmaps.static_map(
                size=size,
                center=center,
                zoom=zoom,
                scale=scale,
                format=format,
                maptype=maptype,
                language=language,
                region=region,
                markers=markers,
                path=path,
                visible=visible
            )
            # You could read/return raw bytes, but to keep consistent we'll just
            # return them as an iterable or store them somewhere.
            return {"static_map_stream": response_stream}
        except Exception as e:
            return {"error": str(e)}

maps_static_tool = Tool(
    name="Google Maps Static Map",
    func=GoogleMapsStaticMapTool().run_static_map,
    description="Generates a static map image using the Google Maps Static API.",
    args_schema=GoogleMapsStaticMapInput
)

#---------------------------------------------------------- Roads API Tool----------------------------------------------------------
import os
import googlemaps
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class GoogleMapsRoadsInput(BaseModel):
    """
    Input schema for the Google Maps Roads API tool.
    """
    path: Optional[List[str]] = Field(
        None, 
        description="A list of lat/lng for snap_to_roads or speed limit calls."
    )
    place_ids: Optional[List[str]] = Field(
        None,
        description="Place IDs for speed_limits call."
    )
    nearest: bool = Field(
        False,
        description="Whether to call nearest_roads (true) or snap_to_roads (false by default)."
    )
    interpolate: bool = Field(
        False,
        description="If true, snap_to_roads will interpolate path."
    )
    get_speed_limits: bool = Field(
        False,
        description="If true, calls speed_limits or snapped_speed_limits."
    )


class GoogleMapsRoadsTool:
    """
    A tool to call the Google Maps Roads API via googlemaps.Client:
      - snap_to_roads, nearest_roads, speed_limits, snapped_speed_limits
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")
        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_roads(
        self,
        path: Optional[List[str]] = None,
        place_ids: Optional[List[str]] = None,
        nearest: bool = False,
        interpolate: bool = False,
        get_speed_limits: bool = False
    ) -> Dict[str, Any]:
        """
        Decides which roads method to call based on input flags.
        """
        try:
            if get_speed_limits:
                # If place_ids is given -> speed_limits
                # else if path is given -> snapped_speed_limits
                if place_ids:
                    resp = self.gmaps.speed_limits(place_ids)
                    return {"speed_limits_result": resp}
                elif path:
                    # snapped_speed_limits
                    resp = self.gmaps.snapped_speed_limits(path)
                    return {"snapped_speed_limits_result": resp}
                else:
                    return {"error": "Must provide either 'place_ids' or 'path' for speed limit queries."}
            else:
                # snap_to_roads or nearest_roads
                if not path:
                    return {"error": "No path data was provided for snap/nearest roads."}
                if nearest:
                    resp = self.gmaps.nearest_roads(path)
                    return {"nearest_roads_result": resp}
                else:
                    resp = self.gmaps.snap_to_roads(path, interpolate=interpolate)
                    return {"snap_to_roads_result": resp}
        except Exception as e:
            return {"error": str(e)}


roads_api_tool = Tool(
    name="Google Maps Roads API",
    func=GoogleMapsRoadsTool().run_roads,
    description="Calls the Google Maps Roads API for snap-to-roads, nearest-roads, and speed limits.",
    args_schema=GoogleMapsRoadsInput
)


#---------------------------------------------------------- Time Zone Tool----------------------------------------------------------
import os
import googlemaps
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union
from datetime import datetime

class GoogleMapsTimeZoneInput(BaseModel):
    """
    Input schema for the Google Maps Time Zone API tool.
    """
    location: str = Field(..., description="Lat/lng coordinate string.")
    timestamp: Optional[Union[int, datetime]] = Field(
        None,
        description="Timestamp as int or datetime. Defaults to current UTC if None."
    )
    language: Optional[str] = Field(None, description="Language for results.")

class GoogleMapsTimeZoneTool:
    """
    A tool to call the Google Maps Time Zone API via googlemaps.Client.timezone(...).
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")
        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_timezone(
        self,
        location: str,
        timestamp: Optional[Union[int, datetime]] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calls googlemaps.Client.timezone(...).
        """
        try:
            response = self.gmaps.timezone(location, timestamp=timestamp, language=language)
            return {"time_zone_result": response}
        except Exception as e:
            return {"error": str(e)}


time_zone_tool = Tool(
    name="Google Maps Time Zone",
    func=GoogleMapsTimeZoneTool().run_timezone,
    description="Calls the Google Maps Time Zone API to get time zone information for a location.",
    args_schema=GoogleMapsTimeZoneInput
)

#---------------------------------------------------------- Places Tool----------------------------------------------------------

import os
import googlemaps
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class GoogleMapsPlacesInput(BaseModel):
    """
    Input schema for the Google Maps Places API tool (search, nearby, details, etc.).
    
    Note: This is a broad input; in practice, you might split this into specialized 
          tools for find_place, text_search, nearby_search, place_details, etc.
    """
    query: Optional[str] = Field(
        None,
        description="Text query to search for, e.g. 'pizza in New York'."
    )
    location: Optional[str] = Field(
        None,
        description="Lat/lng or 'place_id:...' for nearby search or find_place bias."
    )
    radius: Optional[int] = Field(
        None,
        description="Radius in meters for nearby or text search."
    )
    type: Optional[List[str]] = Field(  # Changed to List[str]
        None,
        description="List of types of place, e.g., ['restaurant', 'museum']."
    )
    language: Optional[str] = Field(
        None,
        description="Language code for the response."
    )
    min_price: Optional[int] = Field(
        0,
        description="Minimum price range (0 to 4)."
    )
    max_price: Optional[int] = Field(
        4,
        description="Maximum price range (0 to 4)."
    )
    open_now: Optional[bool] = Field(
        False,
        description="Whether to show only places open now."
    )
    rank_by: Optional[str] = Field(
        None,
        description="For nearby search: 'prominence' or 'distance'."
    )
    name: Optional[str] = Field(
        None,
        description="A term to be matched against place names."
    )
    page_token: Optional[str] = Field(
        None,
        description="Token for pagination of results."
    )
    # Additional: for place details
    place_id: Optional[str] = Field(
        None,
        description="Place ID for retrieving details."
    )
    fields: Optional[List[str]] = Field(
        None,
        description="List of place detail fields to return."
    )


class GoogleMapsPlacesTool:
    """
    A tool to call various Google Places methods via googlemaps.Client:
      - find_place(...)
      - places(...)
      - places_nearby(...)
      - place(...)
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")
        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_places_search(
        self,
        query: Optional[str] = None,
        location: Optional[str] = None,
        radius: Optional[int] = None,
        type: Optional[str] = None,
        language: Optional[str] = None,
        min_price: Optional[int] = 0,
        max_price: Optional[int] = 4,
        open_now: Optional[bool] = False,
        rank_by: Optional[str] = None,
        name: Optional[str] = None,
        page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Example: text search (places) or nearby search if 'location' is set.
        """
        try:
            if location and rank_by == "distance":
                # places_nearby with rank_by=distance
                response = self.gmaps.places_nearby(
                    location=location,
                    keyword=query,
                    language=language,
                    min_price=min_price,
                    max_price=max_price,
                    name=name,
                    open_now=open_now,
                    rank_by=rank_by,
                    type=type,
                    page_token=page_token
                )
            elif location and radius:
                # places_nearby with normal radius
                response = self.gmaps.places_nearby(
                    location=location,
                    radius=radius,
                    keyword=query,
                    language=language,
                    min_price=min_price,
                    max_price=max_price,
                    name=name,
                    open_now=open_now,
                    type=type,
                    page_token=page_token
                )
            else:
                # fallback: text search
                response = self.gmaps.places(
                    query=query,
                    location=location,
                    radius=radius,
                    language=language,
                    min_price=min_price,
                    max_price=max_price,
                    open_now=open_now,
                    type=type,
                    region=None,
                    page_token=page_token
                )
            return {"places_search_result": response}
        except Exception as e:
            return {"error": str(e)}

    def run_find_place(
        self,
        query: str,
        input_type: str = "textquery",
        fields: Optional[List[str]] = None,
        location_bias: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Wraps googlemaps.Client.find_place(...)
        """
        try:
            response = self.gmaps.find_place(
                input=query,
                input_type=input_type,
                fields=fields,
                location_bias=location_bias,
                language=language
            )
            return {"find_place_result": response}
        except Exception as e:
            return {"error": str(e)}

    def run_place_details(
        self,
        place_id: str,
        fields: Optional[List[str]] = None,
        language: Optional[str] = None,
        reviews_no_translations: Optional[bool] = False,
        reviews_sort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Wraps googlemaps.Client.place(...)
        """
        try:
            response = self.gmaps.place(
                place_id=place_id,
                fields=fields,
                language=language,
                reviews_no_translations=reviews_no_translations,
                reviews_sort=reviews_sort
            )
            return {"place_details_result": response}
        except Exception as e:
            return {"error": str(e)}


google_places_tool = Tool(
    name="Google Maps Places API",
    func=GoogleMapsPlacesTool().run_places_search,
    description="Calls the Google Maps Places API for text search and nearby search.",
    args_schema=GoogleMapsPlacesInput
)

google_find_place_tool = Tool(
    name="Google Maps Find Place API",
    func=GoogleMapsPlacesTool().run_find_place,
    description="Calls the Google Maps Find Place API to find places by text query.",
    args_schema=GoogleMapsPlacesInput
)

google_place_details_tool = Tool(
    name="Google Maps Place Details API",
    func=GoogleMapsPlacesTool().run_place_details,
    description="Calls the Google Maps Place Details API to get detailed information about a place.",
    args_schema=GoogleMapsPlacesInput
)

#---------------------------------------------------------- Geolocation Tool----------------------------------------------------------
import os
import googlemaps
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class GoogleMapsGeolocationInput(BaseModel):
    """
    Input schema for the Google Maps Geolocation API tool.
    """
    home_mobile_country_code: Optional[str] = Field(None, description="Mobile country code (MCC).")
    home_mobile_network_code: Optional[str] = Field(None, description="Mobile network code (MNC).")
    radio_type: Optional[str] = Field(None, description="Mobile radio type, e.g. 'lte', 'gsm', etc.")
    carrier: Optional[str] = Field(None, description="Carrier name.")
    consider_ip: Optional[bool] = Field(False, description="Whether to fallback to IP geolocation.")
    cell_towers: Optional[List[Dict[str, Any]]] = Field(None, description="List of cell tower objects.")
    wifi_access_points: Optional[List[Dict[str, Any]]] = Field(None, description="List of Wi-Fi AP objects.")

class GoogleMapsGeolocationTool:
    """
    A tool to call the Google Maps Geolocation API via googlemaps.Client.geolocate().
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")
        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_geolocate(
        self,
        home_mobile_country_code: Optional[str] = None,
        home_mobile_network_code: Optional[str] = None,
        radio_type: Optional[str] = None,
        carrier: Optional[str] = None,
        consider_ip: Optional[bool] = False,
        cell_towers: Optional[List[Dict[str,Any]]] = None,
        wifi_access_points: Optional[List[Dict[str,Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calls googlemaps.Client.geolocate(...).
        """
        try:
            response = self.gmaps.geolocate(
                home_mobile_country_code=home_mobile_country_code,
                home_mobile_network_code=home_mobile_network_code,
                radio_type=radio_type,
                carrier=carrier,
                consider_ip=consider_ip,
                cell_towers=cell_towers,
                wifi_access_points=wifi_access_points
            )
            return {"geolocation_result": response}
        except Exception as e:
            return {"error": str(e)}

geolocation_tool = Tool(
    name="Google Maps Geolocation",
    func=GoogleMapsGeolocationTool().run_geolocate,
    description="Calls the Google Maps Geolocation API to estimate the device's location.",
    args_schema=GoogleMapsGeolocationInput
)


#---------------------------------------------------------- Geocoding Tool----------------------------------------------------------
import os
import googlemaps
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class GoogleMapsGeocodingInput(BaseModel):
    """
    Input schema for the Google Maps Geocoding API tool.
    """
    address: Optional[str] = Field(
        None,
        description="The street address or query to geocode."
    )
    place_id: Optional[str] = Field(
        None,
        description="A place ID to look up. (If used, 'address' can be omitted.)"
    )
    components: Optional[Dict[str, str]] = Field(
        None,
        description="A component filter, e.g. {'country': 'US','postal_code': '94043'}."
    )
    bounds: Optional[Dict[str, tuple]] = Field(
        None,
        description="A dict with 'southwest':(lat,lng) and 'northeast':(lat,lng) to bias results."
    )
    region: Optional[str] = Field(
        None,
        description="Region code for location biasing, e.g. 'us'."
    )
    language: Optional[str] = Field(
        None,
        description="Language for results."
    )


class GoogleMapsGeocodingTool:
    """
    A tool to call the Google Maps Geocoding API via googlemaps.Client.
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")
        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_geocode(
        self,
        address: Optional[str] = None,
        place_id: Optional[str] = None,
        components: Optional[Dict[str, str]] = None,
        bounds: Optional[Dict[str, tuple]] = None,
        region: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calls googlemaps.Client.geocode(...).
        """
        try:
            if place_id:
                # Use place_id priority if present
                response = self.gmaps.geocode(place_id=place_id)
                return {"geocode_result": response}
            else:
                response = self.gmaps.geocode(
                    address=address,
                    components=components,
                    bounds=bounds,
                    region=region,
                    language=language
                )
                return {"geocode_result": response}
        except Exception as e:
            return {"error": str(e)}

geocoding_tool = Tool(
    name="Google Maps Geocoding",
    func=GoogleMapsGeocodingTool().run_geocode,
    description="Calls the Google Maps Geocoding API to convert addresses to geographic coordinates.",
    args_schema=GoogleMapsGeocodingInput
)

#---------------------------------------------------------- Elevation Tool----------------------------------------------------------
import os
import googlemaps
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class GoogleMapsElevationInput(BaseModel):
    """
    Input schema for the Google Maps Elevation API tool.
    """
    locations: List[str] = Field(
        ...,
        description="List of (lat,lng) pairs or place IDs for elevation data."
    )
    samples: Optional[int] = Field(
        None,
        description="Number of sample points (only used with elevation_along_path)."
    )
    use_path: Optional[bool] = Field(
        False,
        description="If true, calls elevation_along_path (requires 2+ points)."
    )


class GoogleMapsElevationTool:
    """
    A tool to call the Google Maps Elevation API via googlemaps.Client.
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")
        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_elevation(
        self,
        locations: List[str],
        samples: Optional[int] = None,
        use_path: bool = False
    ) -> Dict[str, Any]:
        """
        Calls either `elevation` or `elevation_along_path` from googlemaps.Client.
        """
        try:
            if use_path:
                # For elevation_along_path, need 2 or more points and a 'samples' value.
                if len(locations) < 2:
                    return {"error": "elevation_along_path requires at least 2 locations."}
                if not samples:
                    return {"error": "Must provide 'samples' for elevation_along_path."}
                response = self.gmaps.elevation_along_path(locations, samples)
                return {"elevation_along_path_result": response}
            else:
                response = self.gmaps.elevation(locations)
                return {"elevation_result": response}
        except Exception as e:
            return {"error": str(e)}


elevation_tool = Tool(
    name="Google Maps Elevation",
    func=GoogleMapsElevationTool().run_elevation,
    description="Calls the Google Maps Elevation API to get elevation data for locations.",
    args_schema=GoogleMapsElevationInput
)


#---------------------------------------------------------- Distance Matrix Tool----------------------------------------------------------
import os
import googlemaps
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict, Union

class GoogleMapsDistanceMatrixInput(BaseModel):
    """
    Input schema for the Google Maps Distance Matrix API tool.
    """
    origins: List[str] = Field(
        ..., 
        description="One or more starting points (address or lat/lng)."
    )
    destinations: List[str] = Field(
        ..., 
        description="One or more ending points (address or lat/lng)."
    )
    mode: Optional[str] = Field(
        None,
        description="Travel mode: driving, walking, bicycling, or transit."
    )
    language: Optional[str] = Field(
        None,
        description="Language for the results."
    )
    avoid: Optional[str] = Field(
        None,
        description="Features to avoid: tolls, highways, or ferries."
    )
    units: Optional[str] = Field(
        None,
        description="Units of measurement: metric or imperial."
    )
    departure_time: Optional[Any] = Field(
        None,
        description="Desired time of departure (timestamp or datetime)."
    )
    arrival_time: Optional[Any] = Field(
        None,
        description="Desired time of arrival (timestamp or datetime)."
    )
    transit_mode: Optional[List[str]] = Field(
        None,
        description="For transit: e.g. ['bus','train']."
    )
    transit_routing_preference: Optional[str] = Field(
        None,
        description="For transit: 'less_walking' or 'fewer_transfers'."
    )
    traffic_model: Optional[str] = Field(
        None,
        description="For driving+departure_time: 'best_guess','optimistic','pessimistic'."
    )
    region: Optional[str] = Field(
        None,
        description="Region code for biasing geocoding, e.g., 'us'."
    )


class GoogleMapsDistanceMatrixTool:
    """
    A tool to call the Google Maps Distance Matrix API via googlemaps.Client.
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")

        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_distance_matrix(
        self,
        origins: List[str],
        destinations: List[str],
        mode: Optional[str] = None,
        language: Optional[str] = None,
        avoid: Optional[str] = None,
        units: Optional[str] = None,
        departure_time: Optional[Any] = None,
        arrival_time: Optional[Any] = None,
        transit_mode: Optional[List[str]] = None,
        transit_routing_preference: Optional[str] = None,
        traffic_model: Optional[str] = None,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calls the Distance Matrix API using googlemaps.Client.distance_matrix(...).
        """
        try:
            response = self.gmaps.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode=mode,
                language=language,
                avoid=avoid,
                units=units,
                departure_time=departure_time,
                arrival_time=arrival_time,
                transit_mode=transit_mode,
                transit_routing_preference=transit_routing_preference,
                traffic_model=traffic_model,
                region=region
            )
            return {"distance_matrix_result": response}
        except Exception as e:
            return {"error": str(e)}


distance_matrix_tool = Tool(
    name="Google Maps Distance Matrix",
    func=GoogleMapsDistanceMatrixTool().run_distance_matrix,
    description="Calls the Google Maps Distance Matrix API to get travel distance and time data.",
    args_schema=GoogleMapsDistanceMatrixInput
)

#---------------------------------------------------------- Directions Tool----------------------------------------------------------
import os
import googlemaps
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

class GoogleMapsDirectionsInput(BaseModel):
    """
    Input schema for the Google Maps Directions API tool.
    """
    origin: str = Field(..., description="The starting address or lat/lng.")
    destination: str = Field(..., description="The ending address or lat/lng.")
    mode: Optional[str] = Field(
        None,
        description="Travel mode: driving, walking, bicycling, or transit."
    )
    waypoints: Optional[List[str]] = Field(
        None,
        description="An optional list of waypoints or via points in the route."
    )
    optimize_waypoints: Optional[bool] = Field(
        False,
        description="If true, re-order waypoints to optimize the route."
    )
    avoid: Optional[List[str]] = Field(
        None,
        description="Features to avoid: tolls, highways, or ferries."
    )
    language: Optional[str] = Field(
        None,
        description="Language for the directions text."
    )
    units: Optional[str] = Field(
        None,
        description="Units of measurement: metric or imperial."
    )
    region: Optional[str] = Field(
        None,
        description="Region code for biasing results, e.g., 'us'."
    )
    departure_time: Optional[Any] = Field(
        None,
        description="Desired time of departure (int UNIX timestamp or datetime)."
    )
    arrival_time: Optional[Any] = Field(
        None,
        description="Desired time of arrival (int UNIX timestamp or datetime) [transit only]."
    )
    transit_mode: Optional[List[str]] = Field(
        None,
        description="For transit mode, e.g. ['bus','train']."
    )
    transit_routing_preference: Optional[str] = Field(
        None,
        description="For transit: 'less_walking' or 'fewer_transfers'."
    )
    traffic_model: Optional[str] = Field(
        None,
        description="For driving+departure_time: 'best_guess','optimistic','pessimistic'."
    )
    alternatives: Optional[bool] = Field(
        False,
        description="Request multiple route alternatives."
    )


class GoogleMapsDirectionsTool:
    """
    A tool to call the Google Maps Directions API via googlemaps.Client.
    """
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")

        self.gmaps = googlemaps.Client(key=self.api_key)

    def run_directions(
        self,
        origin: str,
        destination: str,
        mode: Optional[str] = None,
        waypoints: Optional[List[str]] = None,
        optimize_waypoints: bool = False,
        avoid: Optional[List[str]] = None,
        language: Optional[str] = None,
        units: Optional[str] = None,
        region: Optional[str] = None,
        departure_time: Optional[Any] = None,
        arrival_time: Optional[Any] = None,
        transit_mode: Optional[List[str]] = None,
        transit_routing_preference: Optional[str] = None,
        traffic_model: Optional[str] = None,
        alternatives: bool = False
    ) -> Dict[str, Any]:
        """
        Calls the Directions API using googlemaps.Client.directions(...).
        """
        try:
            response = self.gmaps.directions(
                origin=origin,
                destination=destination,
                mode=mode,
                waypoints=waypoints if waypoints else None,
                optimize_waypoints=optimize_waypoints,
                avoid=avoid,
                language=language,
                units=units,
                region=region,
                departure_time=departure_time,
                arrival_time=arrival_time,
                transit_mode=transit_mode,
                transit_routing_preference=transit_routing_preference,
                traffic_model=traffic_model,
                alternatives=alternatives
            )
            return {"directions_result": response}
        except Exception as e:
            return {"error": str(e)}


directions_tool = Tool(
    name="Google Maps Directions",
    func=GoogleMapsDirectionsTool().run_directions,
    description="Calls the Google Maps Directions API to get travel directions.",
    args_schema=GoogleMapsDirectionsInput
)


#---------------------------------------------------------- Yelp Business Search Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class YelpBusinessSearchInput(BaseModel):
    """
    Input schema for the Yelp Business Search endpoint.
    See: https://www.yelp.com/developers/documentation/v3/business_search
    """
    term: Optional[str] = Field(None, description="Search term (e.g. 'food', 'restaurants').")
    location: Optional[str] = Field(None, description="Location by address, city, zip code, etc.")
    latitude: Optional[float] = Field(None, description="Latitude for geosearch (if used).")
    longitude: Optional[float] = Field(None, description="Longitude for geosearch (if used).")
    limit: Optional[int] = Field(50, description="Max number of results to return, up to 50.")
    offset: Optional[int] = Field(0, description="Offset the list of returned results.")
    radius: Optional[int] = Field(None, description="Radius in meters, maximum 40000 (25 miles).")
    categories: Optional[str] = Field(None, description="Category filters, e.g. 'bars, pizza'.")
    open_now: Optional[bool] = Field(False, description="Only return businesses open now if True.")
    price: Optional[str] = Field(None, description="Price levels to filter by, e.g. '1,2,3'.")

class YelpBusinessSearchTool:
    """
    A tool to call the Yelp Business Search endpoint.
    """
    def __init__(self):
        self.api_key = os.getenv("YELP_API_KEY")
        if not self.api_key:
            raise ValueError("YELP_API_KEY environment variable is missing.")
        
        self.endpoint = "https://api.yelp.com/v3/businesses/search"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def run_business_search(
        self,
        term: Optional[str] = None,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        limit: Optional[int] = 50,
        offset: Optional[int] = 0,
        radius: Optional[int] = None,
        categories: Optional[str] = None,
        open_now: bool = False,
        price: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Makes a GET request to the Yelp /businesses/search endpoint with the provided parameters.
        """
        try:
            params = {}
            if term: params["term"] = term
            if location: params["location"] = location
            if latitude is not None: params["latitude"] = latitude
            if longitude is not None: params["longitude"] = longitude
            if limit: params["limit"] = limit
            if offset: params["offset"] = offset
            if radius: params["radius"] = radius
            if categories: params["categories"] = categories
            if open_now: params["open_now"] = "true"
            if price: params["price"] = price

            response = requests.get(url=self.endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


yelp_business_search_tool = Tool(
    name="Yelp Business Search",
    func=YelpBusinessSearchTool().run_business_search,
    description="Calls the Yelp Business Search endpoint to find businesses.",
    args_schema=YelpBusinessSearchInput
)


#---------------------------------------------------------- Yelp Phone Search Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Dict, Any

class YelpPhoneSearchInput(BaseModel):
    """
    Input schema for the Yelp Phone Search endpoint.
    See: https://www.yelp.com/developers/documentation/v3/business_search_phone
    """
    phone: str = Field(..., description="Phone number with country code, e.g. '+18584340001'.")
    country_code: str = Field("US", description="Country code if needed.")

class YelpPhoneSearchTool:
    """
    A tool to call the Yelp Phone Search endpoint.
    """
    def __init__(self):
        self.api_key = os.getenv("YELP_API_KEY")
        if not self.api_key:
            raise ValueError("YELP_API_KEY environment variable is missing.")
        
        self.endpoint = "https://api.yelp.com/v3/businesses/search/phone"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

    def run_phone_search(
        self,
        phone: str,
        country_code: str = "US"
    ) -> Dict[str, Any]:
        """
        Makes a GET request to the Yelp /businesses/search/phone endpoint.
        The 'phone' must start with '+' and country code.
        """
        try:
            params = {"phone": phone}
            response = requests.get(url=self.endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


yelp_phone_search_tool = Tool(
    name="Yelp Phone Search",
    func=YelpPhoneSearchTool().run_phone_search,
    description="Calls the Yelp Phone Search endpoint to find businesses by phone number.",
    args_schema=YelpPhoneSearchInput
)

#---------------------------------------------------------- Yelp Business Details Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Dict, Any

class YelpBusinessDetailsInput(BaseModel):
    """
    Input schema for the Yelp Business Details endpoint.
    https://www.yelp.com/developers/documentation/v3/business
    """
    business_id: str = Field(..., description="The Yelp business ID to look up.")

class YelpBusinessDetailsTool:
    """
    A tool to call the Yelp Business Details endpoint: /businesses/{id}
    """
    def __init__(self):
        self.api_key = os.getenv("YELP_API_KEY")
        if not self.api_key:
            raise ValueError("YELP_API_KEY environment variable is missing.")

        self.base_url = "https://api.yelp.com/v3/businesses"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

    def run_business_details(self, business_id: str) -> Dict[str, Any]:
        """
        Makes a GET request to /businesses/{business_id} to get business details.
        """
        try:
            url = f"{self.base_url}/{business_id}"
            response = requests.get(url=url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


yelp_business_details_tool = Tool(
    name="Yelp Business Details",
    func=YelpBusinessDetailsTool().run_business_details,
    description="Calls the Yelp Business Details endpoint to get information about a business.",
    args_schema=YelpBusinessDetailsInput
)

#---------------------------------------------------------- Yelp Business Reviews Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Dict, Any

class YelpBusinessReviewsInput(BaseModel):
    """
    Input schema for the Yelp Business Reviews endpoint.
    https://www.yelp.com/developers/documentation/v3/business_reviews
    """
    business_id: str = Field(..., description="The Yelp business ID to fetch reviews for.")

class YelpBusinessReviewsTool:
    """
    A tool to call the Yelp Business Reviews endpoint: /businesses/{id}/reviews
    """
    def __init__(self):
        self.api_key = os.getenv("YELP_API_KEY")
        if not self.api_key:
            raise ValueError("YELP_API_KEY environment variable is missing.")

        self.base_url = "https://api.yelp.com/v3/businesses"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

    def run_business_reviews(self, business_id: str) -> Dict[str, Any]:
        """
        Makes a GET request to /businesses/{business_id}/reviews to get reviews data.
        """
        try:
            url = f"{self.base_url}/{business_id}/reviews"
            response = requests.get(url=url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


yelp_business_reviews_tool = Tool(
    name="Yelp Business Reviews",
    func=YelpBusinessReviewsTool().run_business_reviews,
    description="Calls the Yelp Business Reviews endpoint to get reviews for a business.",
    args_schema=YelpBusinessReviewsInput
)

#---------------------------------------------------------- Yelp Events Search Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class YelpEventsSearchInput(BaseModel):
    """
    Input schema for the Yelp Events search endpoint.
    See: https://www.yelp.com/developers/documentation/v3/events
    """
    location: Optional[str] = Field(None, description="Location for event search.")
    limit: Optional[int] = Field(20, description="Max results, default 20, up to 50.")
    sort_by: Optional[str] = Field(None, description="Sort by 'time_start' or 'desc'.")
    start_date: Optional[int] = Field(None, description="Unix start date.")
    end_date: Optional[int] = Field(None, description="Unix end date.")
    # More parameters can be added as needed

class YelpEventsSearchTool:
    """
    A tool to call the Yelp Events search endpoint: /events
    """
    def __init__(self):
        self.api_key = os.getenv("YELP_API_KEY")
        if not self.api_key:
            raise ValueError("YELP_API_KEY environment variable is missing.")

        self.endpoint = "https://api.yelp.com/v3/events"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

    def run_events_search(
        self,
        location: Optional[str] = None,
        limit: Optional[int] = 20,
        sort_by: Optional[str] = None,
        start_date: Optional[int] = None,
        end_date: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        GET request to /events with optional location, date range, etc.
        """
        try:
            params = {}
            if location: params["location"] = location
            if limit: params["limit"] = limit
            if sort_by: params["sort_by"] = sort_by
            if start_date: params["start_date"] = start_date
            if end_date: params["end_date"] = end_date

            response = requests.get(url=self.endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


yelp_events_search_tool = Tool(
    name="Yelp Events Search",
    func=YelpEventsSearchTool().run_events_search,
    description="Calls the Yelp Events search endpoint to find local events.",
    args_schema=YelpEventsSearchInput
)

#---------------------------------------------------------- Yelp GraphQL Tool----------------------------------------------------------
import os
import requests
import json
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class YelpGraphQLInput(BaseModel):
    """
    Input schema for querying Yelp's GraphQL endpoint.
    See: https://www.yelp.com/developers/graphql/guides/intro
    """
    query: str = Field(..., description="The raw GraphQL query string to execute.")

class YelpGraphQLTool:
    """
    A tool that hits the Yelp GraphQL endpoint with a user-provided query.
    """
    def __init__(self):
        self.api_key = os.getenv("YELP_API_KEY")
        if not self.api_key:
            raise ValueError("YELP_API_KEY environment variable is missing.")
        self.url = "https://api.yelp.com/v3/graphql"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def run_graphql_query(self, query: str) -> Dict[str, Any]:
        """
        POSTs the GraphQL query to Yelp's GraphQL endpoint.
        """
        try:
            payload = {"query": query}
            resp = requests.post(self.url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


yelp_graphql_tool = Tool(
    name="Yelp GraphQL",
    func=YelpGraphQLTool().run_graphql_query,
    description="Calls the Yelp GraphQL endpoint with a user-provided query.",
    args_schema=YelpGraphQLInput
)

#---------------------------------------------------------- YouTube Search Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class YouTubeSearchInput(BaseModel):
    """
    Input schema for the YouTube Data API's 'search' endpoint.
    See: https://developers.google.com/youtube/v3/docs/search/list
    """
    q: str = Field(..., description="Search query term.")
    part: str = Field("snippet", description="Comma-separated list of search resource parts.")
    maxResults: Optional[int] = Field(25, description="Max results (0-50).")
    type: Optional[str] = Field(None, description="Restrict search to a type: channel, playlist, video.")
    order: Optional[str] = Field(None, description="Order results: date, rating, relevance, etc.")
    pageToken: Optional[str] = Field(None, description="Token for paginated results.")


class YouTubeSearchTool:
    """
    A tool to call the YouTube Data API /search endpoint.
    """
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable is missing.")

        self.endpoint = "https://www.googleapis.com/youtube/v3/search"

    def run_search(
        self,
        q: str,
        part: str = "snippet",
        maxResults: int = 25,
        type: Optional[str] = None,
        order: Optional[str] = None,
        pageToken: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        GET request to the YouTube /search endpoint.
        """
        try:
            params = {
                "key": self.api_key,
                "q": q,
                "part": part,
                "maxResults": maxResults,
            }
            if type: params["type"] = type
            if order: params["order"] = order
            if pageToken: params["pageToken"] = pageToken

            resp = requests.get(url=self.endpoint, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


youtube_search_tool = Tool(
    name="YouTube Search",
    func=YouTubeSearchTool().run_search,
    description="Calls the YouTube Data API's 'search' endpoint to find videos.",
    args_schema=YouTubeSearchInput
)

#---------------------------------------------------------- YouTube Videos Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class YouTubeVideosInput(BaseModel):
    """
    Input schema for the YouTube Data API's 'videos' endpoint.
    See: https://developers.google.com/youtube/v3/docs/videos/list
    """
    video_id: str = Field(..., description="The ID of the YouTube video (e.g. 'qc4yoUqpwEw').")
    part: str = Field("snippet", description="Which resource parts to retrieve, e.g. 'snippet,contentDetails'.")

class YouTubeVideosTool:
    """
    A tool to call the YouTube Data API /videos endpoint for retrieving video details.
    """
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable is missing.")

        self.endpoint = "https://www.googleapis.com/youtube/v3/videos"

    def run_videos(
        self,
        video_id: str,
        part: str = "snippet"
    ) -> Dict[str, Any]:
        """
        GET request to the YouTube /videos endpoint.
        """
        try:
            params = {
                "key": self.api_key,
                "id": video_id,
                "part": part
            }
            resp = requests.get(url=self.endpoint, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


youtube_videos_tool = Tool(
    name="YouTube Videos",
    func=YouTubeVideosTool().run_videos,
    description="Calls the YouTube Data API's 'videos' endpoint to get video details.",
    args_schema=YouTubeVideosInput
)

#---------------------------------------------------------- YouTube CommentThreads Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class YouTubeCommentThreadsInput(BaseModel):
    """
    Input schema for the YouTube Data API's 'commentThreads' endpoint.
    See: https://developers.google.com/youtube/v3/docs/commentThreads/list
    """
    part: str = Field("snippet,replies", description="Resource parts to include.")
    allThreadsRelatedToChannelId: Optional[str] = Field(
        None,
        description="Channel ID to get all comment threads for that channel."
    )
    videoId: Optional[str] = Field(
        None,
        description="If retrieving comment threads for a specific video."
    )
    maxResults: Optional[int] = Field(20, description="Max results per page, up to 100.")
    pageToken: Optional[str] = Field(None, description="For paginated results.")


class YouTubeCommentThreadsTool:
    """
    A tool to call the YouTube Data API /commentThreads endpoint.
    """
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable is missing.")

        self.endpoint = "https://www.googleapis.com/youtube/v3/commentThreads"

    def run_comment_threads(
        self,
        part: str = "snippet,replies",
        allThreadsRelatedToChannelId: Optional[str] = None,
        videoId: Optional[str] = None,
        maxResults: int = 20,
        pageToken: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        GET request to /commentThreads with optional channel or video.
        """
        try:
            params = {
                "key": self.api_key,
                "part": part,
                "maxResults": maxResults
            }
            if allThreadsRelatedToChannelId:
                params["allThreadsRelatedToChannelId"] = allThreadsRelatedToChannelId
            if videoId:
                params["videoId"] = videoId
            if pageToken:
                params["pageToken"] = pageToken

            resp = requests.get(url=self.endpoint, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

youtube_comment_threads_tool = Tool(
    name="YouTube CommentThreads",
    func=YouTubeCommentThreadsTool().run_comment_threads,
    description="Calls the YouTube Data API's 'commentThreads' endpoint to get video comments.",
    args_schema=YouTubeCommentThreadsInput
)

#---------------------------------------------------------- YouTube PlaylistItems Tool----------------------------------------------------------
import os
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class YouTubePlaylistItemsInput(BaseModel):
    """
    Input schema for YouTube Data API's 'playlistItems' endpoint.
    See: https://developers.google.com/youtube/v3/docs/playlistItems/list
    """
    playlistId: str = Field(..., description="ID of the playlist to fetch items from.")
    part: str = Field("snippet,contentDetails,status", description="Which resource parts to retrieve.")
    maxResults: Optional[int] = Field(5, description="Max results, up to 50.")
    pageToken: Optional[str] = Field(None, description="For paginated results.")

class YouTubePlaylistItemsTool:
    """
    A tool to call the YouTube Data API /playlistItems endpoint.
    """
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable is missing.")

        self.endpoint = "https://www.googleapis.com/youtube/v3/playlistItems"

    def run_playlist_items(
        self,
        playlistId: str,
        part: str = "snippet,contentDetails,status",
        maxResults: int = 5,
        pageToken: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        GET request to /playlistItems with a given playlistId.
        """
        try:
            params = {
                "key": self.api_key,
                "playlistId": playlistId,
                "part": part,
                "maxResults": maxResults
            }
            if pageToken:
                params["pageToken"] = pageToken

            resp = requests.get(url=self.endpoint, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}


youtube_playlist_items_tool = Tool(
    name="YouTube PlaylistItems",
    func=YouTubePlaylistItemsTool().run_playlist_items,
    description="Calls the YouTube Data API's 'playlistItems' endpoint to get playlist items.",
    args_schema=YouTubePlaylistItemsInput
)



#---------------------------------------------------------- Tools Condition ----------------------------------------------------------

def flight_tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["flight_tools", "accomodation_node"]:
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "flight_tools"
    return "accomodation_node"    # you can change this to any other node name instead of "__end__"


def accomodation_tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["accomodation_tools", "activity_planner"]:
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "accomodation_tools"
    return "activity_planner"    # you can change this to any other node name instead of "__end__"


def activity_planner_tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["activity_planner_tools", "realtime_provider"]:
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "activity_planner_tools"
    return "realtime_provider"    # you can change this to any other node name instead of "__end__"



#---------------------------------------------------------- Docling Tools ----------------------------------------------------------

import time
import os
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Base Input Schema
class DoclingToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the PDF document")
    max_threads: Optional[int] = Field(default=8, description="Number of processing threads")

# Core PDF Processor
class DoclingPDFProcessor:
    def __init__(self, **processor_options):
        self.processor_options = processor_options
        
    def process_pdf(self, file_path: str, max_threads: int = 8) -> Iterator[dict]:
        """Core PDF processing pipeline"""
        try:
            print(f"🔍 Processing {os.path.basename(file_path)}")
            start_time = time.time()
            
            # Configure processing pipeline
            accelerator = AcceleratorOptions(
                num_threads=max_threads,
                device=AcceleratorDevice.AUTO
            )
            
            pipeline_opts = PdfPipelineOptions(
                **self.processor_options,
                accelerator_options=accelerator
            )
            
            # Execute conversion
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_opts
                    )
                }
            )
            
            docling_doc = converter.convert(file_path).document
            processing_time = time.time() - start_time
            
            yield {
                "content": docling_doc.export_to_markdown(),
                "metadata": {
                    "source": file_path,
                    "processing_time": round(processing_time, 2),
                    "options": self.processor_options
                }
            }
            
        except Exception as e:
            yield {"error": str(e), "file": file_path}

# Specific Tool Implementations
class DoclingTextExtractorTool(DoclingPDFProcessor):
    """Basic text extraction with OCR fallback"""
    def __init__(self):
        super().__init__(
            do_ocr=True,
            do_table_structure=False,
            table_structure_options={"do_cell_matching": False}
        )

class DoclingTableExtractorTool(DoclingPDFProcessor):
    """Structured table extraction"""
    def __init__(self):
        super().__init__(
            do_ocr=False,
            do_table_structure=True,
            table_structure_options={"do_cell_matching": True}
        )

class DoclingFullProcessingTool(DoclingPDFProcessor):
    """Comprehensive processing with OCR and tables"""
    def __init__(self):
        super().__init__(
            do_ocr=True,
            do_table_structure=True,
            table_structure_options={"do_cell_matching": True}
        )

# Tool Factory
def create_docling_tool(tool_class, name: str, description: str) -> Tool:
    class _ToolInput(DoclingToolInput):
        pass
    
    processor = tool_class()
    
    return Tool(
        name=name,
        description=description,
        args_schema=_ToolInput,
        func=lambda params: list(processor.process_pdf(
            params["file_path"],
            params.get("max_threads", 8)
        ))
    )

# Create LangChain Tools
text_extractor_tool = create_docling_tool(
    DoclingTextExtractorTool,
    name="docling_text_extractor",
    description="Extracts text from PDFs with OCR fallback"
)

table_extractor_tool = create_docling_tool(
    DoclingTableExtractorTool,
    name="docling_table_extractor",
    description="Extracts structured tables from PDF documents"
)

full_processor_tool = create_docling_tool(
    DoclingFullProcessingTool,
    name="docling_full_processor",
    description="Comprehensive PDF processing with text, OCR, and table extraction"
)


#---------------------------------------------------------- Google Events Tools ----------------------------------------------------------

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import os
import requests

# Define Input Schema
# Define Input Schema
class GoogleEventsSearchInput(BaseModel):
    query: str = Field(..., description="The search query for Google Events (e.g., 'Events in Austin').")
    location: Optional[str] = Field(None, description="The geographic location for the search (e.g., 'Austin, TX').")
    uule: Optional[str] = Field(None, description="The Google encoded location for the search.")
    gl: Optional[str] = Field(None, description="The country code for localization (e.g., 'us' for the United States).")
    hl: Optional[str] = Field(None, description="The language code for localization (e.g., 'en' for English).")
    start: Optional[int] = Field(None, description="The result offset for pagination (e.g., 0 for the first page).")
    htichips: Optional[str] = Field(None, description="Advanced filters for events (parameters include date:today, date:tomorrow, date:week, date:today, date:next_week, date:month, date:next_month, event_type:Virtual-Event (e.g., 'event_type:Virtual-Event,date:today').")
    no_cache: Optional[bool] = Field(None, description="Disallow results from the cache if set to True.")
    async_: Optional[bool] = Field(None, description="Submit search asynchronously if set to True.")
    zero_trace: Optional[bool] = Field(None, description="Enable ZeroTrace mode (Enterprise only).")
    output: Optional[str] = Field(None, description="The output format (e.g., 'json' or 'html').")

# Define the Tool
class GoogleEventsSearchTool:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")
        self.base_url = "https://serpapi.com/search"
        if not self.api_key:
            raise ValueError("SerpApi API key is missing. Please set the SERPAPI_API_KEY environment variable.")

    def search_events(
        self,
        input: GoogleEventsSearchInput,
    ) -> Dict[str, Any]:
        """
        Search Google Events using the SerpApi service.

        Args:
            query: The search query.
            location: The geographic location for the search.
            uule: The Google encoded location for the search.
            gl: The country code for localization.
            hl: The language code for localization.
            start: The result offset for pagination.
            htichips: Advanced filters for events.
            no_cache: Disallow results from the cache.
            async_: Submit search asynchronously.
            zero_trace: Enable ZeroTrace mode (Enterprise only).
            output: The output format (e.g., 'json' or 'html').

        Returns:
            A dictionary containing the search results.
        """
        try:
            # Prepare the parameters for the API request
            params = {
                "engine": "google_events",
                "q": input.query,
                "api_key": self.api_key,
                "output": input.output or "json",
            }

            # Add optional parameters if provided
            if input.location:
                params["location"] = input.location
            if input.uule:
                params["uule"] = input.uule
            if input.gl:
                params["gl"] = input.gl
            if input.hl:
                params["hl"] = input.hl
            if input.start:
                params["start"] = input.start
            if input.htichips:
                params["htichips"] = input.htichips
            if input.no_cache:
                params["no_cache"] = "true" if input.no_cache else "false"
            if input.async_:
                params["async"] = "true" if input.async_ else "false"
            if input.zero_trace:
                params["zero_trace"] = "true" if input.zero_trace else "false"

            # Make the API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the JSON response
            results = response.json()

            # Extract relevant information from the results
            events_results = results.get("events_results", [])
            simplified_results = []
            for event in events_results:
                simplified_results.append({
                    "title": event.get("title"),
                    "date": event.get("date", {}).get("when"),
                    "address": event.get("address", []),
                    "link": event.get("link"),
                    "description": event.get("description"),
                    "ticket_info": event.get("ticket_info", []),
                    "venue": event.get("venue", {}).get("name"),
                    "thumbnail": event.get("thumbnail"),
                })

            return {
                "search_metadata": results.get("search_metadata", {}),
                "events_results": simplified_results,
                "pagination": results.get("pagination", {}),
            }

        except requests.exceptions.RequestException as e:
            return {"error": f"An error occurred while making the API request: {str(e)}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}

google_events_tool = Tool(
    name="Google Events Search",
    func=GoogleEventsSearchTool().search_events,
    description="Provides event results from Google Events, including titles, dates, addresses, links, descriptions, ticket info, venues, and thumbnails.",
    args_schema=GoogleEventsSearchInput
)


#---------------------------------------------------------- TicketMaster ----------------------------------------------------------
# Define the input schema for the Ticketmaster API tool
class TicketmasterEventSearchInput(BaseModel):
    keyword: Optional[str] = Field(default=None, description="Keyword to search for events (e.g., artist, event name).")
    city: Optional[str] = Field(default=None, description="Filter events by city.")
    country_code: Optional[str] = Field(default=None, description="Filter events by country code (ISO Alpha-2 Code).")
    classification_name: Optional[str] = Field(default=None, description="Filter by classification (e.g., 'Music').")
    start_date_time: Optional[str] = Field(default=None, description="Start date filter in ISO8601 format (YYYY-MM-DDTHH:mm:ssZ).")
    end_date_time: Optional[str] = Field(default=None, description="End date filter in ISO8601 format (YYYY-MM-DDTHH:mm:ssZ).")
    size: int = Field(default=10, description="Number of events to return per page.")
    page: int = Field(default=0, description="Page number to retrieve.")
    sort: Optional[str] = Field(default="relevance,desc", description="Sorting order of the search results.")

    @validator('start_date_time', 'end_date_time')
    def validate_date_format(cls, v):
        if v and "T" not in v:
            raise ValueError("Datetime must be in ISO8601 format (e.g., 'YYYY-MM-DDTHH:mm:ssZ').")
        return v

# Define the Ticketmaster API tool
class TicketmasterAPITool:
    def __init__(self):
        self.base_url = "https://app.ticketmaster.com/discovery/v2"
        self.api_key = os.getenv("TICKETMASTER_API_KEY")  # Set your Ticketmaster API key here
        if not self.api_key:
            raise ValueError("Ticketmaster API key is missing. Please set TICKETMASTER_API_KEY environment variable.")

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Helper method to make API requests to Ticketmaster.
        """
        params = params or {}
        params["apikey"] = self.api_key  # Add the API key to the request parameters
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_message = e.response.json().get("fault", {}).get("faultstring", "HTTP error occurred")
            raise ValueError(f"HTTP error: {error_message}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")

    def search_events(
        self,
        input: TicketmasterEventSearchInput,
    ) -> str:
        """
        Search for events using the Ticketmaster Discovery API.
        """
        params = {
            "keyword": input.keyword,
            "city": input.city,
            "countryCode": input.country_code,
            "classificationName": input.classification_name,
            "startDateTime": input.start_date_time,
            "endDateTime": input.end_date_time,
            "size": input.size,
            "page": input.page,
            "sort": input.sort,
        }
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}

        try:
            data = self._make_request("events.json", params=params)
            events = data.get("_embedded", {}).get("events", [])
            results = []
            for event in events:
                event_details = {
                    "Event": event.get("name"),
                    "Date": event.get("dates", {}).get("start", {}).get("localDate"),
                    "Time": event.get("dates", {}).get("start", {}).get("localTime"),
                    "Venue": event["_embedded"]["venues"][0].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                    "City": event["_embedded"]["venues"][0]["city"].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                    "Country": event["_embedded"]["venues"][0]["country"].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                    "Url": event.get("url"),
                }
                results.append(event_details)
            return results
        except Exception as e:
            raise Exception(f"An error occurred while searching for events: {str(e)}")

    def get_event_details(self, event_id: str) -> Dict[str, Any]:
        """
        Retrieve details for a specific event by its ID.
        """
        try:
            data = self._make_request(f"events/{event_id}.json")
            event = data
            event_details = {
                "Event": event.get("name"),
                "Date": event.get("dates", {}).get("start", {}).get("localDate"),
                "Time": event.get("dates", {}).get("start", {}).get("localTime"),
                "Venue": event["_embedded"]["venues"][0].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                "City": event["_embedded"]["venues"][0]["city"].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                "Country": event["_embedded"]["venues"][0]["country"].get("name") if "_embedded" in event and "venues" in event["_embedded"] else None,
                "Url": event.get("url"),
            }
            return event_details
        except Exception as e:
            raise Exception(f"An error occurred while retrieving event details: {str(e)}")


# Define the Tool
ticketmaster_tool = Tool(
    name="Eventbrite Event Search",
    func=TicketmasterAPITool().search_events,
    description="Searches for events using the Eventbrite API.",
    args_schema=TicketmasterEventSearchInput,
)