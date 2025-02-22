"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""
import time
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from datetime import date
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI as LangchainChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.prebuilt import ToolNode
from typing import Dict, List, Literal, Any
import ipywidgets as widgets
from IPython.display import display
from src.react_agent.state import OverallState
import os 
import re
import json
from datetime import date
from pydantic import BaseModel, Field

from langchain.tools import BaseTool, Tool
from langchain_core.runnables import Runnable
from src.react_agent.configuration import Configuration
from src.react_agent.state import InputState, OverallState, OutputState
from src.react_agent.prompts import SYSTEM_PROMPT, ACTIVITY_PLANNER_PROMPT, FLIGHT_FINDER_PROMPT, RECOMMENDATION_PROMPT, RECOMMENDATION_PROMPT_2

from src.react_agent.tools import (TOOLS, amadeus_tool, amadeus_hotel_tool, geoapify_tool, weather_tool, 
                                   googlemaps_tool, flight_tool, google_scholar_tool, booking_tool,
                                   google_places_tool,tavily_search_tool,
                                   flight_tools_condition, accomodation_tools_condition, activity_planner_tools_condition, 
                                   FlightSearchInput, AmadeusFlightSearchInput, BookingSearchInput, GoogleMapsPlacesInput,
                                   TicketmasterEventSearchInput, ticketmaster_tool, AirbnbSearchInput, airbnb_tool,
                                   google_flights_tool, FlightSearchInput_2)

from src.react_agent.utils import load_chat_model
from langchain_core.tools import tool

# Load environment variables from the .env file
load_dotenv()
# Define the function that calls the model

#-----------------------------------------------LLM------------------------------------------------
# Initialize the LLM
llm = LangchainChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model= "deepseek-chat",
    base_url="https://api.deepseek.com",
)


llm = LangchainChatDeepSeek(
    api_key=os.getenv("OPENAI_API_KEY"),
    model= "gpt-4o"
)
#-----------------------------------------------------------------------------------------------
# Nodes and Agents
#-----------------------------------------------------------------------------------------------

#----------------------------------------------- First Node--------------------------------------------
class TravelItinerary(BaseModel):
    location: Optional[str] = Field(description="The user's current location or starting point.")
    loc_code: Optional[str] = Field(description="The airport code of the user's current location.")
    destination: Optional[str] = Field(description="The destination the user wants to travel to.")
    dest_code: Optional[str] = Field(description="The airport code of the user's destination.")
    travel_class: Optional[Literal[0, 1, 2, 3, 4]] = Field(description="The travel class specified by the user.")
    start_date: Optional[date] = Field(description="The start date of the trip.")
    end_date: Optional[date] = Field(description="The end date of the trip.")
    num_adults: Optional[int] = Field(description="The number of adults traveling.")
    num_children: Optional[int] = Field(description="The number of children traveling.")
    user_preferences: Optional[Dict[str, Any]] = Field(description="User preferences and requirements.")
    accommodation_options: Optional[Literal["Hotel", "Airbnb"]] = Field(description="The type of accommodation chosen by the user.")
    sort_by: Optional[str] = Field(description="The sorting criteria of flight chosen by the user (e.g., price, duration, departure, arrival).")
    
# def travel_itinerary_planner(state: OverallState) -> OverallState:
#     # Define the LLM with structured output
       
#     # Define the LLM with structured output
#     llm_with_structure_op = llm.with_structured_output(TravelItinerary)
    
#     # Define the prompt template
#     prompt = PromptTemplate(
#         template="""You are a travel itinerary planner. Your task is to extract relevant information from the user's query to plan a trip.
#                     Here is the user's query: {query}
#                     Extract the following information:
#                     - Location (starting point)
#                     - loc_code (an uppercase 3-letter airport code of the starting point, use the IATA airport codes, and if not available, use the city name. If there are multiple airports, choose the main one. Unless the user specifies a specific airport, use the city name.)
#                     - destination
#                     - dest_code (an uppercase 3-letter airport code of the destination. use the IATA airport codes, and if not available, use the city name. If there are multiple airports, choose the main one. Unless the user specifies a specific airport, use the city name.)
#                     - travel_class (0: Not Specified, 1: Economy, 2: Premium Economy, 3: Business, 4: First). the travel class specified by the user, else 0 if not specified. This should be numeric values mapped to the corresponding travel class names.
#                     - Start date
#                     - End date
#                     - Number of adults
#                     - Number of children
#                     - User preferences
#                     - Accommodation options (Hotel or Airbnb). If the user does not specify, use the default value "Hotel".
#                     - sort_by - Sorting criteria for flights. Default is "price". Determine the sorting criteria based on the user's query:
#                                 - If the user is looking for the cheapest flights, set sort_by to "price".
#                                 - If the user is looking for the shortest flights, set sort_by to "duration".
#                                 - If the user is looking for the earliest flight that leaves the airport, set sort_by to "departure".
#                                 - If the user is looking for the earliest flight that arrives at the destination, set sort_by to "arrival".
#                                 - If the user does not specify any preference, or if the query does not indicate urgency, set sort_by to "price".
                                
#                     If any information is missing, leave it as None. The year will always be 2025, unless specified otherwise by the user query.
#                     The default number of adults is 1, and the default number of children is 0. If the user does not specify the number of adults or children, use the default values.
                    
#                     For example, if the user query is "I want to travel from New York to Los Angeles on July 1st, 2022, with a budget of $5000 for 2 adults and 1 child, and return on July 20th, 2022. I plan to stay in a hotel." you should extract the following information:
#                     - Location: New York
#                     - loc_code: JFK
#                     - Destination: Los Angeles
#                     - dest_code: LAX
#                     - travel_class: 0
#                     - Start date: 2022-07-01
#                     - End date: 2022-07-20
#                     - Number of adults: 2
#                     - Number of children: 1
#                     - User preferences: {{ "budget": "$5000"}}
#                     - Accommodation options: Hotel
#                     - sort_by: price

#                     Example 2:
#                     User query: "I'm planning a luxury honeymoon trip from London to Tokyo in December 12, 2025 via Gatwick Its a five day trip with my husband and We'd like to fly first class with a budget of $15,000, staying in a high-end hotel. We want private airport transfers and dietary accommodations for a vegetarian diet, and we would like our flights to leave early. Activities should include cultural tours, fine dining, and shopping. There will be 2 adults, no children, and we need 1 suite."
#                     Extracted information:
#                     - Location: London
#                     - loc_code: LGW
#                     - Destination: Tokyo
#                     - dest_code: HND
#                     - travel_class: 4
#                     - Start date: 2025-12-12
#                     - End date: 2025-12-17
#                     - Number of adults: 2
#                     - Number of children: 0
#                     - User preferences: {{ "trip_type": "Luxury", "budget": "$15,000", "transportation": "Private transfer", "diet": "Vegetarian", "activities": ["Cultural tours", "Fine dining", "Shopping"] }}
#                     - Accommodation options: Hotel
#                     - sort_by: departure
#                     """,
#         input_variables=["query"]
#     )
    
#     # Create the chain
#     chain = prompt | llm_with_structure_op
    
#     # Get the last message from the state
#     messages = state.messages
#     last_message = messages[-1].content
    
#     def invoke_chain():
#         """Runs the chain and handles errors gracefully."""
#         try:
#             structured_output = chain.invoke({"query": last_message})
#             if structured_output is None:
#                 raise ValueError("LLM returned None instead of a valid response.")
#             return structured_output
#         except Exception as e:
#             print(f"Error occurred: {e}")
#             return None

#     # **First attempt**
#     structured_output = invoke_chain()

#     # **Retry if error occurs**
#     if structured_output is None:
#         print("Retrying after error...")
#         time.sleep(2)
#         structured_output = invoke_chain()

#     if structured_output is None:
#         print("Retry failed. Assigning fallback response.")
#         structured_output = TravelItinerary(
#             location=None, loc_code=None, destination=None, dest_code=None,
#             travel_class=0, start_date=None, end_date=None,
#             num_adults=1, num_children=0, user_preferences={},
#             accommodation_options="Hotel", sort_by="price"
#         )  # Assign fallback values
        
    
#     # Update the state with the structured output
#     updated_state = {
#         "location": structured_output.location,
#         "loc_code": structured_output.loc_code,
#         "destination": structured_output.destination,
#         "dest_code": structured_output.dest_code,
#         "travel_class": structured_output.travel_class,
#         "start_date": structured_output.start_date,
#         "end_date": structured_output.end_date,
#         "num_adults": structured_output.num_adults,
#         "num_children": structured_output.num_children,
#         "user_preferences": structured_output.user_preferences,
#         "accommodation_options": structured_output.accommodation_options,
#         "sort_by": structured_output.sort_by
#     }
    
#     # Return the updated state
#     return updated_state




def travel_itinerary_planner(state: InputState) -> OverallState:
    """
    Extracts travel itinerary details from user input.
    Uses DeepSeek LLM first, and switches to GPT-4o if DeepSeek fails.
    If both fail, retries once more before returning a fallback response.
    """

    llm_deepseek = LangchainChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
    )

    llm_openai = LangchainChatDeepSeek(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    def create_chain(llm):
        """Creates a prompt processing chain for the given LLM."""
        return PromptTemplate(
            template="""You are a travel itinerary planner. Your task is to extract relevant information from the user's query to plan a trip.
                        Here is the user's query: {query}
                        Extract the following information:
                        - Location (starting point)
                        - loc_code (an uppercase 3-letter airport code of the starting point, use the IATA airport codes, and if not available, use the city name. If there are multiple airports, choose the main one. Unless the user specifies a specific airport, use the city name.)
                        - destination
                        - dest_code (an uppercase 3-letter airport code of the destination. use the IATA airport codes, and if not available, use the city name. If there are multiple airports, choose the main one. Unless the user specifies a specific airport, use the city name.)
                        - travel_class (0: Not Specified, 1: Economy, 2: Premium Economy, 3: Business, 4: First). the travel class specified by the user, else 0 if not specified. This should be numeric values mapped to the corresponding travel class names.
                        - Start date
                        - End date
                        - Number of adults
                        - Number of children
                        - User preferences
                        - Accommodation options (Hotel or Airbnb). If the user does not specify, use the default value "Hotel".
                        - sort_by - Sorting criteria for flights. Default is "price". Determine the sorting criteria based on the user's query:
                                    - If the user is looking for the cheapest flights, set sort_by to "price".
                                    - If the user is looking for the shortest flights, set sort_by to "duration".
                                    - If the user is looking for the earliest flight that leaves the airport, set sort_by to "departure".
                                    - If the user is looking for the earliest flight that arrives at the destination, set sort_by to "arrival".
                                    - If the user does not specify any preference, or if the query does not indicate urgency, set sort_by to "price".
                                    
                        If any information is missing, leave it as None. The year will always be 2025, unless specified otherwise by the user query.
                        The default number of adults is 1, and the default number of children is 0. If the user does not specify the number of adults or children, use the default values.
                        
                        For example, if the user query is "I want to travel from New York to Los Angeles on July 1st, 2022, with a budget of $5000 for 2 adults and 1 child, and return on July 20th, 2022. I plan to stay in a hotel." you should extract the following information:
                        - Location: New York
                        - loc_code: JFK
                        - Destination: Los Angeles
                        - dest_code: LAX
                        - travel_class: 0
                        - Start date: 2022-07-01
                        - End date: 2022-07-20
                        - Number of adults: 2
                        - Number of children: 1
                        - User preferences: {{ "budget": "$5000"}}
                        - Accommodation options: Hotel
                        - sort_by: price

                        Example 2:
                        User query: "I'm planning a luxury honeymoon trip from London to Tokyo in December 12, 2025 via Gatwick Its a five day trip with my husband and We'd like to fly first class with a budget of $15,000, staying in a high-end hotel. We want private airport transfers and dietary accommodations for a vegetarian diet, and we would like our flights to leave early. Activities should include cultural tours, fine dining, and shopping. There will be 2 adults, no children, and we need 1 suite."
                        Extracted information:
                        - Location: London
                        - loc_code: LGW
                        - Destination: Tokyo
                        - dest_code: HND
                        - travel_class: 4
                        - Start date: 2025-12-12
                        - End date: 2025-12-17
                        - Number of adults: 2
                        - Number of children: 0
                        - User preferences: {{ "trip_type": "Luxury", "budget": "$15,000", "transportation": "Private transfer", "diet": "Vegetarian", "activities": ["Cultural tours", "Fine dining", "Shopping"] }}
                        - Accommodation options: Hotel
                        - sort_by: departure
                        """,
            input_variables=["query"]
        ) | llm.with_structured_output(TravelItinerary)

    # Start with DeepSeek
    chain = create_chain(llm_deepseek)

    messages = state.messages
    last_message = messages[-1].content if messages else "No query provided"

    def invoke_chain(chain):
        """Runs the chain and handles errors gracefully."""
        try:
            structured_output = chain.invoke({"query": last_message})
            if structured_output is None:
                raise ValueError("LLM returned None instead of a valid response.")
            return structured_output
        except Exception as e:
            # print(f"Error occurred with LLM: {e}")
            return None

    # **First Attempt (DeepSeek)**
    structured_output = invoke_chain(chain)

    # **Switch to GPT-4o if DeepSeek Fails**
    if structured_output is None:
        chain = create_chain(llm_openai)
        structured_output = invoke_chain(chain)

    # **Retry Once More if Both Fail**
    if structured_output is None:
        time.sleep(2)
        structured_output = invoke_chain(chain)

    # # **If Retry Also Fails, Assign a Fallback Response**
    # if structured_output is None:
    #     structured_output = TravelItinerary(
    #         location=None, loc_code=None, destination=None, dest_code=None,
    #         travel_class=0, start_date=None, end_date=None,
    #         num_adults=1, num_children=0, user_preferences={},
    #         accommodation_options="Hotel", sort_by="price"
    #     )  # Assign fallback values

    # Update the state with structured output
    updated_state = {
        "location": structured_output.location,
        "loc_code": structured_output.loc_code,
        "destination": structured_output.destination,
        "dest_code": structured_output.dest_code,
        "travel_class": structured_output.travel_class,
        "start_date": structured_output.start_date,
        "end_date": structured_output.end_date,
        "num_adults": structured_output.num_adults,
        "num_children": structured_output.num_children,
        "user_preferences": structured_output.user_preferences,
        "accommodation_options": structured_output.accommodation_options,
        "sort_by": structured_output.sort_by
    }

    return updated_state

#--------------------------------------------Flight Finder Node--------------------------------------------

def flight_finder_node(state: OverallState) -> OverallState:
    """
    A Node that calls the Google Flights tool to find flights based on the state inputs.
    The results are stored in the `state.flights` list.
    """
    # Extract inputs from the state
    departure_airport = state.loc_code
    arrival_airport = state.dest_code
    departure_date = state.start_date.strftime("%Y-%m-%d") if state.start_date else None
    return_date = state.end_date.strftime("%Y-%m-%d") if state.end_date else None
    adults = state.num_adults or 1
    children = state.num_children or 0
    travel_class = state.travel_class
    sort_by = state.sort_by or "price"

    # Define travel class mapping
    travel_class_mapping = {
        1: "economy",
        2: "economy",  # Map premium economy to economy since it's not supported
        3: "business",
        4: "first",
    }

    def search_flights():
        """
        Calls the Google Flights tool and returns results.
        If an error occurs, it logs it and returns None.
        """
        try:
            flights_list = []
            if travel_class == 0:
                input_data = FlightSearchInput_2(
                    departure_airport=departure_airport,
                    arrival_airport=arrival_airport,
                    departure_date=departure_date,
                    return_date=return_date,
                    adults=adults,
                    children=children,
                    travel_class="all",
                    sort_by=sort_by,
                )
            else:
                class_name = travel_class_mapping.get(travel_class, "economy")
                input_data = FlightSearchInput_2(
                    departure_airport=departure_airport,
                    arrival_airport=arrival_airport,
                    departure_date=departure_date,
                    return_date=return_date,
                    adults=adults,
                    children=children,
                    travel_class=class_name,
                    sort_by=sort_by,
                )

            # Call the Google Flights tool
            result = google_flights_tool.func(input_data)

            # Check if the response contains an error message
            if isinstance(result, list) and len(result) > 0 and "error" in result[0]:
                # print(f"Flight API error detected: {result[0]['error']}")
                return result  # Return the error message instead of treating it as None
            
            if not result:
                raise ValueError("No flights found.")

            flights_list.extend(result)
            
            return flights_list

        except Exception as e:
            # print(f"Flight search error: {e}")
            return [{"error": f"Flight search failed: {str(e)}"}]

    # **First Attempt**
    flights = search_flights()

    # **Retry if an error message is present**
    attempts = 0
    while isinstance(flights, list) and len(flights) > 0 and "error" in flights[0] and attempts < 4:
        # print(f"Retrying flight search (attempt {attempts + 1})...")
        time.sleep(3)  # Short delay before retrying
        flights = search_flights()
        attempts += 1

    # **If all retries fail, assign the last error message as fallback**
    if isinstance(flights, list) and len(flights) > 0 and "error" in flights[0]:
        # print("Final retry failed. Assigning error response.")
        state.flights = flights
    else:
        state.flights = flights

    # Return the updated state
    return {
        "flights": state.flights
    }



def flights_router(
    state: OverallState, messages_key: str = "flights"
) -> Literal["flight_node_1", "airbnb_node", "booking_com_node"]:
    """
    Determines the next step in the workflow based on the accommodation list.

    Args:
        state (OverallState): The state to evaluate.
        messages_key (str): The key to check for accommodation information.

    Returns:
        Literal["booking_com_node", "activities"]: The next node to route to.
    """
    flights = getattr(state, messages_key, None)
    accommodation_options = getattr(state, messages_key, None)
    
    if flights is not None:
        if not flights:  # If the flights list is empty
            return "flight_node_1"

        # Check if flights contain an error message (e.g., API limit exceeded)
        if isinstance(flights, list) and any(
            isinstance(entry, dict) and any(
                "error" in class_info and "429" in class_info["error"]
                for class_info in entry.values()
            )
            for entry in flights
        ):
            return "flight_node_1"  # Route to an alternative method for retrieving flights

    if accommodation_options:
        if accommodation_options == "Hotel":
            return "booking_com_node"
        else:
            return "airbnb_node"

    else:
        raise ValueError(f"Invalid state or missing key '{messages_key}': {state}")


def flight_finder_tool_node(state: OverallState) -> OverallState:
    """
    A Tool Node that calls both Amadeus and Google Flights tools in parallel
    and stores the results in the `flights` state variable.
    """
    # Extract inputs from the state
    location = state.location
    loc_code = state.loc_code
    destination = state.destination
    dest_code = state.dest_code
    start_date = state.start_date.strftime("%Y-%m-%d") if state.start_date else None
    end_date = state.end_date.strftime("%Y-%m-%d") if state.end_date else None
    num_adults = state.num_adults or 1
    num_children = state.num_children or 0
    travel_class = state.travel_class
    user_preferences = state.user_preferences

    # Initialize a temporary list to store flight results
    flights_dict = {}

    # Define travel class mapping
    travel_class = {
        1: "Economy",
        2: "Premium Economy",
        3: "Business",
        4: "First",
    }
    
    # Call both tools in parallel (or sequentially if parallel execution is not supported)
    if location and destination and start_date:
        # Loop through travel classes (1 to 4)
        for i in range(1, 5):
            flight_class = travel_class[i]
                            
            # Call Google Flights Search Tool        
            flight_search_input = FlightSearchInput(
                departure_id=loc_code,
                arrival_id=dest_code,
                outbound_date=start_date,
                return_date=end_date,
                adults=num_adults,
                children=num_children,
                currency=user_preferences.get("currency", "USD"),
                travel_class=i,
                sort_by=user_preferences.get("sort_by", 1)
            )
        
            google_flights_results = flight_tool.func(flight_search_input)

            # Add the results to the temporary dictionary under the travel class key
            flights_dict[flight_class] = google_flights_results
            
    else:
        # Handle incomplete travel details
        flights_dict["error"] = "Incomplete travel details. Missing location, destination, or start date."

    # Assign the temporary list to the state.flights after the loop
    # state.flights = [flights_dict]
    
    # Return the updated state
    return {
        "flights": [flights_dict]
    }

#-------------------------------------------- Accommodation Finder --------------------------------------------

def accomodation_router(
    state: OverallState, messages_key: str = "accommodation_options"
) -> Literal["booking_com_node", "airbnb_node"]:
    """
    Determines the next step in the workflow based on the accommodation options.

    Args:
        state (OverallState): The state to evaluate.
        messages_key (str): The key to check for accommodation options.

    Returns:
        Literal["booking_com_node", "airbnb_node"]: The next node to route to.
    """
    accommodation_options = getattr(state, messages_key, None)
    if accommodation_options:
        if accommodation_options == "Hotel":
            return "booking_com_node"
        else:
            return "airbnb_node"
    else:
        raise ValueError(f"Invalid state or missing key '{messages_key}': {state}")


def accomodation_router_2(
    state: OverallState, messages_key: str = "accommodation"
) -> Literal["booking_com_node", "activities_node"]:
    """
    Determines the next step in the workflow based on the accommodation list.

    Args:
        state (OverallState): The state to evaluate.
        messages_key (str): The key to check for accommodation information.

    Returns:
        Literal["booking_com_node", "activities"]: The next node to route to.
    """
    accommodation = getattr(state, messages_key, None)
    if accommodation is not None:
        if not accommodation:  # If the accommodation list is empty
            return "booking_com_node"
        else:
            return "activities_node"
    else:
        raise ValueError(f"Invalid state or missing key '{messages_key}': {state}")


class AccommodationOutput(BaseModel):
    location: str = Field(..., description="The exact location or neighborhood where the traveler wants to stay (e.g., 'Brooklyn').")
    checkin_date: str = Field(..., description="The check-in date in YYYY-MM-DD format.")
    checkout_date: str = Field(..., description="The check-out date in YYYY-MM-DD format.")
    adults: int = Field(default=2, description="The number of adult guests.")
    rooms: int = Field(default=1, description="The number of rooms.")
    currency: str = Field(default="USD", description="The currency for the prices.")


def airbnb_node(state: OverallState) -> OverallState:
    """
    This node extracts accommodation details from the user's query in state.messages
    and returns a structured output that can be passed to the booking tool.
    """
    llm_with_structure = llm.with_structured_output(AccommodationOutput)

    # Define the prompt template
    prompt = PromptTemplate(
        template="""
        You are an advanced travel planner assistant. Your task is to extract accommodation details
        from the traveler's query. Use the following information to generate a structured output for
        booking accommodations:

        ### Traveler Query:
        {query}

        ### Instructions:
        1. Extract the exact location or neighborhood where the traveler wants to stay (e.g., "Brooklyn").
           - If the traveler does not specify a location, use the city or city code provided in the state.
        2. Extract the check-in and check-out dates from the query.
           - If the dates are not explicitly mentioned, use the default dates from the state.
        3. Extract the number of adults and rooms from the query.
           - If not specified, use the default values: 1 adult and 1 room.
        4. Use the default currency 'USD' unless specified otherwise.
        5. Return the structured output in the following format:
           - location: The exact location or neighborhood.
           - checkin_date: The check-in date in YYYY-MM-DD format.
           - checkout_date: The check-out date in YYYY-MM-DD format.
           - adults: The number of adult guests.
           - rooms: The number of rooms.
           - currency: The currency for the prices.

        ### Example Output:
        - location: "Brooklyn"
        - checkin_date: "2023-12-01"
        - checkout_date: "2023-12-10"
        - adults: 2
        - rooms: 1
        - currency: "USD"
        """,
        input_variables=["query"]
    )

    # Create the chain
    chain = prompt | llm_with_structure

    # Extract the user's query from state.messages
    query = state.messages[-1].content  # Assuming the last message is the user's query

    # Invoke the chain to generate the structured output
    structured_output = chain.invoke({"query": query})

    # Call Google Flights Search Tool        
    booking_search_input = AirbnbSearchInput(
        location=structured_output.location,
        checkin_date=structured_output.checkin_date,
        checkout_date=structured_output.checkout_date,
        currency=structured_output.currency,
        margin_km=5.0
    )

    airbnb_results = airbnb_tool.func(booking_search_input)
    
    # Update the state with the structured output
    # state.accommodation = airbnb_results

    # Return the updated state
    return {
        "accommodation": airbnb_results
    }


# Define the structured output for the accommodation finder

def accommodation_finder_node(state: OverallState) -> OverallState:
    """
    This node extracts accommodation details from the user's query in state.messages
    and returns a structured output that can be passed to the booking tool.
    """
    llm_with_structure = llm.with_structured_output(AccommodationOutput)

    # Define the prompt template
    prompt = PromptTemplate(
        template="""
        You are an advanced travel planner assistant. Your task is to extract accommodation details
        from the traveler's query. Use the following information to generate a structured output for
        booking accommodations:

        ### Traveler Query:
        {query}

        ### Instructions:
        1. Extract the exact location or neighborhood where the traveler wants to stay (e.g., "Brooklyn").
           - If the traveler does not specify a location, use the city or city code provided in the state.
        2. Extract the check-in and check-out dates from the query.
           - If the dates are not explicitly mentioned, use the default dates from the state.
        3. Extract the number of adults and rooms from the query.
           - If not specified, use the default values: 1 adult and 1 room.
        4. Use the default currency 'USD' unless specified otherwise.
        5. Return the structured output in the following format:
           - location: The exact location or neighborhood.
           - checkin_date: The check-in date in YYYY-MM-DD format.
           - checkout_date: The check-out date in YYYY-MM-DD format.
           - adults: The number of adult guests.
           - rooms: The number of rooms.
           - currency: The currency for the prices.

        ### Example Output:
        - location: "Brooklyn"
        - checkin_date: "2023-12-01"
        - checkout_date: "2023-12-10"
        - adults: 2
        - rooms: 1
        - currency: "USD"
        """,
        input_variables=["query"]
    )

    # Create the chain
    chain = prompt | llm_with_structure

    # Extract the user's query from state.messages
    query = state.messages[-1].content  # Assuming the last message is the user's query

    # Invoke the chain to generate the structured output
    structured_output = chain.invoke({"query": query})

    # Call Google Flights Search Tool        
    booking_search_input = BookingSearchInput(
        location=structured_output.location,
        checkin_date=structured_output.checkin_date,
        checkout_date=structured_output.checkout_date,
        adults=structured_output.adults,
        rooms=structured_output.rooms,
        currency=structured_output.currency,
    )

    booking_results = booking_tool.func(booking_search_input)
    
    # Update the state with the structured output
    # state.accommodation = booking_results

    # Return the updated state
    return {
        "accommodation": booking_results
    }



#-------------------------------------------- Activity Planner Node --------------------------------------------

# def activities_node(state: OverallState) -> OverallState:
#     """
#     This node uses a React agent to find exciting activities and places for the user.
#     """
    
#     def parse_activities_output(activities_output: str) -> List[Dict[str, any]]:
#         """
#         Parses the raw string output of activities into a list of dictionaries.
#         Each dictionary contains the title, description, and type of the activity.
#         """
#         # Split the output into individual activities
#         activities = re.split(r"\n\d+\.\s+", activities_output)  # Split by numbered list items
#         activities = [act.strip() for act in activities if act.strip()]  # Remove empty strings

#         # Parse each activity into a dictionary
#         parsed_activities = []
#         for activity in activities:
#             # Extract the title
#             title_match = re.match(r"^\*\*(.*?)\*\*", activity)
#             if title_match:
#                 title = title_match.group(1).strip()

#                 # Extract the description (everything after the title until "Type:")
#                 description_match = re.search(r":\s*(.*?)\s*(?=Type:|$)", activity, re.DOTALL)
#                 description = description_match.group(1).strip() if description_match else "No description provided."

#                 # Extract the type (inside square brackets after "Type:")
#                 type_match = re.search(r"Type:\s*\[(.*?)\]", activity)
#                 if type_match:
#                     # Split the types by comma and strip quotes and whitespace
#                     activity_type = [t.strip().strip('"') for t in type_match.group(1).split(",")]
#                 else:
#                     activity_type = []

#                 # Add the parsed activity to the list
#                 parsed_activities.append({
#                     "title": title,
#                     "description": description,
#                     "type": activity_type  # Now a clean list of strings
#                 })

#         return parsed_activities

#     # Extract user preferences and query from the state
#     preferences = state.user_preferences
#     query = state.messages[-1].content if state.messages else "No query provided"  # Assume the last message is the user's query
#     destination = state.dest_code  # Extract the destination from the state
    
#     # Format the user preferences into a readable string for the prompt
#     preferences_str = "\n".join([f"{key}: {value}" for key, value in preferences.items()])
    
    
#     # Create the React agent prompt
#     prompt = PromptTemplate.from_template(ACTIVITY_PLANNER_PROMPT)
    
#     # Create the React agent
#     search_agent = create_react_agent(
#         llm=llm,
#         tools=[tavily_search_tool],
#         prompt=prompt
#     )

#     # Create the agent executor
#     agent_executor = AgentExecutor(
#         agent=search_agent,
#         tools=[tavily_search_tool],
#         verbose=False,
#         return_intermediate_steps=True,
#         handle_parsing_errors=True
#     )

#     # Invoke the agent
#     result = agent_executor.invoke({
#         "input": f"Find exciting activities and places for the user in {query}.",
#         "preferences": preferences_str,
#         "query": query,
#         "agent_scratchpad": ""  # Initialize with an empty scratchpad
#     })

#     # Extract the final answer from the result
#     activities_output = result.get("output", "")
    
#     state.messages.append(AIMessage(content=activities_output))  # Append the AI's response to the state messages
    
#     # Parse the activities_output string into a list of dictionaries
#     activities_list = parse_activities_output(activities_output)

#     detailed_places_list = []
    
#     for activity in activities_list:
#         # Search for places using the activity title and type
#         places_input = GoogleMapsPlacesInput(
#             query=activity["title"],  # Use the activity title as the query
#             location=destination,     # Use the destination from the state
#             radius=5000,              # Search within a 5km radius
#             type=activity["type"],    # Use the activity type
#             language="en",
#             min_price=0,
#             # max_price=4,
#             open_now=False
#         )

#         try:
#             results = google_places_tool.func(places_input)
#         except Exception as e:
#             print(f"Error calling Google Maps Places API: {e}")
#             results = {"places_search_result": {"status": "ZERO_RESULTS"}}  # Simulate no results if the API call fails
        
#         # Initialize a list to store places for this activity
#         places_list = []
#         # Check if the search was successful and has results
#         if "places_search_result" in results:
#             if results["places_search_result"]["status"] == "ZERO_RESULTS":
#                 # If no results are found, include the activity title with empty fields
#                 place_details = {
#                     "name": activity["title"],  # Use the activity title as the name
#                     "description": activity["description"],
#                     "address": None,
#                     "rating": None,
#                     "photo": None
#                 }
#                 places_list.append(place_details)
#             else:
#                 # If results are found, process them
#                 places = results["places_search_result"]["results"]
                
#                 # Limit the results to 3 places per activity
#                 for place in places[:3]:
#                     # Get the photo reference (if available)
#                     photo_reference = place.get("photos", [{}])[0].get("photo_reference") if place.get("photos") else None
                    
#                     # Create a dictionary for the place
#                     place_details = {
#                         "name": place.get("name"),
#                         "address": place.get("formatted_address"),
#                         "rating": place.get("rating"),
#                         "photo": photo_reference
#                     }
                
#                     # Append the place details to the list
#                     places_list.append(place_details)
        
#         # Extend the detailed_places_list with the places_list
#         detailed_places_list.extend(places_list)     
               
#     # Update the state with the detailed places list
#     state.activities = detailed_places_list
#     # Return the updated state
#     return state


def activities_node(state: OverallState) -> OverallState:
    """
    This node uses a React agent to find exciting activities and places for the user.
    If an error occurs (e.g., "No generation chunks were returned"), the function retries up to 3 times.
    """

    def parse_activities_output(activities_output: str) -> List[Dict[str, Any]]:
        """
        Parses the raw string output of activities into a list of dictionaries.
        """
        activities = re.split(r"\n\d+\.\s+", activities_output)
        activities = [act.strip() for act in activities if act.strip()]

        parsed_activities = []
        for activity in activities:
            title_match = re.match(r"^\*\*(.*?)\*\*", activity)
            if title_match:
                title = title_match.group(1).strip()
                description_match = re.search(r":\s*(.*?)\s*(?=Type:|$)", activity, re.DOTALL)
                description = description_match.group(1).strip() if description_match else "No description provided."

                type_match = re.search(r"Type:\s*\[(.*?)\]", activity)
                activity_type = [t.strip().strip('"') for t in type_match.group(1).split(",")] if type_match else []

                parsed_activities.append({
                    "title": title,
                    "description": description,
                    "type": activity_type
                })

        return parsed_activities

    preferences = state.user_preferences
    query = state.messages[-1].content if state.messages else "No query provided"
    destination = state.dest_code  

    preferences_str = "\n".join([f"{key}: {value}" for key, value in preferences.items()]) if preferences else "{}"

    prompt = PromptTemplate.from_template(ACTIVITY_PLANNER_PROMPT)
    
    search_agent = create_react_agent(
        llm=llm,
        tools=[tavily_search_tool],
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=search_agent,
        tools=[tavily_search_tool],
        verbose=False,  # Enable for debugging
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    MAX_RETRIES = 3
    success = False  # Flag to track if execution succeeds

    for attempt in range(MAX_RETRIES):
        try:
            result = agent_executor.invoke({
                "input": f"Find exciting activities and places for the user in {query}.",
                "preferences": preferences_str,
                "query": query,
                "agent_scratchpad": ""  
            })
            # print(f"Attempt {attempt + 1}: Agent output - {result.get('output', '')}")
            
            activities_output = result.get("output", "").strip()

            # Explicit check for invalid responses
            if not activities_output or activities_output in ["No generation chunks were returned", ""]:
                raise ValueError("Invalid or empty response from agent.")

            success = True  # Mark success and break out of retry loop
            break  

        except ValueError as e:
            # print(f"Attempt {attempt + 1}: Error - {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)  
            else:
                # print("Max retries reached. Using fallback response.")
                activities_output = "No activities found due to an error."

        except Exception as e:
            # print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
            else:
                # print("Max retries reached. Using fallback response.")
                activities_output = "Error processing activities request."

    if not success:
        activities_output = "No activities found due to an error."

    state.messages.append(AIMessage(content=activities_output))
    activities_list = parse_activities_output(activities_output)

    detailed_places_list = []
    
    for activity in activities_list:
        places_input = GoogleMapsPlacesInput(
            query=activity["title"],
            location=destination,
            radius=5000,
            type=activity["type"],
            language="en",
            min_price=0,
            open_now=False
        )

        try:
            results = google_places_tool.func(places_input)
        except Exception as e:
            # print(f"Error calling Google Maps Places API: {e}")
            results = {"places_search_result": {"status": "ZERO_RESULTS"}}

        places_list = []
        if "places_search_result" in results:
            if results["places_search_result"]["status"] == "ZERO_RESULTS":
                places_list.append({
                    "name": activity["title"],
                    "description": activity["description"],
                    "address": None,
                    "rating": None,
                    "photo": None
                })
            else:
                places = results["places_search_result"]["results"]
                for place in places[:3]:
                    photo_reference = place.get("photos", [{}])[0].get("photo_reference") if place.get("photos") else None
                    places_list.append({
                        "name": place.get("name"),
                        "address": place.get("formatted_address"),
                        "rating": place.get("rating"),
                        "photo": photo_reference
                    })

        detailed_places_list.extend(places_list)

    # state.activities = detailed_places_list
    return {
        "activities": detailed_places_list
    }


#--------------------------------------------TicketMaster Node --------------------------------------------

# Define the structured output class for the LLM
class TicketmasterOutput(BaseModel):
    location: str = Field(..., description="The city or region where the traveler is going (e.g., 'New York').")
    start_date_time: str = Field(..., description="The start date and time for event search in ISO 8601 format (e.g., '2025-02-01T00:00:00Z').")
    end_date_time: str = Field(..., description="The end date and time for event search in ISO 8601 format (e.g., '2025-02-28T23:59:59Z').")
    keywords: List[str] = Field(..., description="A list of keywords representing exciting activities or events for the location (e.g., ['music', 'theater', 'tech']).")
    country_code: str = Field(default="US", description="The country code for the location (e.g., 'US').")
    size: int = Field(default=15, description="The number of events to retrieve per keyword.")
    page: int = Field(default=1, description="The page number for pagination.")
    sort: str = Field(default="relevance,desc", description="The sorting criteria for events.")

# Define the Ticketmaster node
def ticketmaster_node(state: OverallState) -> OverallState:
    """
    This node extracts event details from the user's query in state.messages,
    generates a list of keywords based on the location and preferences,
    and searches for events using the Ticketmaster API.
    """
    llm_with_structure = llm.with_structured_output(TicketmasterOutput)

    # Define the prompt template
    prompt = PromptTemplate(
        template="""
        You are an advanced event assistant for TicketMaster, a company that sells and distributes tickets for live events. 
        It's the world's largest ticket marketplace, offering tickets for concerts, sports games, theater shows, and more.
        Your task is to extract event details from the traveler's destination in the traveler's query and generate a list of keywords 
        representing exciting live events specifically tailored to the location they are visiting. The focus is solely on live events that 
        can be sold via tickets, such as concerts, comedy, shows, theater, movies, Broadway, club, and similar. 
        DO NOT include user preferences like museums, parks, or sightseeing. Use the following information to generate
        a structured output for searching events:
        
        ### Traveler Query:
        {query}
        
        ### Instructions:
        1. Extract the city or region where the traveler is going (e.g., "New York").
           - If the traveler does not specify a location, use the city or city code provided in the state.
        2. Extract the start and end dates for event search from the query.
           - If the dates are not explicitly mentioned, use the default dates from the state.
        3. Generate a list of **exciting** keywords representing live events for the location.
           - Keywords should be specific to the destination's live event culture. 
           - For example:
                - New York: ["Broadway", "theater", "music", "comedy shows", "movies"]
                - Los Angeles: ["music", "concerts", "movies", "comedy"]
                - Nashville: ["country", "music", "concerts", "live", "shows", "music, "festivals"]
           - Keywords must focus on ticketable live events and exclude non-ticketed activities or generic places.
           - I would recommend from only 3 to 5 keywords, from the following categories:
                - ["Broadway", "theater", "music", "comedy, shows", "movies", "concerts", "live", "festivals", "country"]
        4. Use the default country code 'US' unless specified otherwise.
        5. Return the structured output in the following format:
           - location: The city or region.
           - start_date_time: The start date and time in ISO 8601 format.
           - end_date_time: The end date and time in ISO 8601 format.
           - keywords: A list of keywords.
           - country_code: The country code.
           - size: The number of events to retrieve per keyword.
           - page: The page number for pagination.
           - sort: The sorting criteria for events.
        ### Example Output:
        - location: "New York"
        - start_date_time: "2025-02-01T00:00:00Z"
        - end_date_time: "2025-02-28T23:59:59Z"
        - keywords: ["music", "theater", "movies"]
        - country_code: "US"
        - size: 15
        - page: 1
        - sort: "relevance,desc"
        """,
        input_variables=["query"]
    )

    # Create the chain
    chain = prompt | llm_with_structure

    # Extract the user's query from state.messages
    query = state.messages[-1].content  # Assuming the last message is the user's query

    # Invoke the chain to generate the structured output
    structured_output = chain.invoke({"query": query})

    # Initialize an empty list to store all event results
    all_event_results = []

    # Loop through each keyword and call the Ticketmaster API
    for keyword in structured_output.keywords:
        # Prepare the input for the Ticketmaster API
        ticketmaster_input = TicketmasterEventSearchInput(
            keyword=keyword,
            city=structured_output.location,
            country_code=structured_output.country_code,
            start_date_time=structured_output.start_date_time,
            end_date_time=structured_output.end_date_time,
            size=structured_output.size,
            page=structured_output.page,
            sort=structured_output.sort
        )

        # Call the Ticketmaster API
        event_results = ticketmaster_tool.func(ticketmaster_input)
        # Extend the all_event_results list with the results
        all_event_results.extend(event_results)

    # Update the state with the event results
    # state.live_events = all_event_results

    # Return the updated state
    return {
        "live_events": all_event_results
    }


#---------------------------------------------------------- Recommendations Node --------------------------------------------
# use weather_tool to get the weather forecast for the location
# the use LLM to interpret the weather forecast,
# and return the interpretation as a string

def recommendations_node(state: OverallState) -> OverallState:
    """
    This node uses a React agent to find crucial travel advice and insights for the user.
    """

    def extract_recommendations_3(output: str):
        """
        Extracts a list of dictionaries from the provided string output.

        Args:
            output (str): The string containing travel recommendations.

        Returns:
            list[dict]: A list of dictionaries with the extracted recommendations.
        """
        # Define a regex pattern to capture the key-value pairs in the recommendations
        pattern = r'\*\*\{"(.*?)": "(.*?)"\}\*\*'
        
        # Find all matches using regex
        matches = re.findall(pattern, output)
        
        # Convert matches to a list of dictionaries
        recommendations = [{key: value} for key, value in matches]
        
        return recommendations


    def extract_recommendations_1(output: str):
        """
        Extracts a list of dictionaries from the provided string output.

        Args:
            output (str): The string containing travel recommendations.

        Returns:
            list[dict]: A list of dictionaries with the extracted recommendations.
        """
        # Split the output into lines and look for the numbered recommendations
        recommendations = []
        lines = output.split("\n")

        for line in lines:
            # Match lines that start with a number followed by a period and a space
            match = re.match(r"^(\d+)\.\s*\*\*(.*?)\*\*:\s*(.*)$", line)
            if match:
                number = match.group(1)  # The recommendation number (optional if needed)
                key = match.group(2).strip()  # The key (e.g., "Crime Rate")
                value = match.group(3).strip()  # The value (e.g., "New York City is generally safe...")
                recommendations.append({key: value})

        return recommendations


    def extract_recommendations_2(output: str):
        """
        Extracts a list of dictionaries from the provided string output.

        Args:
            output (str): The string containing travel recommendations.

        Returns:
            list[dict]: A list of dictionaries with the extracted recommendations.
        """
        # Match JSON-like structure for recommendations in the output
        recommendations = []

        try:
            # Attempt to load the output as JSON if it's already formatted that way
            recommendations = json.loads(output)
        except json.JSONDecodeError:
            # Fallback to regex-based extraction for non-JSON outputs
            lines = output.split("\n")

            for line in lines:
                # Match lines that start with a number followed by a period and a space
                match = re.match(r"^\s*\{\s*\"(.*?)\"\s*:\s*\"(.*?)\"\s*\}\s*$", line)
                if match:
                    key = match.group(1).strip()  # Extract the key
                    value = match.group(2).strip()  # Extract the value
                    recommendations.append({key: value})

        return recommendations

    # Extract user preferences and query from the state
    if not state.messages:
        state.messages.append(AIMessage(content="Please provide a destination or query."))
        return state

    query = state.messages[0].content  
    
    # Create the React agent prompt
    prompt = PromptTemplate.from_template(RECOMMENDATION_PROMPT)

    # Create the React agent
    search_agent = create_react_agent(
        llm=llm,
        tools=[tavily_search_tool, weather_tool],
        prompt=prompt
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=search_agent,
        tools=[tavily_search_tool, weather_tool],
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    # Invoke the agent
    result = agent_executor.invoke({
        "input": f"Find exciting activities and places for the user in {query}.",
        "query": query,
        "agent_scratchpad": ""  # Initialize with an empty scratchpad
    })

    activities_output = result.get("output", "")
    
    # Try extract_recommendations_1 first
    activities = extract_recommendations_1(activities_output)

    # If no result, try extract_recommendations_2
    if not activities:
        activities = extract_recommendations_2(activities_output)

    # If still no result, try extract_recommendations_3
    if not activities:
        activities = extract_recommendations_3(activities_output)

    # state.recommendations = activities
    state.messages.append(AIMessage(content=activities_output))

    return {
        "recommendations": activities,
        
    }




class Recommendations(BaseModel):
    recommendations: Optional[List[Dict[str, str]]] = Field(description="The user's current location or starting point.")

def recommendations_node_2(state: OverallState) -> OverallState:
    import openai
    
    # If you haven't set up your API key globally, do so here:
    # openai.api_key = "YOUR_OPENAI_API_KEY"
    
    client = openai  # or adapt to your environment if needed

    # Combine all user messages into a single query
    all_messages = "\n".join([message.content for message in state.messages])
    preferences_text = "\n".join([f"{key}: {value}" for key, value in state.user_preferences.items()])
    query = f"{all_messages}\n\nUser Preferences:\n{preferences_text}"

    # Define the structured response format with a JSON Schema
    completion = client.chat.completions.create(
        model="gpt-4o",  # Replace with a valid model you have access to.
        messages=[
            {"role": "system", "content": RECOMMENDATION_PROMPT_2},
            {"role": "user", "content": query},
        ],
        # The 'response_format' parameter needs 'json_schema' -> 'name' + 'schema'
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "recommendation_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "recommendations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {
                                        "type": "string",
                                        "description": "Short label for the recommendation"
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "Concise recommendation content"
                                    }
                                },
                                "required": ["key", "value"],
                                "additionalProperties": False
                            },
                            "description": "A list of travel recommendations."
                        }
                    },
                    "required": ["recommendations"],
                    "additionalProperties": False
                }
            },
        },
    )

    # Parse and return the generated structured output
    try:
        structured_output = completion.choices[0].message.content
        parsed_output = json.loads(structured_output)
        recommendation_list = parsed_output["recommendations"]  # This is the list of dictionaries
        transformed_list = [{item["key"]: item["value"]} for item in recommendation_list]
        
        state.recommendations = transformed_list
        
        return state 
    
    except Exception as e:
        # print(f"Error occurred: {e}")
        return {"error": "Failed to generate recommendations."}
    


def recommendation_router(
    state: OverallState, messages_key: str = "recommendations"
) -> Literal["recommendation_node_2", "__end__"]:
    """
    Determines the next step in the workflow based on the accommodation list.

    Args:
        state (OverallState): The state to evaluate.
        messages_key (str): The key to check for accommodation information.

    Returns:
        Literal["booking_com_node", "activities"]: The next node to route to.
    """
    recommendation = getattr(state, messages_key, None)
    if recommendation is not None:
        if not recommendation: # If the accommodation list is empty
            return "recommendation_node_2"
        else:
            return "__end__"
    else:
        raise ValueError(f"Invalid state or missing key '{messages_key}': {state}")
