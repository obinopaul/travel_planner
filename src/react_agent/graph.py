"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
import os 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.react_agent.configuration import Configuration
from src.react_agent.state import InputState, OverallState, OutputState
from src.react_agent.tools import TOOLS, AmadeusFlightSearchInput, AmadeusFlightSearchTool
from src.react_agent.utils import load_chat_model
from langgraph.prebuilt import tools_condition
from src.react_agent.prompts import SYSTEM_PROMPT, FLIGHT_FINDER_PROMPT, ACTIVITY_PLANNER_PROMPT

from src.react_agent.agent import ( travel_itinerary_planner, flight_finder_tool_node,
                                accommodation_finder_node, activities_node, ticketmaster_node, recommendations_node)

from src.react_agent.tools import (TOOLS, amadeus_tool, amadeus_hotel_tool, geoapify_tool, weather_tool,  AmadeusFlightSearchInput,
                                   FlightSearchInput,
                                   googlemaps_tool, flight_tool, google_scholar_tool, booking_tool, tavily_search_tool,
                                   flight_tools_condition, accomodation_tools_condition, activity_planner_tools_condition,
                                   multiply_tool, GoogleMapsPlacesInput, google_places_tool, google_find_place_tool, google_place_details_tool,
                                   google_events_tool, GoogleEventsSearchInput, TicketmasterEventSearchInput, ticketmaster_tool,)

from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from datetime import date
import logging

# Suppress debug messages from ipywidgets
logging.getLogger('ipywidgets').setLevel(logging.WARNING)
logging.getLogger('comm').setLevel(logging.WARNING)
logging.getLogger('tornado').setLevel(logging.WARNING)
logging.getLogger('traitlets').setLevel(logging.WARNING)

# Disable all logging globally
logging.disable(logging.CRITICAL)  # Disable all logging below CRITICAL level

# Redirect all logging output to os.devnull
logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()])

# Suppress warnings as well (optional)
import warnings
warnings.filterwarnings("ignore")


# Define a new graph
# ----------------------------------------- Nodes ---------------------------------------------------
builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("interface", travel_itinerary_planner)
builder.add_node("flight_node", flight_finder_tool_node)
builder.add_node("accomodation_node", accommodation_finder_node)
builder.add_node("activities", activities_node)
builder.add_node("live_events_node", ticketmaster_node)
builder.add_node("recommendation_node", recommendations_node)


# ----------------------------------------- Edges ---------------------------------------------------
builder.add_edge(START, "interface")
builder.add_edge("interface", "flight_node")
builder.add_edge("flight_node", "accomodation_node")
builder.add_edge("accomodation_node", "activities")
builder.add_edge("activities_node", "live_events_node")
builder.add_edge("live_events_node", "recommendation_node")
builder.add_edge("recommendation_node", END)

# ---------------------------------------- Graph ---------------------------------------------------
graph = builder.compile()
graph.name = "Travel Itinerary Planner"




# ----------------------------------------- Invoke the Graph ---------------------------------------------------
# # **Input Collection**
# user_input = """ I want to travel from Los Angeles to New York on 2025-2-15 and return on 2025-2-22 via La Guardia Airport. 
# There is 1 adult. My budget is $5000. I need 1 room in The Bronx for 5 days. I prefer an AirBnB with free breakfast and a 
# swimming pool. I also want to visit the museums and enjoy local cuisine, and go to the club at night. I might also want a massage.
# """    


# # **Input State**
# input_state = {"messages": [HumanMessage(content=user_input)]}

# # **Graph Invocation**
# output = graph.invoke(input_state, {"recursion_limit": 3000})