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
from src.react_agent.prompts import SYSTEM_PROMPT, PERSONAL_INFO_PROMPT, SYSTEM_PROMPT, FLIGHT_FINDER_PROMPT, ACCOMODATION_PROMPT, ACTIVITY_PLANNER_PROMPT

from src.react_agent.agent import (create_tool_node_with_fallback, travel_itinerary_planner,
                               create_llm_with_tools_node, activity_planner_node, flight_finder, accomodation_finder,
                               activity_planner, realtime_provider, itinerary_generator, reat_time_data_node
                                )

from src.react_agent.tools import (TOOLS, amadeus_tool, amadeus_hotel_tool, geoapify_tool, weather_tool, 
                                   googlemaps_tool, flight_tool, google_scholar_tool, booking_tool,
                                   google_places_tool, google_find_place_tool, google_place_details_tool, tavily_search_tool,
                                   flight_tools_condition, accomodation_tools_condition, activity_planner_tools_condition)
# Define the function that calls the model



# Initialize the LLM
llm = LangchainChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model= "deepseek-coder", # use deepseek-chat for DeepSeek V3 or deepseek-reasoner for DeepSeek R1, or deepseek-coder for DeepSeek Coder
    base_url="https://api.deepseek.com",
)

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="Welcome to GradeMaster! Please provide your personal information, course details, and submission document."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Define a new graph
builder = StateGraph(OverallState, input=InputState, output = OutputState, config_schema=Configuration)

# ----------------------------------------- Add Node ---------------------------------------------------
builder.add_node("interface", travel_itinerary_planner)

# Flight Nodes
flight_finder = create_llm_with_tools_node(llm, FLIGHT_FINDER_PROMPT, [amadeus_tool, flight_tool]) # Add the flight tools here
builder.add_node("flight_node", flight_finder)
builder.add_node("flight_tools", create_tool_node_with_fallback([amadeus_tool, flight_tool]))

# Accomodation Nodes
accomodation_finder = create_llm_with_tools_node(llm, ACCOMODATION_PROMPT, [amadeus_hotel_tool, booking_tool]) # Add the flight tools here
builder.add_node("accomodation_node", accomodation_finder)
builder.add_node("accomodation_tools", create_tool_node_with_fallback([amadeus_hotel_tool, booking_tool]))

# Activity Planner Nodes
activity_planner_tool = create_llm_with_tools_node(llm, ACTIVITY_PLANNER_PROMPT, [geoapify_tool, google_places_tool, tavily_search_tool]) # Add the flight tools here
builder.add_node("activity_planner", activity_planner_node)
builder.add_node("activity_planner_tools", activity_planner_tool)

# real time data Node
# Super Node Router
builder.add_node("realtime_provider", realtime_provider)


# real time data Node
builder.add_node("itinerary_generator", itinerary_generator)









# ----------------------------------------- Add Edge ---------------------------------------------------
# Define edges
builder.add_edge(START, "interface")
builder.add_edge("interface", "flight_node")
builder.add_conditional_edges(
    "flight_node",
    flight_tools_condition,
)
builder.add_edge("flight_tools", "flight_node")

builder.add_conditional_edges(
    "accomodation_node",
    accomodation_tools_condition,
)
builder.add_edge("accomodation_tools", "accomodation_node")

builder.add_conditional_edges(
    "activity_planner",
    activity_planner_tools_condition,
)
builder.add_edge("activity_planner_tools", "activity_planner")

builder.add_edge("activity_planner", "realtime_provider")
builder.add_edge("realtime_provider", "itinerary_generator")

builder.add_edge("itinerary_generator", END)

graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
