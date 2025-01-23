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

from src.react_agent.agent import (create_tool_node_with_fallback, create_travel_interface, chat_node,
                               create_llm_with_tools_node, activity_planner_node, flight_finder, accomodation_finder,
                               activity_planner, realtime_provider, itinerary_generator
                                )

from src.react_agent.tools import (TOOLS, amadeus_tool, amadeus_hotel_tool, geoapify_tool, weather_tool, 
                                   googlemaps_tool, flight_tool, google_scholar_tool, booking_tool)
# Define the function that calls the model



# Initialize the LLM
llm = LangchainChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model= "deepseek-chat",
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
builder.add_node("interface", create_travel_interface(OverallState()))
builder.add_node("chat_node", chat_node)

# Flight Nodes
flight_finder = create_llm_with_tools_node(llm, system_prompt, [amadeus_tool, flight_tool]) # Add the flight tools here
builder.add_node("flight_node", flight_finder)
builder.add_node("flight_tools", create_tool_node_with_fallback([amadeus_tool, flight_tool]))

# Accomodation Nodes
accomodation_finder = create_llm_with_tools_node(llm, system_prompt, [amadeus_hotel_tool, booking_tool]) # Add the flight tools here
builder.add_node("accomodation_node", accomodation_finder)
builder.add_node("accomodation_tools", create_tool_node_with_fallback([amadeus_hotel_tool, booking_tool]))

# Activity Planner Nodes
builder.add_node("activity_planner", activity_planner_node)

# Add the Personal Info Supervisor Node
builder.add_node("personal_info_supervisor", chat_node)
builder.add_node("tool_node", flight_finder)

# ----------------------------------------- Add Edge ---------------------------------------------------
# Define edges
builder.add_edge(START, "interface")
builder.add_edge("personal_info_supervisor", END)

graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
