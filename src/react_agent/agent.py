"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
from pydantic import BaseModel, Field
from typing import Optional
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
import json
from datetime import date
from pydantic import BaseModel, Field

from langchain.tools import BaseTool, Tool
from langchain_core.runnables import Runnable
from src.react_agent.configuration import Configuration
from src.react_agent.state import InputState, OverallState, OutputState
from src.react_agent.prompts import PERSONAL_INFO_PROMPT, USER_PROMPT, SYSTEM_PROMPT

from src.react_agent.tools import (TOOLS, amadeus_tool, amadeus_hotel_tool, geoapify_tool, weather_tool, 
                                   googlemaps_tool, flight_tool, google_scholar_tool, booking_tool,
                                   google_places_tool, google_find_place_tool, google_place_details_tool, tavily_search_tool,
                                   flight_tools_condition, accomodation_tools_condition, activity_planner_tools_condition, 
                                   FlightSearchInput, AmadeusFlightSearchInput)

from src.react_agent.utils import load_chat_model
from langchain_core.tools import tool

# Define the function that calls the model

#-----------------------------------------------LLM------------------------------------------------
# Initialize the LLM
llm = LangchainChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model= "deepseek-chat",
    base_url="https://api.deepseek.com",
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
    budget: Optional[float] = Field(description="The user's travel budget in their chosen currency.")
    start_date: Optional[date] = Field(description="The start date of the trip.")
    end_date: Optional[date] = Field(description="The end date of the trip.")
    num_adults: Optional[int] = Field(description="The number of adults traveling.")
    num_children: Optional[int] = Field(description="The number of children traveling.")
    num_rooms: Optional[int] = Field(description="The number of rooms required for accommodation.")
    user_preferences: Optional[Dict[str, Any]] = Field(description="User preferences and requirements.")
    
def travel_itinerary_planner(state: OverallState) -> OverallState:
    # Define the LLM with structured output
    
    llm = LangchainChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model= "deepseek-chat",
        base_url="https://api.deepseek.com",
        )
    
    # Define the LLM with structured output
    llm_with_structure_op = llm.with_structured_output(TravelItinerary)
    
    # Define the prompt template
    prompt = PromptTemplate(
        template="""You are a travel itinerary planner. Your task is to extract relevant information from the user's query to plan a trip.
                    Here is the user's query: {query}
                    Extract the following information:
                    - Location (starting point)
                    - loc_code (an uppercase 3-letter airport code of the starting point, use the IATA airport codes, and if not available, use the city name. If there are multiple airports, choose the main one. Unless the user specifies a specific airport, use the city name.)
                    - Destination
                    - dest_code (an uppercase 3-letter airport code of the destination. use the IATA airport codes, and if not available, use the city name. If there are multiple airports, choose the main one. Unless the user specifies a specific airport, use the city name.)
                    - Budget
                    - travel_class (1: Economy, 2: Premium Economy, 3: Business, 4: First). the travel class specified by the user, else None.
                    - Start date
                    - End date
                    - Number of adults
                    - Number of children
                    - Number of rooms
                    - User preferences
                    If any information is missing, leave it as None.
                    
                    For example, if the user query is "I want to travel from New York to Los Angeles on July 1st, 2022, with a budget of $5000 for 2 adults and 1 child, and return on July 20th, 2022" you should extract the following information:
                    - Location: New York
                    - loc_code: JFK
                    - Destination: Los Angeles
                    - dest_code: LAX
                    - Budget: $5000
                    - Start date: 2022-07-01
                    - End date: 2022-07-20
                    - Number of adults: 2
                    - Number of children: 1
                    - Number of rooms: None
                    - User preferences: None
                    """,
        input_variables=["query"]
    )
    
    # Create the chain
    chain = prompt | llm_with_structure_op
    
    # Get the last message from the state
    messages = state.messages
    last_message = messages[-1].content
    
    # Invoke the chain with the user's query
    structured_output = chain.invoke({"query": last_message})
    
    # Update the state with the structured output
    updated_state = {
        "location": structured_output.location,
        "loc_code": structured_output.loc_code,
        "destination": structured_output.destination,
        "dest_code": structured_output.dest_code,
        "budget": structured_output.budget,
        "start_date": structured_output.start_date,
        "end_date": structured_output.end_date,
        "num_adults": structured_output.num_adults,
        "num_children": structured_output.num_children,
        "num_rooms": structured_output.num_rooms,
        "user_preferences": structured_output.user_preferences,
    }
    
    # Return the updated state
    return updated_state


#--------------------------------------------Flight Finder Node--------------------------------------------
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
    budget = state.budget
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

            # Debugging: Print the raw API response
            print(f"API Response for {flight_class}: {google_flights_results}")

            # Add the results to the temporary dictionary under the travel class key
            flights_dict[flight_class] = google_flights_results
            
    else:
        # Handle incomplete travel details
        flights_dict["error"] = "Incomplete travel details. Missing location, destination, or start date."

    # Assign the temporary list to the state.flights after the loop
    state.flights = [flights_dict]
    
    # Return the updated state
    return state



#--------------------------------------------Chat Node --------------------------------------------
# (uses conversational agent and returns a JSON format)
# extracts user preferences and requirements from the conversation

def chat_node(state: OverallState) -> OverallState:
    """
    Node to interact with the user, collect personal info, course info,
    and submission document, and save it to the state.
    """
    llm = LangchainChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model= "deepseek-chat",
        base_url="https://api.deepseek.com",
        )
    try:
        # Initialize the conversation if no messages exist
        if not state.messages:
            state.messages = [
                SystemMessage(content="Welcome! Please provide your travel preferences.")
            ]
        
        # Check if the last message is from the user (HumanMessage)
        if state.messages and isinstance(state.messages[-1], HumanMessage):
            # Extract the user's message
            user_message = state.messages[-1].content
            
            # Update user preferences based on the chat
            state.user_preferences.update({"chat_preferences": user_message})
            
            # Prepare the context for the LLM
            context = {
                "location": state.location,
                "destination": state.destination,
                "start_date": state.start_date,
                "end_date": state.end_date,
                "num_adults": state.num_adults,
                "num_children": state.num_children,
                "num_rooms": state.num_rooms,
                "num_rooms": state.num_rooms,
                "user_preferences": state.user_preferences
            }
            
            # Invoke the LLM with the current state and context
            response = llm.invoke([SystemMessage(content=str(context)), HumanMessage(content=user_message)])
            
            # Append the LLM's response as an Assistant Message (AIMessage)
            state.messages.append(AIMessage(content=response.content))  # Assuming `response` has a `content` attribute
        
        return state
    except Exception as e:
        # Handle errors gracefully
        state.messages = state.messages + [SystemMessage(content=f"An error occurred: {str(e)}")]
        state.next_node = "__end__"  # End the workflow if an error occurs
        return state
    
#--------------------------------------------Flight Tool Node --------------------------------------------


from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

def create_tool_node_with_fallback(tools: list) -> OverallState:
    
    def handle_tool_error(state) -> OverallState:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                    )
                for tc in tool_calls
                ]
            }
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
        )






from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable

def create_llm_with_tools_node(llm, system_prompt: str, tools: list):
    """
    Creates an Assistant class that uses an LLM with tool binding.
    
    Args:
        llm: The LLM to use (e.g., ChatOpenAI).
        system_prompt: The system prompt for the LLM.
        tools: A list of tools to bind to the LLM.
    
    Returns:
        A callable Assistant class that can be used as a node.
    """
    # Define the assistant class
    class Assistant:
        def __init__(self, runnable: Runnable):
            # Initialize with the runnable that defines the process for interacting with the tools
            self.runnable = runnable

        def __call__(self, state: OverallState) -> OverallState:
            while True:
                # Define the human message (query) based on the state
                query = f"""
                The user wants to travel from {state.loc_code} to {state.dest_code}. 
                The trip starts on {state.start_date} and ends on {state.end_date}. 
                The user is traveling with {state.num_adults} adults and {state.num_children} children. 
                The budget for flights is {state.budget}, 
                Travel class: {state.travel_class},  # Use state.travel_class here, 
                Additional preferences include: {state.user_preferences}.
                """
                
                # Combine the system prompt, human message (query), and existing state messages
                structured_prompt = ChatPromptTemplate.from_messages(
                                [
                                    SystemMessage(content=system_prompt),
                                    HumanMessage(content=query),
                                    MessagesPlaceholder(variable_name="messages"),
                                ]
                        )

                # Format the structured prompt with the required variables
                formatted_prompt = structured_prompt.format(messages=state.messages)

                # Update the state with the formatted prompt
                updated_state = OverallState(
                    loc_code=state.loc_code,
                    dest_code=state.dest_code,
                    start_date=state.start_date,
                    end_date=state.end_date,
                    num_adults=state.num_adults,
                    num_children=state.num_children,
                    budget=state.budget,
                    user_preferences=state.user_preferences,
                    travel_class=state.travel_class,
                    messages=formatted_prompt,  # Update messages with the formatted prompt
                )
                
                # Invoke the runnable with the updated state
                result = self.runnable.invoke(updated_state)
                
                # If the tool fails to return valid output, re-prompt the user to clarify or retry
                if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
                ):
                    # Add a message to request a valid response
                    updated_state.messages = "Please provide more details or clarify your request."
                else:
                    # Break the loop when valid output is obtained
                    break

            # Return the final state after processing the runnable
            return updated_state

    # Define the assistant's prompt
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    
    # Bind the tools to the assistant's workflow
    assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)
    
    # Return the Assistant class
    return Assistant(assistant_runnable)


#-------------------------------------------- Accommodation Finder --------------------------------------------



#-------------------------------------------- Activity Planner --------------------------------------------

def activity_planner_node(state: OverallState) -> OverallState:
    """
    Node to plan activities using a React agent and tools like Google Places.
    Updates the state with a structured output of activities and places.
    """
    # Define the LLM
    llm = LangchainChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model= "deepseek-chat",
        base_url="https://api.deepseek.com",
        )

    # Define the tools (e.g., Google Places, Tavily Search, etc.)
    tools = [
        google_places_tool,  # Replace with actual Google Places tool
        tavily_search_tool,  # Replace with actual Tavily Search tool
    ]

    # Define the detailed prompt for the React agent
    react_agent_prompt = PromptTemplate.from_template(
        '''You are a helpful travel activity planner. Your job is to suggest activities and places for users based on their preferences.
        Use the tools provided to find information about activities and places. Return a structured output in the following format:
        {
            "activities": [
                {
                    "name": "Activity Name",
                    "description": "Description of the activity",
                    "places": [
                        {
                            "name": "Place Name",
                            "address": "Place Address",
                            "rating": "Place Rating"
                        }
                    ]
                }
            ]
        }

        Tools:
        {tools}

        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action you should take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        Thought: I now know the final answer
        Final Answer: [Your final answer here as a structured JSON object]

        Examples:
        1. Question: What are some fun activities in New York City?
           Thought: I should search for popular activities in New York City.
           Action: google_places_tool
           Action Input: {"query": "popular activities in New York City"}
           Observation: [{"name": "Central Park", "description": "A large, iconic park in Manhattan.", "places": [{"name": "Central Park Zoo", "address": "64th St and 5th Ave, New York, NY 10021", "rating": "4.5"}]}]
           Thought: I now know the final answer.
           Final Answer: {
               "activities": [
                   {
                       "name": "Central Park",
                       "description": "A large, iconic park in Manhattan.",
                       "places": [
                           {
                               "name": "Central Park Zoo",
                               "address": "64th St and 5th Ave, New York, NY 10021",
                               "rating": "4.5"
                           }
                       ]
                   }
               ]
           }

        2. Question: What are some historical places to visit in Washington, D.C.?
           Thought: I should search for historical places in Washington, D.C.
           Action: tavily_search_tool
           Action Input: {"query": "historical places in Washington, D.C."}
           Observation: [{"name": "Lincoln Memorial", "description": "A national monument built to honor Abraham Lincoln.", "places": [{"name": "Lincoln Memorial", "address": "2 Lincoln Memorial Cir NW, Washington, DC 20037", "rating": "4.8"}]}]
           Thought: I now know the final answer.
           Final Answer: {
               "activities": [
                   {
                       "name": "Lincoln Memorial",
                       "description": "A national monument built to honor Abraham Lincoln.",
                       "places": [
                           {
                               "name": "Lincoln Memorial",
                               "address": "2 Lincoln Memorial Cir NW, Washington, DC 20037",
                               "rating": "4.8"
                           }
                       ]
                   }
               ]
           }

        Begin!
        Question: {input}
        Thought: {agent_scratchpad}
        '''
    )

    # Create the React agent
    react_agent = create_react_agent(
        llm=llm,
        prompt=react_agent_prompt,
        tools=tools,
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
    )

    # Extract the user's query from the state
    user_query = state.messages[-1].content if state.messages else "No query provided."

    # Invoke the agent executor
    try:
        response = agent_executor.invoke({
            "input": user_query,
            "agent_scratchpad": "",  # Initialize with an empty scratchpad
        })

        # Parse the structured output
        structured_output = json.loads(response["output"])  # Assuming the output is a JSON string
        state.activities = structured_output["activities"]
        print("Structured output:", structured_output)
    except Exception as e:
        # Handle errors gracefully
        print(f"Error in activity_planner_node: {e}")
        state.warnings = {"activity_planner_error": str(e)}

    return state


#------------------------------------------------Real-Time Data Provider-------------------------------------------------------
# Googles Places API
# Google Maps API
# Google Weather API
# Recommends things yoiu should be watch out for:
# Car Rental
# Crime rate
# Weather
# Currency Exchange
# Language
# Timezone
# Vaccination
# Visa
# Electricity
# Emergency Number
# Driving
# Tipping
# Dress Code
# Food
# Shopping
# Public Holidays
# Festivals
# Events
# Local Customs
# Local Laws
# Local Etiquette
# Local Time
# Local Business Hours

def realtime_provider(state: OverallState) -> OverallState:
    """
    Node to interact with the user, collect personal info, course info,
    and submission document, and save it to the state.
    """
    pass


#--------------------------------------------Itinerary Generator --------------------------------------------

def itinerary_generator(state: OverallState) -> OverallState:
    """
    Node to interact with the user, collect personal info, course info,
    and submission document, and save it to the state.
    """
    pass




#--------------------------------------------personal_info_supervisor_node--------------------------------------------
class PersonalInfoSupervisor:
    """Class to handle the personal info supervisor node."""
    class StudentInfo(BaseModel):
        """Schema for the structured output containing student information."""
        student_name: str = Field(description="The full name of the student.")
        student_email: str = Field(description="The email address of the student.")
        student_id: str = Field(description="The unique identifier for the student.")
        course_number: str = Field(description="The course number or code.")
        assignment_name: str = Field(description="The name of the assignment.")
        file_path: str = Field(description="The file path to the submission document.")
    
    def __init__(self, llm: LangchainChatDeepSeek):
        """
        Initializes the PersonalInfoSupervisor with an LLM.
        """
        self.llm = llm if llm else LangchainChatDeepSeek(
                                            api_key=os.getenv("DEEPSEEK_API_KEY"),
                                            model= "deepseek-chat",
                                            base_url="https://api.deepseek.com",
                                        )
        
        self.prompt = self._create_prompt()
        self.structured_llm = self.llm.with_structured_output(self.StudentInfo)
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Creates the prompt for the agent.
        """
        return ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        


def personal_info_supervisor_node(state: OverallState)-> OverallState:
    """
    Supervisor node to interact with the student, collect personal info, course info,
    and submission document, and save it to the state.
    """
    try:
        supervisor = PersonalInfoSupervisor(llm)
        # Initialize the conversation if no messages exist
        
        if not state.messages:
            state.messages = supervisor.prompt.messages  # Initialize with the system message
        
                # Check if the last message is from the user (HumanMessage)
        if state.messages and isinstance(state.messages[-1], HumanMessage):
            # Invoke the structured LLM with the current state
            structured_output = supervisor.structured_llm.invoke(state.messages)
            # Save the structured output to the state
            state.student_name = structured_output.student_name
            state.student_email = structured_output.student_email
            state.student_id = structured_output.student_id
            state.course_number = structured_output.course_number
            state.assignment_name = structured_output.assignment_name
            state.file_path = structured_output.file_path
            # Append the structured output as a system message for feedback
            state.messages.append(AIMessage(content=f"Thank you! Here's what I collected:\n{structured_output}"))

        # Route to the next node (Grading or Complaint)
        # return state
    except Exception as e:
        # Handle errors gracefully
        state.messages = state.messages + [SystemMessage(content=f"An error occurred: {str(e)}")]
        state.next_node = "__end__"  # End the workflow if an error occurs
        # return state









#--------------------------------------------Notification Node--------------------------------------------

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any

def notification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends grades and feedback to the student via email.
    Attaches the Excel report to the email.
    """
    try:
        # Email configuration
        sender_email = "grademaster@example.com"
        sender_password = "your_email_password"
        receiver_email = state["student_email"]
        
        # Create the email
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = f"Grades and Feedback for {state['assignment_name']}"
        
        # Email body
        body = f"""
        Dear {state['student_name']},
        
        Please find attached your grades and feedback for the assignment: {state['assignment_name']}.
        
        Best regards,
        GradeMaster Team
        """
        msg.attach(MIMEText(body, "plain"))
        
        # Attach the Excel report
        with open(state["report_path"], "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(state['report_path'])}",
        )
        msg.attach(part)
        
        # Send the email
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        
        # Update the state
        state["notification_sent"] = True
        state["next_node"] = "__end__"
        return state
    except Exception as e:
        state["feedback_comments"].append(f"Error during notification: {str(e)}")
        state["next_node"] = "__end__"
        return state