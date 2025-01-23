"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

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
from src.react_agent.prompts import PERSONAL_INFO_PROMPT, USER_PROMPT
from src.react_agent.tools import (TOOLS, AmadeusFlightSearchInput, AmadeusFlightSearchTool, 
                                   AmadeusHotelSearchTool, AmadeusHotelSearchInput,
                                   GeoapifyPlacesSearchTool, GeoapifyPlacesSearchInput,
                                   WeatherSearchTool, WeatherSearchInput,
                                   GoogleFlightsSearchTool, FlightSearchInput, 
                                   GoogleScholarSearchTool, GoogleScholarSearchInput,
                                   BookingScraperTool, BookingSearchInput,
                                   tavily_search_tool, google_places_tool, google_find_place_tool, google_place_details_tool,
                                   )
from src.react_agent.utils import load_chat_model
from langchain.agents import create_react_agent, AgentExecutor
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

#--------------------------------------------Tool Node--------------------------------------------
def create_travel_interface(state: OverallState) -> OverallState:
    """
    Creates an interactive interface for users to input travel details and preferences.
    Updates the state with the user's input.
    """
    # Create widgets for user input
    destination_input = widgets.Text(
        placeholder='Enter your destination (e.g., New York City)',
        description='Destination:',
        disabled=False
    )

    start_date_picker = widgets.DatePicker(
        description='Start Date:',
        value=date.today(),
        disabled=False
    )

    end_date_picker = widgets.DatePicker(
        description='End Date:',
        value=date.today(),
        disabled=False
    )

    num_adults_dropdown = widgets.Dropdown(
        options=[(str(i), i) for i in range(1, 11)],
        value=1,
        description='Adults:',
        disabled=False
    )

    num_children_dropdown = widgets.Dropdown(
        options=[(str(i), i) for i in range(0, 11)],
        value=0,
        description='Children:',
        disabled=False
    )

    num_rooms_dropdown = widgets.Dropdown(
        options=[(str(i), i) for i in range(1, 6)],
        value=1,
        description='Rooms:',
        disabled=False
    )

    preferences_text = widgets.Textarea(
        placeholder='Enter your preferences (e.g., "I want a beach vacation with good food")',
        description='Preferences:',
        disabled=False
    )

    submit_button = widgets.Button(
        description='Submit',
        disabled=False,
        button_style='success',
        tooltip='Submit your travel details'
    )

    # Function to handle button click
    def on_submit_button_clicked(b):
        # Update the state with the user's input
        state.destination = destination_input.value
        state.start_date = start_date_picker.value
        state.end_date = end_date_picker.value
        state.num_adults = num_adults_dropdown.value
        state.num_children = num_children_dropdown.value
        state.num_rooms = num_rooms_dropdown.value
        state.user_preferences = {"preferences": preferences_text.value}
        
        # Print the state to confirm the data is saved
        print("State updated with the following information:")
        print(f"Destination: {state.destination}")
        print(f"Start Date: {state.start_date}")
        print(f"End Date: {state.end_date}")
        print(f"Number of Adults: {state.num_adults}")
        print(f"Number of Children: {state.num_children}")
        print(f"Number of Rooms: {state.num_rooms}")
        print(f"Preferences: {state.user_preferences['preferences']}")
        
        # Proceed to the next step (e.g., calling the chat_node)
        print("Proceeding to the next step...")

    # Attach the button click handler
    submit_button.on_click(on_submit_button_clicked)

    # Display the widgets
    display(destination_input)
    display(start_date_picker)
    display(end_date_picker)
    display(num_adults_dropdown)
    display(num_children_dropdown)
    display(num_rooms_dropdown)
    display(preferences_text)
    display(submit_button)

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

def create_tool_node_with_fallback(tools: list) -> dict:
    
    def handle_tool_error(state) -> dict:
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




#--------------------------------------------Flight Finder Node--------------------------------------------

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

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

        def __call__(self, state: dict):
            while True:
                # Invoke the runnable with the current state (messages and context)
                result = self.runnable.invoke(state)
                
                # If the tool fails to return valid output, re-prompt the user to clarify or retry
                if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
                ):
                    # Add a message to request a valid response
                    messages = state["messages"] + [("user", "Respond with a real output.")]
                    state = {**state, "messages": messages}
                else:
                    # Break the loop when valid output is obtained
                    break
            
            # Return the final state after processing the runnable
            return {"messages": result}

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
                SystemMessage(content=PERSONAL_INFO_PROMPT),
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