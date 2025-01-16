"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, OverallState, OutputState
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model
from react_agent.agent import (PersonalInfoSupervisor, personal_info_supervisor_node, tool_node, 
                               parallel_grading, reflection_node, insights_node, reporting_node, notification_node
                                )
# Define the function that calls the model


async def call_model(
    state: OverallState, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}



from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, MessagesPlaceholder
from langchain.agents import create_react_agent
from typing import Dict, List, Literal, Any
from react_agent.state import OverallState
import os 

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

# Define the two nodes we will cycle between
builder.add_node(call_model)

# Add the Personal Info Supervisor Node
supervisor = PersonalInfoSupervisor(llm)
builder.add_node("personal_info_supervisor", lambda state: personal_info_supervisor_node(OverallState, supervisor))
builder.add_node("tool_node", tool_node)
builder.add_node("grading_node", parallel_grading)
builder.add_node("reflection_node", reflection_node)
builder.add_node("insights_node", insights_node)
builder.add_node("reporting_node", reporting_node)
builder.add_node("notification_node", notification_node)

# Define edges
builder.add_edge(START, "personal_info_supervisor")
# builder.add_edge(START, "call_model") # Set the entrypoint as `call_model`
builder.add_edge("personal_info_supervisor", "tool_node")
builder.add_edge("tool_node", "grading_node")
builder.add_edge("grading_node", "reflection_node")
builder.add_conditional_edge(
    "reflection_node",
    lambda state: state["next_node"],
    {
        "grading_node": "grading_node",
        "insights_node": "insights_node",
    },
)
builder.add_edge("reflection_node", "insights_node")
builder.add_edge("insights_node", "reporting_node")
builder.add_edge("reporting_node", "notification_node")
builder.add_edge("notification_node", END)




graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
