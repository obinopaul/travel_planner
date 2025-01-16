"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Sequence, Annotated
from langchain.schema import BaseMessage
import operator


@dataclass  
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """



@dataclass
class OverallState(InputState):
    # Student Information
    """Represents the complete state of the grading process, including intermediate and final results."""
    student_name: str = field(default=None)
    "Name of the student."
    student_email: str = field(default=None)
    "Email address of the student."
    student_id: str = field(default=None)
    "Unique identifier for the student."
    phone_number: str = field(default=None)
    "Phone number of the student."
    
    # Assignment Information
    course_name: str = field(default=None)
    "Name of the course."
    course_number: str = field(default=None)
    "Course number or code."
    assignment_name: str = field(default=None)
    "Name of the assignment."
    file_path: str = field(default=None)
    
    # Grading Information
    student_grades: Dict[str, float] = field(default_factory=dict)
    "Dictionary of grades for each question or section of the assignment."
    scores: List[Dict[str, Any]] = field(default_factory=list)
    "List of Dictionary of actual scores for each question or section of the assignment."
    solution_key: Dict[str, Any] = field(default_factory=dict)
    "Solution key containing correct answers and scoring details."
    student_submission: Optional[str] = field(default=None)
    "The student's solution to the question."
    feedback_comments: List[str] = field(default_factory=list)
    "List of feedback comments for the student's submission."
    reflection: Optional[str] = field(default=None)
    "Reflection or analysis provided by the reflection agent."
    
    # State Management
    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise."
    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed."
    is_last_step: bool = field(default=False)
    "Indicates whether the current step is the last one before the graph raises an error."
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    "Dictionary to store intermediate results during the grading process."
    final_output: Dict[str, Any] = field(default_factory=dict)
    "Dictionary to store the final output of the grading process."
    
    # Agent Workflow Tracking
    messages: Annotated[Sequence[BaseMessage], operator.add] = field(default_factory=list)
    "Stores the sequence of messages exchanged between the user and the agent."
    search_queries: List[str] = field(default_factory=list)
    "List of generated search queries to find relevant information."
    
@dataclass
class OutputState:
    """Represents the final output state to be returned to the user or system."""
    student_name: str
    "Name of the student."
    student_email: str
    "Email address of the student."
    student_id: str
    "Unique identifier for the student."
    course_name: str
    "Name of the course."
    course_number: str
    "Course number or code."
    assignment_name: str
    "Name of the assignment."
    student_grades: Dict[str, float]
    "Dictionary of grades for each question or section of the assignment."
    scores: Dict[str, float]
    "Dictionary of Actual scores for each question or section of the assignment."
    feedback_comments: List[str]
    "List of feedback comments for the student's submission."
    reflection: str
    "Reflection or analysis provided by the reflection agent."
    is_satisfactory: bool
    "True if all required fields are well populated, False otherwise."