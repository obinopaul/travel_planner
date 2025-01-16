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
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI as LangchainChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage, MessagesPlaceholder
from langchain.agents import create_react_agent
from langgraph.prebuilt import ToolNode
from typing import Dict, List, Literal, Any
from react_agent.state import OverallState
import os 
from langchain_core.pydantic_v1 import BaseModel, Field

from react_agent.configuration import Configuration
from react_agent.state import InputState, OverallState, OutputState
from react_agent.prompts import PERSONAL_INFO_PROMPT
from react_agent.tools import TOOLS, split_pdf_tool, search_tool, advanced_ocr_tool, retrieve_solution_key_tool
from react_agent.utils import load_chat_model
 
# Define the function that calls the model

#-----------------------------------------------LLM------------------------------------------------
# Initialize the LLM
llm = LangchainChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model= "deepseek-chat",
    base_url="https://api.deepseek.com",
)


#-----------------------------------------------Prompts------------------------------------------------
# Define the prompt for the agent
# prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(content="Welcome to GradeMaster! Please provide your personal information, course details, and submission document."),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

#-----------------------------------------------------------------------------------------------
# Nodes and Agents
#-----------------------------------------------------------------------------------------------

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






#--------------------------------------------personal_info_supervisor_node--------------------------------------------
class PersonalInfoSupervisor:
    """Class to handle the personal info supervisor node."""
    class StudentInfo(BaseModel):
        """Schema for the structured output containing student information."""
        student_name: str = Field(description="The full name of the student.")
        student_email: str = Field(description="The email address of the student.")
        student_id: str = Field(description="The unique identifier for the student.")
        phone_number: str = Field(description="The phone number of the student.")
        course_name: str = Field(description="The name of the course.")
        course_number: str = Field(description="The course number or code.")
        assignment_name: str = Field(description="The name of the assignment.")
        file_path: str = Field(description="The file path to the submission document.")
        has_complaint: bool = Field(description="Whether the student has a complaint about their grades.")
    
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
        
        
def personal_info_supervisor_node(state: Dict[str, Any], supervisor: PersonalInfoSupervisor) -> Dict[str, Any]:
    """
    Supervisor node to interact with the student, collect personal info, course info,
    and submission document, and save it to the state.
    """
    try:
        # Initialize the conversation if no messages exist
        if not state.get("messages"):
            state["messages"] = [
                SystemMessage(content="Welcome to GradeMaster! Please provide your personal information, course details, and submission document.")
            ]
        # Invoke the structured LLM with the current state
        structured_output = supervisor.structured_llm.invoke({"messages": state["messages"]})
        # Save the structured output to the state
        state["student_name"] = structured_output.student_name
        state["student_email"] = structured_output.student_email
        state["student_id"] = structured_output.student_id
        state["phone_number"] = structured_output.phone_number
        state["course_name"] = structured_output.course_name
        state["course_number"] = structured_output.course_number
        state["assignment_name"] = structured_output.assignment_name
        state["file_path"] = structured_output.file_path
        # Route to the next node (Grading or Complaint)
        if structured_output.has_complaint:
            state["next_node"] = "complaint_node"
        else:
            state["next_node"] = "grading_node"
        return state
    except Exception as e:
        return {
            "messages": state.get("messages", []) + [SystemMessage(content=f"An error occurred: {str(e)}")],
            "next_node": "__end__",  # End the workflow if an error occurs
        }



#--------------------------------------------Extract Tool Node--------------------------------------------
# Tool Node Implementation
def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool node to process submitted files (PDF), split them by page ranges, extract text or images using OCR,
    and structure the data into a list of dictionaries for each question.
    Updates the state with structured data for grading or complaint processing.
    """
    try:
        # Step 1: Split the PDF into smaller PDFs by page ranges
        split_files = split_pdf_tool(state["file_path"], page_ranges=state["submission_page_ranges"])  # Example page ranges
        # Step 2: Extract text from each split PDF using OCR
        student_submission = []
        for i, file_path in enumerate(split_files):
            extracted_text = advanced_ocr_tool(file_path)
            student_submission.append({f"question {i +1}": extracted_text})
        # Step 3: Retrieve the solution key
        solution_key = retrieve_solution_key_tool(state["course_number"], state["assignment_name"])
        # Step 4: Update the state
        state["student_submission"] = student_submission
        state["solution_key"] = solution_key["text"]
        state["scores"] = solution_key["scores"]
        # Step 5: Route to the next node
        state["next_node"] = "grading_node" if not state["has_complaint"] else "complaint_node"
        return state
    except Exception as e:
        return {
            "messages": [SystemMessage(content=f"An error occurred: {str(e)}")],
            "next_node": "__end__",  # End the workflow if an error occurs
        }


#--------------------------------------------Grading Node--------------------------------------------

from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

# Grading Node Function
def grading_node(state: Dict[str, Any], question_index: int) -> Dict[str, Any]:
    """
    Grades a specific question in the student's submission by comparing it with the solution key.
    Updates the state with the grade and feedback for the question.
    """
    try:
        # Extract the student's answer and solution for the question
        student_answer = state["student_submission"][question_index]["question"]
        solution = state["solution_key"]["questions"][question_index]
        max_score = state["solution_key"]["scores"][question_index]
        
        # Simulate grading logic (e.g., compare student answer with solution)
        # This can be replaced with more advanced grading logic (e.g., LLM-based grading)
        if student_answer.strip().lower() == solution.strip().lower():
            grade = max_score  # Full marks if the answer matches the solution
            feedback = "Correct answer!"
        else:
            grade = max_score * 0.5  # Partial marks if the answer is partially correct
            feedback = "Partially correct. Review the solution for better understanding."
        
        # Update the state with the grade and feedback
        state["student_grades"][f"question_{question_index + 1}"] = grade
        state["feedback_comments"].append(f"Question {question_index + 1}: {feedback}")
        return state
    except Exception as e:
        state["feedback_comments"].append(f"Error grading question {question_index + 1}: {str(e)}")
        return state

# Parallel Grading Function
def parallel_grading(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grades all questions in parallel using multiple grading nodes.
    Updates the state with grades and feedback for all questions.
    """
    try:
        with ThreadPoolExecutor() as executor:
            # Submit grading tasks for all questions
            futures = [
                executor.submit(grading_node, state, i)
                for i in range(len(state["student_submission"]))
            ]
            # Wait for all tasks to complete
            for future in futures:
                state = future.result()
        return state
    except Exception as e:
        state["feedback_comments"].append(f"Error during parallel grading: {str(e)}")
        return state




#--------------------------------------------Reflection Node--------------------------------------------

def reflection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the graded answers and verifies the quality of grading.
    Reroutes for regrading if discrepancies are found.
    """
    try:
        # Check if all questions have been graded
        if len(state["student_grades"]) != len(state["student_submission"]):
            state["feedback_comments"].append("Not all questions have been graded. Rerouting to grading nodes.")
            state["next_node"] = "grading_node"
            return state
        
        # Simulate reflection logic (e.g., check for inconsistencies in grading)
        # This can be replaced with more advanced reflection logic (e.g., LLM-based analysis)
        total_score = sum(state["student_grades"].values())
        max_possible_score = sum(state["solution_key"]["scores"])
        if total_score < max_possible_score * 0.5:  # Example threshold for regrading
            state["feedback_comments"].append("Low overall score detected. Rerouting for regrading.")
            state["next_node"] = "grading_node"
        else:
            state["feedback_comments"].append("Grading quality verified. Proceeding to insights node.")
            state["next_node"] = "insights_node"
        return state
    except Exception as e:
        state["feedback_comments"].append(f"Error during reflection: {str(e)}")
        state["next_node"] = "__end__"
        return state






#--------------------------------------------Insights and Solution Node--------------------------------------------

def insights_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identifies weak points in the student's answers and prepares personalized notes or solutions.
    Updates the state with insights and final output.
    """
    try:
        # Simulate insights generation (e.g., identify weak points and provide solutions)
        weak_points = []
        for i, (question, grade) in enumerate(state["student_grades"].items()):
            max_score = state["solution_key"]["scores"][i]
            if grade < max_score * 0.8:  # Example threshold for weak points
                weak_points.append(f"Question {i + 1}: Score {grade}/{max_score}. Review the solution: {state['solution_key']['questions'][i]}")
        
        # Prepare personalized notes
        if weak_points:
            state["reflection"] = "Weak points identified:\n" + "\n".join(weak_points)
        else:
            state["reflection"] = "Great job! No significant weak points identified."
        
        # Update the final output
        state["final_output"] = {
            "student_grades": state["student_grades"],
            "feedback_comments": state["feedback_comments"],
            "reflection": state["reflection"],
        }
        state["next_node"] = "__end__"
        return state
    except Exception as e:
        state["feedback_comments"].append(f"Error during insights generation: {str(e)}")
        state["next_node"] = "__end__"
        return state





#--------------------------------------------Reporting Node--------------------------------------------

from openpyxl import Workbook
from typing import Dict, Any
import os

def reporting_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compiles grades and feedback into an Excel sheet.
    Saves the Excel file and updates the state with the file path.
    """
    try:
        # Create a new Excel workbook
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Grades and Feedback"
        
        # Add headers
        sheet.append(["Question", "Grade", "Feedback"])
        
        # Add grades and feedback
        for i, (question, grade) in enumerate(state["student_grades"].items()):
            feedback = state["feedback_comments"][i]
            sheet.append([question, grade, feedback])
        
        # Save the Excel file
        report_path = os.path.join("reports", f"{state['student_id']}_grades.xlsx")
        os.makedirs("reports", exist_ok=True)
        workbook.save(report_path)
        
        # Update the state with the report path
        state["report_path"] = report_path
        state["next_node"] = "notification_node"
        return state
    except Exception as e:
        state["feedback_comments"].append(f"Error during report generation: {str(e)}")
        state["next_node"] = "__end__"
        return state





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