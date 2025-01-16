"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from PyPDF2 import PdfWriter, PdfReader
import os
from react_agent.configuration import Configuration
from pdf2image import convert_from_path
import pytesseract
from langdetect import detect
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict
from pymongo import MongoClient

# async def search(
#     query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
# ) -> Optional[list[dict[str, Any]]]:
#     """Search for general web results.

#     This function performs a search using the Tavily search engine, which is designed
#     to provide comprehensive, accurate, and trusted results. It's particularly useful
#     for answering questions about current events.
#     """
#     configuration = Configuration.from_runnable_config(config)
#     wrapped = TavilySearchResults(max_results=configuration.max_search_results)
#     result = await wrapped.ainvoke({"query": query})
#     return cast(list[dict[str, Any]], result)


# Define Input Schema
class SplitPDFInput(BaseModel):
    input_pdf_path: str = Field(..., description="Path to the input PDF file.")
    page_ranges: List[str] = Field(..., description="List of page ranges to split the PDF into. Example: ['1-5', '6-10'].")
    output_dir: Optional[str] = Field(default="output", description="Directory to save the split PDF files. Default is 'output'.")

# Create the Tool
@tool("split_pdf_tool", args_schema=SplitPDFInput, return_direct=True)
def split_pdf_tool(input_pdf_path: str, page_ranges: List[str], output_dir: str = "output") -> str:
    """
    Splits a multi-page PDF into smaller PDFs based on the provided page ranges.
    Each range will create a separate PDF file.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Read the input PDF
        input_pdf = PdfReader(input_pdf_path)
        total_pages = len(input_pdf.pages)

        # Process each page range
        for i, page_range in enumerate(page_ranges):
            start_page, end_page = map(int, page_range.split('-'))
            
            # Validate page range
            if start_page < 1 or end_page > total_pages or start_page > end_page:
                return f"Invalid page range: {page_range}. Ensure the range is within 1-{total_pages}."

            # Create a new PDF writer for the current range
            output_pdf = PdfWriter()
            for page_num in range(start_page - 1, end_page):  # PyPDF2 uses 0-based indexing
                output_pdf.add_page(input_pdf.pages[page_num])

            # Save the split PDF
            output_file_path = os.path.join(output_dir, f"split_pdf_{i + 1}.pdf")
            with open(output_file_path, "wb") as output_stream:
                output_pdf.write(output_stream)

        return f"PDF successfully split into {len(page_ranges)} files. Saved in '{output_dir}'."
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

#----------------------------------------------------------------------------------------------------------------------------

# Define Input Schema
class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up.")
    config: Annotated[Configuration, InjectedToolArg] = Field(
        default_factory=Configuration,
        description="Configuration for the search tool, including max_results."
    )

# Create the Tool
@tool("search_tool", args_schema=SearchToolInput, return_direct=True)
async def search_tool(
    query: str, 
    config: Annotated[Configuration, InjectedToolArg] = Configuration()
) -> Optional[List[Dict[str, Any]]]:
    """
    Search for general web results using the Tavily search engine.
    This tool is designed to provide comprehensive, accurate, and trusted results.
    It's particularly useful for answering questions about current events.
    """
    try:
        # Extract max_results from the config
        max_results = config.max_search_results

        # Initialize the Tavily search tool with the configured max_results
        search_tool = TavilySearchResults(max_results=max_results)

        # Perform the search
        result = await search_tool.ainvoke({"query": query})

        # Return the search results
        return result
    except Exception as e:
        return {"error": str(e)}   
    
    
#----------------------------------------------------------------------------------------------------------------------------   
    
# Define Input Schema
class OCRToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the image or PDF file.")
    language: Optional[str] = Field(default="eng", description="Language for OCR (e.g., 'eng' for English).")
    clean_text: Optional[bool] = Field(default=True, description="Whether to clean the extracted text.")

# Create the Tool
@tool("advanced_ocr_tool", args_schema=OCRToolInput, return_direct=True)
def advanced_ocr_tool(file_path: str, language: str = "eng", clean_text: bool = True) -> Dict[str, Any]:
    """
    Advanced OCR tool to extract text from images or PDFs.
    Supports language detection, text cleaning, and structured output.
    """
    try:
        # Step 1: Extract text from the file
        if file_path.lower().endswith(".pdf"):
            # Convert PDF to images
            images = convert_from_path(file_path)
            extracted_text = ""
            for image in images:
                extracted_text += pytesseract.image_to_string(image, lang=language) + "\n"
        else:
            # Extract text from image
            extracted_text = pytesseract.image_to_string(file_path, lang=language)

        # Step 2: Clean the text (if enabled)
        if clean_text:
            extracted_text = re.sub(r"\s+", " ", extracted_text)  # Remove extra whitespace
            extracted_text = extracted_text.strip()  # Remove leading/trailing spaces

        # Step 3: Detect the language of the extracted text
        detected_language = detect(extracted_text) if extracted_text else "unknown"

        # Step 4: Return structured output
        return {
            "extracted_text": extracted_text,
            "detected_language": detected_language,
            "file_path": file_path,
            "language_used_for_ocr": language,
            "clean_text": clean_text,
        }
    except Exception as e:
        return {"error": str(e)}
       

#-------------------------------------------------------------------------------------------------------------------------
class RetrieveSolutionKeyInput(BaseModel):
    course_id: str = Field(..., description="ID of the course.")
    assignment_id: str = Field(..., description="ID of the assignment.")

@tool("retrieve_solution_key_tool", args_schema=RetrieveSolutionKeyInput, return_direct=True)
def retrieve_solution_key_tool(course_id: str, assignment_id: str) -> List[Dict[str, str]]:
    """
    Retrieves all questions and solutions for a specific assignment from the database.
    Returns a list of dictionaries, where each dictionary contains the text for a question and solution.
    """
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/")
        db = client["GradeMaster"]
        solution_keys = db.solution_keys

        # Query the database for all questions and solutions for the assignment
        results = solution_keys.find(
            {"course_id": course_id, "assignment_id": assignment_id},
            {"_id": 0, "text": 1, "scores": 1},  # Exclude MongoDB's _id field
        )

        # Convert the results to a list of dictionaries
        questions_and_solutions = [
            {"text": result["text"], "scores": result["scores"]} for result in results
        ]

        if not questions_and_solutions:
            return [{"error": f"No solution key found for assignment {assignment_id}."}]

        return questions_and_solutions
    except Exception as e:
        return [{"error": str(e)}]

        
# List of available tools
TOOLS: List[Callable[..., Any]] = [search_tool, split_pdf_tool, advanced_ocr_tool, retrieve_solution_key_tool]