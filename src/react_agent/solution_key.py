from pymongo import MongoClient
from PyPDF2 import PdfReader
from typing import List, Optional, Dict, Any
from datetime import datetime
import os

class SolutionKeySaver:
    def __init__(self, database_name: str):
        """
        Initializes the SolutionKeySaver with a connection to the MongoDB database.
        """
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[database_name]
        self.solution_keys = self.db.solution_keys

    def split_and_save_solution_key(
        self,
        course_id: str,
        assignment_id: str,
        solution_key_pdf_path: str,
        page_ranges: List[str],
        scores: List[Dict[str, int]],
    ) -> None:
        """
        Splits the solution key PDF into individual questions and solutions based on page ranges,
        extracts the text, and saves it to the database along with the scores.
        """
        try:
            # Validate the PDF file path
            if not os.path.exists(solution_key_pdf_path):
                raise FileNotFoundError(f"The file {solution_key_pdf_path} does not exist.")

            # Read the solution key PDF
            reader = PdfReader(solution_key_pdf_path)

            # Iterate through the page ranges and extract text for each question
            for i, page_range in enumerate(page_ranges):
                try:
                    # Parse the page range
                    start_page, end_page = map(int, page_range.split("-"))

                    # Validate the page range
                    if start_page < 1 or end_page > len(reader.pages) or start_page > end_page:
                        raise ValueError(f"Invalid page range: {page_range}. Valid range is 1-{len(reader.pages)}.")

                    # Extract text from the specified pages
                    solution_key_text = ""
                    for page_num in range(start_page - 1, end_page):  # PyPDF2 uses 0-based indexing
                        page_text = reader.pages[page_num].extract_text()
                        if page_text:
                            solution_key_text += page_text.strip() + "\n"

                    # Save the question and solution to the database
                    self.solution_keys.insert_one(
                        {
                            "course_id": course_id,
                            "assignment_id": assignment_id,
                            "question_number": i + 1,  # Optional: Can be removed if not needed
                            "text": solution_key_text.strip(),  # Save the entire text for the question and solution
                            "scores": scores[i],  # Save the scores for the question
                            "timestamp": datetime.utcnow(),  # Track when the solution key was saved
                        }
                    )
                except ValueError as ve:
                    print(f"Skipping invalid page range {page_range}: {ve}")
                except Exception as e:
                    print(f"An error occurred while processing page range {page_range}: {e}")

            print("Solution key saved successfully!")
        except FileNotFoundError as fe:
            print(f"File error: {fe}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
            
            
# Usage Example
if __name__ == "__main__":
    # Initialize the SolutionKeySaver
    saver = SolutionKeySaver(database_name="GradeMaster")

    # Example data
    course_id = "CS501"
    assignment_id = "CS501_A1"
    solution_key_pdf_path = "solution_key.pdf"
    page_ranges = ["1-2", "3-4", "5-6"]  # Page ranges for each question

    # Example scores for each question
    scores = [
        [{"part": "Q1a", "score": 5}, {"part": "Q1b", "score": 10}, {"part": "Q1c", "score": 5}],  # Scores for Q1
        [{"part": "Q2a", "score": 5}, {"part": "Q2b", "score": 5}],  # Scores for Q2
        [{"part": "Q3", "score": 20}],  # Scores for Q3
    ]
    
    # Save the solution key
    saver.split_and_save_solution_key(course_id, assignment_id, solution_key_pdf_path, page_ranges, scores)
    