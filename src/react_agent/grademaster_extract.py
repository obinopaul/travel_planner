import os
from pdf2image import convert_from_path  # For PDFs
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from pptx import Presentation  # For PPTs
from PIL import Image  # For image handling
import io
import base64
import concurrent.futures
import openai  # For OpenAI's GPT-4 Vision

class DocumentProcessor:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize the document processor with LLM configuration.
        
        :param api_key: API key for the LLM service
        :param model_name: Name of the vision-capable LLM model
        """
        openai.api_key = api_key
        self.model_name = model_name

    def split_document(self, file_path: str) -> List[str]:
        """
        Split the document into page images without reading content.
        
        :param file_path: Path to PDF or PPT file
        :return: List of base64-encoded page images
        """
        page_images = []
        if file_path.endswith(".pdf"):
            # Convert PDF pages to images
            images = convert_from_path(file_path, fmt="png")
            for img in images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                page_images.append(base64.b64encode(img_byte_arr.getvalue()).decode("utf-8"))
        elif file_path.endswith(".pptx"):
            # Convert PPT slides to images
            prs = Presentation(file_path)
            for slide in prs.slides:
                slide_img = io.BytesIO()
                slide.save(slide_img, format="PNG")
                img = Image.open(slide_img)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                page_images.append(base64.b64encode(img_byte_arr.getvalue()).decode("utf-8"))
        else:
            raise ValueError("Unsupported file format. Use PDF or PPTX.")
        return page_images

    def process_page_with_llm(self, page_image: str) -> Dict[str, Any]:
        """
        Process a single page image through the LLM.
        
        :param page_image: Base64-encoded image of a document page
        :return: LLM analysis of the page
        """
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze the content of this page in detail."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_image}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            return {
                "page_content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens
            }
        except Exception as e:
            return {"error": str(e)}

    def process_document(self, file_path: str, max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Process entire document by splitting pages and analyzing in parallel.
        
        :param file_path: Path to document file
        :param max_workers: Maximum number of parallel workers
        :return: List of page analysis results
        """
        # Split document into page images
        page_images = self.split_document(file_path)
        
        # Process pages in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit page processing tasks
            future_to_page = {
                executor.submit(self.process_page_with_llm, page): page 
                for page in page_images
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        
        return results

# Example usage
def main():
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    processor = DocumentProcessor(api_key=api_key)
    file_path = "Paul Okafor - Travel Itinerary Planner.pdf"  # Replace with your PDF or PPT file path
    
    page_analyses = processor.process_document(file_path)
    
    for i, analysis in enumerate(page_analyses, 1):
        print(f"Page {i} Analysis: {analysis}")

if __name__ == "__main__":
    main()