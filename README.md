# GradeMaster AI: Multi-Agent Intelligent Grading System

![GradeMaster AI Logo](https://via.placeholder.com/150)  
**Automate Grading, Provide Feedback, and Generate Insights with AI**

---
## Introduction

**GradeMaster AI** is an advanced, multi-agent AI system designed to automate and enhance the grading process for educational assignments. It leverages **LangGraph**, **LangChain**, and **Large Language Models (LLMs)** to create a robust, intelligent workflow for grading student submissions, providing feedback, and generating insights.

Whether you're grading PDFs, handwritten notes, or CSV files, GradeMaster AI ensures accuracy, efficiency, and personalized feedback for every student.

---

## Features

- **Automated Grading**: Grades student submissions by comparing them with solution keys.
- **Multi-Agent Workflow**: Uses specialized agents (e.g., Grading Agents, Reflection Agent, Insights Agent) for accurate and efficient grading.
- **Feedback and Insights**: Provides detailed feedback and identifies weak points in student submissions.
- **Reflection and Verification**: Ensures grading quality by rerouting submissions for regrading if necessary.
- **Reporting and Notification**: Compiles grades and feedback into Excel sheets and sends them to students via email.
- **State Management**: Tracks all intermediate and final results using a centralized `OverallState`.

---
---
### THIS PROJECT IS STILL IN PRODUCTION
---
---

## Installation

### Prerequisites
- Python 3.8 or higher

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/grade-master-ai.git
   cd grade-master-ai
  ```

2. Install dependencies from the rewuirements.txt file:
   ```bash
   pip install requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add the following variables:
     ```bash
     DEEPSEEK_API_KEY=your_deepseek_api_key
     ```


## Usage

### Running the Workflow


1. Run the main workflow:
   ```bash
   python main.py
   ```
"# grade_master" 
