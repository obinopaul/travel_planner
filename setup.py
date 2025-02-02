from setuptools import setup, find_packages

setup(
    name="Pocket Travel",  # Replace with your project name
    version="0.1.0",  # Replace with your project version
    description="Travel Iterinery Traveller Agent that gives tailored travel recommendations, flights, and accomodations bookings etc.",  # Replace with a description
    author="Paul Okafor",  # Replace with your name
    author_email="acobapaul@gmail.com",  # Replace with your email
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        # List your dependencies here, e.g.:
        "langgraph",
        "numpy",
        "pandas",
        "langchain",
        "langchain_core",
        "langchain_anthropic",
        "langchain_openai",
        "tavily_python",
        "langchain_community",
        "langchain-groq",
        "langgraph",
        "pyairbnb",
        "geopy",
        "python-dotenv",
        "fast-flights",
        "pydantic",
        "langchain_openai",
        "PyPDF2",
        "amadeus",
        "beautifulsoup4",
        "googlemaps",
        "docling",
        "arxiv",
        "needle-python"
        # Add other dependencies from requirements.txt
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],  # Optional: Add development dependencies
    },
    python_requires=">=3.8",  # Specify the Python version requirement
)