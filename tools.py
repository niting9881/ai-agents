from langchain.tools import tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage, ToolMessage
load_dotenv()
import json


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city
    """
    # Mock implementation
    weather_data = {
        "bangalore": "Sunny, 28°C",
        "mumbai": "Rainy, 26°C",
        "delhi": "Cloudy, 22°C"
    }
    return weather_data.get(city.lower(), "Weather data not available")


@tool
def book_flight(origin: str, destination: str, date: str) -> dict:
    """Book a flight from one city to another.

    Args:
        origin: The origin city
        destination: The destination city
        date: The date of the flight
    """
    # Mock implementation
    return {
        "booking_id": "1234567890",
        "route": f"{origin} to {destination}",
        "date": date,
        "status": "confirmed"
    }

tools = [get_weather, book_flight]
# llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5) #initializing the google model
# llm_google.bind_tools(tools)
llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)
llm_openai.bind_tools(tools)

def run(prompt: str) -> str:
    response = llm_openai.invoke([HumanMessage(content=prompt)])
    print("Model Output: ", response.content)
    print("Model Tools Calls:", response.tool_calls)
    print("*************************************************")
    print("Model Full output: ", response)
    print("*************************************************")
    for call in response.tool_calls:
        print(f"Tool called: {call.tool_name} with arguments {call.arguments}")
    
    return response.content
run("What's the weather in mumbai?")
#run("Book a flight from New York to San Francisco on 2024-07-15.")