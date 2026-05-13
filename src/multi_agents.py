import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv

load_dotenv()

PASTA_PROCESSADOS = os.path.join("..", "data", "processed")
PASTA_OUTPUT = os.path.join("..", "data", "outputs", "multi_agent")
os.makedirs(PASTA_OUTPUT, exist_ok=True)