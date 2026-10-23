import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

class Exercise(BaseModel):
    name: str = Field(description="Name of exercise")
    sets: int = Field(description="Number of sets")
    reps: str = Field(description="Rep range or duration")
    target_muscle: str = Field(description="Primary muscle group")

class WorkoutDay(BaseModel):
    day_title: str = Field(description="e.g. Day 1: Upper Body Push")
    exercises: List[Exercise] = Field(description="List of planned exercises")

class WeeklyPlan(BaseModel):
    program_goal: str = Field(description="Primary fitness objective")
    routine: List[WorkoutDay] = Field(description="Daily exercise schedule")

parser = PydanticOutputParser(pydantic_object=WeeklyPlan)

prompt = PromptTemplate(
    template="""You are a certified strength & conditioning specialist.
Build a safe, progressive workout regimen tailored to the following profile.
{format_instructions}

Client Profile:
- Goal: {goal}
- Experience: {level}
- Weekly Availability: {days_per_week} days
- Equipment: {equipment}
""",
    input_variables=["goal", "level", "days_per_week", "equipment"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

if __name__ == "__main__":
    print("=== Lab 5 - Exercise 6: Structured Fitness Routine Generator ===")
    plan = chain.invoke({
        "goal": "Hypertrophy & Core Stability",
        "level": "Intermediate",
        "days_per_week": "3",
        "equipment": "Dumbbells and Pull-up bar only"
    })
    
    print(f"Goal: {plan.program_goal}\n")
    for day in plan.routine:
        print(f"--- {day.day_title} ---")
        for ex in day.exercises:
            print(f"  • {ex.name} ({ex.target_muscle}): {ex.sets} sets x {ex.reps}")
        print()