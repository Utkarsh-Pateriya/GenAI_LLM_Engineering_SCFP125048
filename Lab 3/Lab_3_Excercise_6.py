import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4
)

# Step 1: Draft generator
draft_prompt = ChatPromptTemplate.from_template(
    "Write an introductory email pitching an automated cloud cost optimization SaaS to a CTO."
)

# Step 2: Critic prompt
critic_prompt = ChatPromptTemplate.from_template(
    """Analyze the following sales pitch email:
"{draft}"

Evaluate against three criteria:
1. Is the value proposition quantified with concrete metrics?
2. Is the call-to-action low friction?
3. Does it sound like generic corporate spam?
List concise suggestions for improvement."""
)

# Step 3: Refiner prompt
refine_prompt = ChatPromptTemplate.from_template(
    """Original Email:
{draft}

Critique & Suggestions:
{critique}

Rewrite the email incorporating all critique points. Make it succinct, numbers-driven, and high-impact."""
)

parser = StrOutputParser()

if __name__ == "__main__":
    print("=== Lab 3 - Exercise 6: Iterative Draft -> Critique -> Refine Loop ===")
    
    # 1. Draft
    draft = (draft_prompt | llm | parser).invoke({})
    print(f"\n--- Initial Draft ---\n{draft}")
    
    # 2. Critique
    critique = (critic_prompt | llm | parser).invoke({"draft": draft})
    print(f"\n--- Critique Feedback ---\n{critique}")
    
    # 3. Refine
    final_email = (refine_prompt | llm | parser).invoke({"draft": draft, "critique": critique})
    print(f"\n--- Polished Final Email ---\n{final_email}")