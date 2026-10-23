import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

translation_template = """You are an international localization and translation expert.
Translate the following marketing message into {target_language}.
Do not perform a literal word-by-word translation; adapt the tone, cultural idioms, 
and rhetorical hooks so it feels naturally written for native speakers in that market.

Provide:
1. Natural Localization
2. Explanation of Cultural/Idiomatic Adjustments Made

Source English Copy:
\"{source_text}\"
"""

prompt = PromptTemplate.from_template(translation_template)
chain = prompt | llm

copy = "Don't let opportunity pass you by! Grab the bull by the horns and skyrocket your small business today."

if __name__ == "__main__":
    print("=== Lab 3 - Exercise 5: Idiomatic Cross-Cultural Localization ===")
    for lang in ["Hindi", "German", "Japanese"]:
        print(f"\n--- Target: {lang} ---")
        output = chain.invoke({"target_language": lang, "source_text": copy})
        print(output.content)