import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Hospital Attending Physician.
Transform clinical notes into a professional Discharge Summary.
Include exactly four sections:
1. Primary Diagnosis & Severity
2. Key Clinical Findings
3. Prescribed Discharge Medications (with dosages)
4. Critical Red-Flag Warning Signs for Emergency Care

Do not fabricate information not present in the transcript."""),
    ("user", "Clinical Notes:\n{clinical_notes}")
])

chain = prompt_template | llm

clinical_notes = """
Patient is a 54-year-old male admitted 48 hours ago presenting with severe retrosternal 
chest tightness, diaphoresis, and shortness of breath upon exertion. Troponin T levels peaked 
at 1.4 ng/mL, ECG revealed ST elevation in anterior leads. Bedside echocardiogram showed 
mild left ventricular anterior wall hypokinesia with an estimated LVEF of 48%. Successfully 
underwent primary percutaneous coronary intervention (PCI) with a drug-eluting stent placed in the proximal LAD. 
Vitals stable post-procedure: BP 118/76, HR 68 bpm. Initiated on Aspirin 81mg daily, 
Ticagrelor 90mg BID, Atorvastatin 80mg QHS, and Metoprolol Succinate 25mg daily. Patient ambulating comfortably.
"""

if __name__ == "__main__":
    print("=== Lab 3 - Exercise 1: Clinical Discharge Summary ===")
    response = chain.invoke({"clinical_notes": clinical_notes})
    print(response.content)