import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)

extraction_template = """You are an automated Applicant Tracking System (ATS) parser.
Extract the relevant candidate attributes from the unformatted resume text.
Output MUST be valid JSON only with keys: "full_name", "total_years_experience", "primary_role", "technical_skills", "certifications", "education".

Resume:
{raw_resume}

JSON Output:"""

prompt = PromptTemplate.from_template(extraction_template)
chain = prompt | llm

sample_resume = """
Kavita Deshmukh - Senior DevOps & Cloud Infrastructure Engineer
Phone: +91-9823000000 | LinkedIn: linkedin.com/in/kavita-deshmukh
Professional Summary: Over 7 years of production experience architecting scalable Kubernetes clusters and automated CI/CD pipelines across AWS and GCP environments.
Core Competencies: Kubernetes, Docker, Terraform, Helm, Prometheus, Grafana, Python, Bash, AWS EKS, GitLab CI.
Work History:
- Lead DevOps Specialist at CloudScale Tech (2022 - Present): Designed multi-region disaster recovery setups and reduced cloud compute bills by 32%.
- Cloud Engineer at FinEdge Solutions (2019 - 2022): Migrated monolithic banking applications into containerized microservices.
Certifications: AWS Certified Solutions Architect Professional, Certified Kubernetes Administrator (CKA).
Education: B.E. in Information Technology, Savitribai Phule Pune University, 2019.
"""

if __name__ == "__main__":
    print("=== Lab 3 - Exercise 3: Zero-Shot ATS Data Extraction ===")
    result = chain.invoke({"raw_resume": sample_resume})
    print(result.content)