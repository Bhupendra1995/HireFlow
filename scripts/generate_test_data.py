import os
import random
from faker import Faker
from fpdf import FPDF

fake = Faker('en_US')
random.seed(42)
fake.seed_instance(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESUMES_DIR = os.path.join(BASE_DIR, "data", "resumes")
JDS_DIR = os.path.join(BASE_DIR, "data", "job_descriptions")

UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "UC Berkeley", "Georgia Institute of Technology",
    "University of Michigan", "University of Texas Austin",
    "Cornell University", "University of Washington",
    "Purdue University", "UIUC", "Duke University",
    "Johns Hopkins University", "Columbia University",
    "University of Southern California", "NYU", "UCLA",
    "Boston University", "University of Toronto", "IIT Bombay",
]

CERTIFICATIONS = {
    "ML Engineer": [
        "AWS Certified Machine Learning Specialty",
        "TensorFlow Developer Certificate",
        "Deep Learning Specialization - Coursera",
        "Hugging Face NLP Course Certificate",
    ],
    "DevOps Engineer": [
        "AWS Solutions Architect Associate",
        "Certified Kubernetes Administrator (CKA)",
        "HashiCorp Terraform Associate",
        "Google Professional Cloud DevOps Engineer",
    ],
    "Frontend Developer": [
        "Meta Frontend Developer Professional Certificate",
        "Google UX Design Certificate",
        "AWS Cloud Practitioner",
    ],
    "Backend Developer": [
        "AWS Developer Associate",
        "MongoDB Certified Developer Associate",
        "Google Professional Cloud Developer",
    ],
    "Product Manager": [
        "Certified Scrum Product Owner (CSPO)",
        "Google Project Management Certificate",
        "AWS Cloud Practitioner",
        "Product School - Product Manager Certification",
    ],
}

DOMAINS = {
    "ML Engineer": {
        "skills_pool": [
            "Python", "PyTorch", "TensorFlow", "scikit-learn", "LangChain",
            "LlamaIndex", "Hugging Face Transformers", "MLflow", "Pandas",
            "NumPy", "FAISS", "RAG pipelines", "NLP", "Deep Learning",
            "AWS SageMaker", "Vertex AI", "Gemini API", "OpenAI API",
            "Pinecone", "Prompt Engineering", "Fine-tuning LLMs", "Vector databases",
        ],
        "job_titles": ["ML Engineer", "Data Scientist", "AI Engineer", "NLP Engineer", "Research Engineer"],
        "degree": "B.S./M.S. in Computer Science, Data Science, or AI",
    },
    "DevOps Engineer": {
        "skills_pool": [
            "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
            "Ansible", "Jenkins", "GitHub Actions", "CircleCI", "Prometheus",
            "Grafana", "Linux", "Bash scripting", "CI/CD pipelines",
            "Helm", "ArgoCD", "EKS", "CloudFormation", "HashiCorp Vault",
            "Datadog", "Nginx", "Istio",
        ],
        "job_titles": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Platform Engineer"],
        "degree": "B.S. in Computer Science or Information Technology",
    },
    "Frontend Developer": {
        "skills_pool": [
            "React", "TypeScript", "JavaScript", "Next.js", "Vue.js",
            "Redux", "Zustand", "TailwindCSS", "CSS3", "HTML5",
            "Jest", "Cypress", "React Testing Library", "Webpack", "Vite",
            "GraphQL", "REST APIs", "Storybook", "Figma", "Web Performance",
            "React Query", "Accessibility (WCAG)",
        ],
        "job_titles": ["Frontend Developer", "React Developer", "UI Engineer", "Frontend Engineer"],
        "degree": "B.S. in Computer Science or Software Engineering",
    },
    "Backend Developer": {
        "skills_pool": [
            "Python", "FastAPI", "Django", "Flask", "PostgreSQL",
            "MongoDB", "Redis", "Docker", "Celery", "SQLAlchemy",
            "Microservices", "AWS Lambda", "RabbitMQ", "Kafka", "gRPC",
            "JWT", "OAuth2", "Pytest", "ElasticSearch", "REST APIs",
            "GraphQL", "SQLite",
        ],
        "job_titles": ["Backend Developer", "Python Developer", "Software Engineer", "API Developer"],
        "degree": "B.S. in Computer Science or Software Engineering",
    },
    "Product Manager": {
        "skills_pool": [
            "Product Roadmap", "Agile", "Scrum", "JIRA", "Stakeholder Management",
            "User Research", "A/B Testing", "SQL", "Google Analytics", "Figma",
            "OKRs", "Go-to-market Strategy", "Competitive Analysis", "Confluence",
            "Product Discovery", "Feature Prioritization", "Sprint Planning",
            "Customer Journey Mapping", "Data-driven Decision Making",
            "Cross-functional Leadership", "Mixpanel",
        ],
        "job_titles": ["Product Manager", "Senior Product Manager", "Associate PM", "Technical PM"],
        "degree": "B.S. in Business, Computer Science, or Engineering",
    },
}

JD_TEMPLATES = {
    "ML Engineer": {
        "title": "Senior ML / AI Engineer",
        "about": (
            "We are building the next generation of AI-powered products and are looking for a "
            "Senior ML/AI Engineer to join our core AI team. You will design and ship production-grade "
            "ML pipelines, work on LLM applications including RAG systems, and collaborate with "
            "cross-functional teams to bring AI features to millions of users."
        ),
        "responsibilities": [
            "Design and build end-to-end ML pipelines for production deployment on AWS SageMaker and Vertex AI",
            "Develop and fine-tune large language models (LLMs) for NLP and document understanding tasks",
            "Implement Retrieval-Augmented Generation (RAG) systems using LlamaIndex and LangChain",
            "Track and optimize model experiments using MLflow and custom evaluation metrics",
            "Build semantic search systems using FAISS and Pinecone vector databases",
            "Research and implement state-of-the-art NLP techniques including embeddings and transformers",
            "Evaluate models using domain-specific metrics (BLEU, ROUGE, faithfulness, context precision)",
            "Mentor junior ML engineers and drive technical design reviews",
        ],
        "required_skills": [
            "5+ years of Python programming with a strong ML/AI focus",
            "Hands-on experience with PyTorch or TensorFlow and model fine-tuning",
            "Proven experience with Hugging Face Transformers and pre-trained LLMs",
            "Solid understanding of RAG pipelines, vector databases (FAISS, Pinecone)",
            "Experience with LangChain or LlamaIndex for LLM orchestration",
            "Familiarity with cloud ML platforms: AWS SageMaker or Vertex AI",
            "Strong NLP fundamentals: embeddings, attention mechanisms, transformers",
        ],
        "nice_to_have": [
            "Experience with prompt engineering, RLHF, and LLM alignment techniques",
            "Knowledge of MLOps best practices and model monitoring in production",
            "Open-source contributions to ML frameworks or NLP libraries",
        ],
    },
    "DevOps Engineer": {
        "title": "Senior DevOps / Cloud Engineer",
        "about": (
            "We are scaling our platform infrastructure and need a Senior DevOps/Cloud Engineer "
            "to own our cloud architecture, CI/CD pipelines, and reliability engineering. You will "
            "work across AWS, GCP, and Azure, ensuring our systems are highly available, secure, "
            "and cost-efficient at scale."
        ),
        "responsibilities": [
            "Design, provision, and maintain scalable cloud infrastructure on AWS and GCP using Terraform",
            "Build and manage CI/CD pipelines using GitHub Actions, Jenkins, and ArgoCD",
            "Containerize applications using Docker and manage Kubernetes clusters (EKS/GKE)",
            "Implement Infrastructure as Code (IaC) with Terraform and CloudFormation",
            "Set up observability stacks using Prometheus, Grafana, and Datadog",
            "Define and maintain SLOs, SLAs, and incident response playbooks",
            "Manage secrets and access control with HashiCorp Vault and AWS IAM",
            "Collaborate with engineering teams to optimize deployment pipelines and reduce MTTR",
        ],
        "required_skills": [
            "4+ years of DevOps or Cloud Engineering experience",
            "Deep expertise in AWS (EC2, EKS, ECS, RDS, S3, Lambda, IAM)",
            "Strong hands-on experience with Docker and Kubernetes",
            "Proficiency with Terraform for cloud infrastructure provisioning",
            "Experience designing and maintaining CI/CD pipelines (GitHub Actions, Jenkins)",
            "Strong Linux administration and Bash scripting skills",
            "Experience with monitoring and alerting: Prometheus, Grafana, Datadog",
        ],
        "nice_to_have": [
            "AWS Solutions Architect or CKA certification",
            "Experience with service mesh technologies (Istio, Linkerd)",
            "Knowledge of GitOps principles and ArgoCD workflows",
        ],
    },
    "Frontend Developer": {
        "title": "Senior Frontend Developer (React / TypeScript)",
        "about": (
            "We are looking for a Senior Frontend Developer to lead the development of our "
            "customer-facing web applications. You will work closely with designers and backend "
            "engineers to build performant, accessible, and beautiful React/TypeScript applications "
            "used by thousands of users daily."
        ),
        "responsibilities": [
            "Build responsive and accessible web applications using React and TypeScript",
            "Architect and implement state management solutions using Redux or Zustand",
            "Write comprehensive unit and integration tests with Jest, Cypress, and React Testing Library",
            "Optimize application performance: Core Web Vitals, lazy loading, code splitting",
            "Build and maintain a shared component library documented with Storybook",
            "Integrate RESTful and GraphQL APIs with proper error handling and caching",
            "Ensure WCAG 2.1 accessibility compliance across all UI components",
            "Collaborate with UX designers translating Figma designs into pixel-perfect implementations",
        ],
        "required_skills": [
            "4+ years of frontend development experience",
            "Expert-level proficiency in React and TypeScript",
            "Strong understanding of modern JavaScript (ES2020+) and browser APIs",
            "Experience with Next.js for server-side rendering and static generation",
            "Proficiency with CSS frameworks: TailwindCSS or CSS-in-JS",
            "Experience integrating GraphQL and REST APIs",
            "Solid testing skills: Jest, React Testing Library, and Cypress",
        ],
        "nice_to_have": [
            "Experience with micro-frontend architecture",
            "Knowledge of Web Performance optimization and Lighthouse scoring",
            "Familiarity with design systems and component-driven development",
        ],
    },
    "Backend Developer": {
        "title": "Senior Python Backend Developer",
        "about": (
            "We are seeking a Senior Python Backend Developer to design and build the core APIs "
            "and services that power our platform. You will work on high-throughput microservices, "
            "database design, and event-driven architectures, collaborating closely with frontend "
            "engineers and ML teams."
        ),
        "responsibilities": [
            "Design and develop scalable RESTful and GraphQL APIs using FastAPI or Django REST Framework",
            "Build and maintain microservices architecture with Docker and Kubernetes",
            "Design and optimize PostgreSQL databases including query optimization and indexing",
            "Implement async task queues using Celery with Redis as message broker",
            "Write comprehensive unit and integration tests with Pytest following TDD practices",
            "Implement secure authentication and authorization using JWT and OAuth2",
            "Integrate message brokers (Kafka, RabbitMQ) for event-driven system components",
            "Define and review API contracts and collaborate with frontend and ML teams",
        ],
        "required_skills": [
            "5+ years of Python backend development experience",
            "Expert knowledge of FastAPI or Django REST Framework",
            "Deep expertise in PostgreSQL including query optimization and schema design",
            "Experience with Redis for caching, sessions, and pub/sub patterns",
            "Proficiency with Docker and containerized microservice deployments",
            "Strong understanding of event-driven architecture using Kafka or RabbitMQ",
            "Excellent Pytest skills and TDD/BDD practices",
        ],
        "nice_to_have": [
            "Experience with ElasticSearch for full-text search features",
            "Knowledge of AWS Lambda and serverless architecture patterns",
            "Familiarity with gRPC for inter-service communication",
        ],
    },
    "Product Manager": {
        "title": "Senior Product Manager (Tech Platform)",
        "about": (
            "We are looking for a Senior Product Manager to define the vision and roadmap for our "
            "core platform products. You will work at the intersection of business strategy, user "
            "research, and engineering execution, leading cross-functional teams to ship impactful "
            "features that delight users and drive company growth."
        ),
        "responsibilities": [
            "Define and own the product roadmap aligned with company OKRs and business strategy",
            "Lead product discovery through user research, interviews, and quantitative data analysis",
            "Write detailed PRDs, user stories, and acceptance criteria in Confluence and JIRA",
            "Drive A/B testing and experimentation frameworks to validate product hypotheses",
            "Analyze product metrics using SQL and Google Analytics to identify growth opportunities",
            "Manage stakeholder communication including executive-level presentations",
            "Lead Agile sprint planning, backlog grooming, and retrospective sessions",
            "Collaborate with design, engineering, marketing, and sales teams",
        ],
        "required_skills": [
            "5+ years of product management experience at a tech company",
            "Strong analytical skills with proficiency in SQL and data visualization tools",
            "Hands-on experience with Agile/Scrum and JIRA for team management",
            "Proven track record of successfully launching B2B or B2C products",
            "Experience with user research methods: interviews, surveys, usability testing",
            "Excellent communication skills for technical and non-technical audiences",
            "Data-driven mindset with experience in A/B testing and funnel optimization",
        ],
        "nice_to_have": [
            "Experience building AI/ML-powered product features",
            "Technical background enabling review of engineering specifications",
            "Experience with growth metrics and product-led growth strategies",
        ],
    },
}


def safe_text(text: str) -> str:
    """Replace non-latin-1 chars so fpdf core fonts don't error."""
    return text.encode('latin-1', 'replace').decode('latin-1')


def generate_experience(domain_name: str, domain_info: dict, num_jobs: int = 2) -> list:
    entries = []
    years_back = 0
    for i in range(num_jobs):
        duration = random.randint(1, 3)
        end_year = 2025 - years_back
        start_year = end_year - duration
        years_back += duration + random.randint(0, 1)

        skills_used = random.sample(domain_info["skills_pool"], min(4, len(domain_info["skills_pool"])))
        title = random.choice(domain_info["job_titles"])
        if i == 0 and random.random() > 0.4:
            title = "Senior " + title

        period = f"{start_year} - {'Present' if i == 0 else str(end_year)}"
        improvement = random.randint(15, 50)
        description = (
            f"Designed and delivered {domain_name.lower()} solutions using {skills_used[0]} "
            f"and {skills_used[1]}. Reduced operational overhead by {improvement}% through "
            f"automation and best practices with {skills_used[2] if len(skills_used) > 2 else skills_used[0]}. "
            f"Collaborated with cross-functional teams to deliver features on schedule and "
            f"mentored junior engineers in code reviews and technical design."
        )
        entries.append({
            "title": safe_text(title),
            "company": safe_text(fake.company()),
            "period": period,
            "description": safe_text(description),
        })
    return entries


def build_resume_data(domain_name: str, domain_info: dict) -> dict:
    num_jobs = random.randint(2, 3)
    skills = random.sample(domain_info["skills_pool"], min(10, len(domain_info["skills_pool"])))
    primary = skills[0]
    secondary = skills[1]
    total_years = random.randint(3, 10)

    summary = safe_text(
        f"Results-driven {domain_name} with {total_years}+ years of experience specializing in "
        f"{primary} and {secondary}. Passionate about building scalable, production-grade systems "
        f"and delivering measurable business impact. Strong communicator with a track record of "
        f"leading technical initiatives and collaborating across engineering, product, and design teams."
    )

    certs = random.sample(CERTIFICATIONS[domain_name], k=random.randint(1, 2))

    return {
        "name": safe_text(fake.name()),
        "email": fake.ascii_email(),
        "phone": fake.phone_number(),
        "location": safe_text(f"{fake.city()}, {fake.state_abbr()}"),
        "summary": summary,
        "experience": generate_experience(domain_name, domain_info, num_jobs),
        "skills": [safe_text(s) for s in skills],
        "education": [{
            "degree": safe_text(domain_info["degree"]),
            "university": random.choice(UNIVERSITIES),
            "year": random.randint(2010, 2021),
        }],
        "certifications": [safe_text(c) for c in certs],
        "total_years": total_years,
        "domain": domain_name,
    }


def create_resume_pdf(resume: dict, file_path: str) -> None:
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # --- Header ---
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, resume['name'], ln=True, align='C')
    pdf.set_font("Arial", '', 9)
    pdf.cell(0, 5, f"{resume['email']}  |  {resume['phone']}  |  {resume['location']}", ln=True, align='C')
    pdf.ln(3)
    pdf.set_draw_color(120, 120, 120)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # --- Summary ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Professional Summary", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, resume['summary'])
    pdf.ln(3)

    # --- Experience ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Work Experience", ln=True)
    for exp in resume['experience']:
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 5, f"{exp['title']}  -  {exp['company']}", ln=True)
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 5, exp['period'], ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 5, exp['description'])
        pdf.ln(2)
    pdf.ln(1)

    # --- Skills ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Technical Skills", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.multi_cell(0, 5, "  |  ".join(resume['skills']))
    pdf.ln(3)

    # --- Education ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 7, "Education", ln=True)
    pdf.set_font("Arial", '', 10)
    for edu in resume['education']:
        pdf.cell(0, 5, f"{edu['degree']}  -  {edu['university']}  ({edu['year']})", ln=True)
    pdf.ln(3)

    # --- Certifications ---
    if resume.get('certifications'):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 7, "Certifications", ln=True)
        pdf.set_font("Arial", '', 10)
        for cert in resume['certifications']:
            pdf.cell(0, 5, f"  - {cert}", ln=True)

    pdf.output(file_path)


def create_jd_txt(jd_template: dict, domain_name: str, file_path: str) -> None:
    company = fake.company()
    location = f"{fake.city()}, {fake.state_abbr()}"
    lines = [
        f"Job Title: {jd_template['title']}",
        f"Company: {company}",
        f"Location: {location}",
        "",
        "About the Role:",
        jd_template['about'],
        "",
        "Key Responsibilities:",
        *[f"  - {r}" for r in jd_template['responsibilities']],
        "",
        "Required Qualifications:",
        *[f"  - {q}" for q in jd_template['required_skills']],
        "",
        "Nice to Have:",
        *[f"  - {n}" for n in jd_template['nice_to_have']],
    ]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    os.makedirs(RESUMES_DIR, exist_ok=True)
    os.makedirs(JDS_DIR, exist_ok=True)

    domain_names = list(DOMAINS.keys())  # 5 domains

    # --- Generate 50 resumes: exactly 10 per domain ---
    print("Generating 50 resumes...")
    resume_count = 0
    for domain_name in domain_names:
        domain_info = DOMAINS[domain_name]
        slug = domain_name.lower().replace(" ", "_")
        for j in range(1, 11):
            resume = build_resume_data(domain_name, domain_info)
            file_path = os.path.join(RESUMES_DIR, f"resume_{slug}_{j:02d}.pdf")
            create_resume_pdf(resume, file_path)
            resume_count += 1
            print(f"  [{resume_count}/50] {file_path.split(os.sep)[-1]}")

    # --- Generate 5 JDs: one per domain ---
    print("\nGenerating 5 job descriptions...")
    for i, domain_name in enumerate(domain_names, 1):
        slug = domain_name.lower().replace(" ", "_")
        jd_template = JD_TEMPLATES[domain_name]
        file_path = os.path.join(JDS_DIR, f"jd_{slug}.txt")
        create_jd_txt(jd_template, domain_name, file_path)
        print(f"  [{i}/5] {file_path.split(os.sep)[-1]}")

    print(f"\nDone! Files saved to:")
    print(f"  Resumes : {RESUMES_DIR}")
    print(f"  JDs     : {JDS_DIR}")


if __name__ == "__main__":
    main()
