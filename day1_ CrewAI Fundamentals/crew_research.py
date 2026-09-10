from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv()


# ============================================================
# AGENT 1 - RESEARCHER
# ============================================================

researcher = Agent(
    role="AI Researcher",
    goal="Research the given topic and identify accurate, useful information.",
    backstory=(
        "You are an experienced technology researcher. "
        "You collect important facts, identify key concepts, "
        "and organize information clearly for another writer."
    ),
    verbose=True,
    allow_delegation=False
)


# ============================================================
# AGENT 2 - WRITER
# ============================================================

writer = Agent(
    role="Technical Writer",
    goal="Convert research findings into a clear and structured report.",
    backstory=(
        "You are a skilled technical writer who explains complex "
        "technology topics in simple and understandable language."
    ),
    verbose=True,
    allow_delegation=False
)


# ============================================================
# AGENT 3 - REVIEWER
# ============================================================

reviewer = Agent(
    role="Research Reviewer",
    goal="Review the report for accuracy, clarity, completeness, and quality.",
    backstory=(
        "You are a strict technical reviewer. You check reports "
        "for missing information, unclear explanations, unsupported "
        "claims, and logical problems."
    ),
    verbose=True,
    allow_delegation=False
)


# ============================================================
# TASK 1 - RESEARCH
# ============================================================

research_task = Task(
    description=(
        "Research the topic: Generative AI and its applications in software development. "
        "Identify the main concepts, important applications, benefits, limitations, "
        "and real-world use cases. Organize the findings clearly."
    ),
    expected_output=(
        "A structured research summary containing key concepts, "
        "applications, benefits, limitations, and use cases."
    ),
    agent=researcher
)


# ============================================================
# TASK 2 - WRITING
# ============================================================

writing_task = Task(
    description=(
        "Using the research produced by the Researcher, write a clear "
        "technical report about Generative AI and its applications "
        "in software development. Include an introduction, applications, "
        "benefits, limitations, and conclusion."
    ),
    expected_output=(
        "A well-structured technical report written in clear English "
        "with headings and paragraphs."
    ),
    agent=writer
)


# ============================================================
# TASK 3 - REVIEW
# ============================================================

review_task = Task(
    description=(
        "Review the report created by the Writer. Check whether it is "
        "accurate, complete, clear, logically organized, and useful. "
        "Identify weaknesses and suggest improvements."
    ),
    expected_output=(
        "A review containing strengths, weaknesses, missing information, "
        "and specific improvement suggestions."
    ),
    agent=reviewer
)


# ============================================================
# CREW
# ============================================================

crew = Crew(
    agents=[
        researcher,
        writer,
        reviewer
    ],
    tasks=[
        research_task,
        writing_task,
        review_task
    ],
    process=Process.sequential,
    verbose=True
)


# ============================================================
# RUN CREW
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("   CREWAI MULTI-AGENT RESEARCH CREW")
    print("========================================\n")

    result = crew.kickoff()

    print("\n========================================")
    print("              FINAL OUTPUT")
    print("========================================\n")

    print(result)