from langchain_core.documents import Document
from src.states.case_state import CaseState


class ProsecutionNode:
    """Creates a prosecution-side analysis for the case simulation."""

    def __init__(self, llm):
        self.llm = llm

    def prosecutor_argument(self, state: CaseState):
        ctx = state.get("rag_context", "").strip()

        if not ctx:
            return {
                "prosecution_argument": (
                    "No case facts were retrieved, so a grounded prosecution "
                    "argument cannot be prepared."
                )
            }

        prompt = f"""
Prepare an educational, prosecution-side analysis for a courtroom simulation.
This is not legal advice or a real court submission.

Use only the case facts below. Explain the strongest reasonable points
supporting the prosecution's position. Do not invent facts, treat allegations
as proven facts, or claim the accused is guilty. Identify uncertainties or
missing evidence where relevant.

Case facts:
{ctx}

Write a clear, structured prosecution-side analysis.
"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content

            if isinstance(content, str):
                argument = content.strip()
            else:
                argument = str(content).strip()

            if not argument:
                argument = "The model returned an empty prosecution analysis."

            return {"prosecution_argument": argument}

        except Exception as exc:
            return {
                "prosecution_argument": (
                    f"Unable to generate prosecution analysis: {exc}"
                )
            }