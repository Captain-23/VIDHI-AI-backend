from src.states.case_state import CaseState


class DefenseNode:
    """Creates a defense-side analysis for the case simulation."""

    def __init__(self, llm):
        self.llm = llm

    def defense_argument(self, state: CaseState):
        ctx = state.get("rag_context", "").strip()
        prosecution_argument = state.get("prosecution_argument", "").strip()

        if not ctx:
            return {
                "defense_argument": (
                    "No case facts were retrieved, so a grounded defense "
                    "analysis cannot be prepared."
                )
            }

        prompt = f"""
Prepare an educational, defense-side analysis for a courtroom simulation.
This is not legal advice or a real court submission.

Use only the case facts below. Respond to the prosecution's points by
identifying reasonable weaknesses, uncertainties, alternative
interpretations, and gaps in the evidence. Do not invent facts or claim
that the accused is innocent. If the supplied facts do not support a
defense point, say so clearly.

Case facts:
{ctx}

Prosecution argument:
{prosecution_argument or "No prosecution argument was provided."}

Write a clear, structured defense-side analysis.
"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content

            if isinstance(content, str):
                argument = content.strip()
            else:
                argument = str(content).strip()

            if not argument:
                argument = "The model returned an empty defense analysis."

            return {"defense_argument": argument}

        except Exception as exc:
            return {
                "defense_argument": f"Unable to generate defense analysis: {exc}"
            }