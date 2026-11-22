from src.states.case_state import CaseState


class DefenseNode:
    """
    Defense Lawyer Agent
    Constructs a strong argument defending the accused,
    responding directly to the prosecution argument and
    using only case facts retrieved via RAG.
    """

    def __init__(self, llm):
        self.llm = llm

    def defense_argument(self, state: CaseState):
        ctx = state.get("rag_context", "")[:3000]
        prosecution_argument = state.get("prosecution_argument", "")[:1500]

        prompt = f"""
You are the DEFENSE LAWYER in an Indian courtroom.

Your duties:
- Argue that the accused is NOT guilty.
- Respond directly to the prosecution's claims.
- Use ONLY the facts present in the authenticated case documents.
- Point out inconsistencies, missing evidence, or lack of proof.
- Build a strong, structured legal defense.

Case Facts:
{ctx}

Prosecution Argument:
{prosecution_argument}

Now write the defense argument.
"""

        try:
            response = self.llm.invoke(prompt)
            return {
                "defense_argument": response.content
            }
        except Exception as exc:
            return {
                "defense_argument":
                f"Unable to generate defense argument: {exc}"
            }
