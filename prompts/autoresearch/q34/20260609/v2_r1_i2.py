"""Autoresearch candidate — category: Uncapped Liability."""

CATEGORY_SYSTEM_PROMPTS = {
    "Uncapped Liability": """Category: Uncapped Liability

Task: Extract only the exact text span(s) from the contract that explicitly state that a party's liability for breach of obligation (or for specific breaches such as IP infringement or breach of confidentiality) is uncapped. This means the text must clearly remove or negate any cap on liability, whether the cap is a monetary limit or a restriction on types of damages (e.g., exclusion of consequential or indirect damages). A cap is removed when the contract states that the limitation does not apply to certain breaches, or that liability for those breaches is not subject to the limitation, even if the word "unlimited" is not used.

Guidelines:
- Look for explicit statements like "liability shall be unlimited", "without limitation", "the cap does not apply to...", "nothing in this section shall limit liability for...", "notwithstanding the foregoing, the limitation of liability shall not apply to...", "the limitations set forth in [section] shall not apply in respect of...", or similar direct statements.
- Also look for exception phrases at the start of a limitation clause that carve out certain breaches or obligations from the limitation, such as "Except for [list of breaches], ...", "EXCLUDING LIABILITY FOR [claims] ...", "except with respect to [obligations] ...". In such cases, the exception effectively uncaps liability for the listed items. Extract the entire exception phrase or the full sentence that contains the carve-out, so the context is clear.
- When a limitation clause includes a sentence that says "The foregoing limitations shall not apply to..." or "Notwithstanding the above, the limitations shall not apply to [breaches]", extract that sentence as uncapped liability for those breaches.
- Do not extract a limitation clause that merely excludes certain types of damages (e.g., "In no event shall either party be liable for indirect, incidental, or consequential damages") unless it contains an express exception for specific breaches as described above. The presence of an exception within such a clause is sufficient to uncap liability for the excepted breaches, even if the clause does not impose a monetary cap.
- Indemnification provisions are generally not extracted unless they explicitly state that the indemnification obligation is uncapped or are referenced directly in a carve-out from a limitation of liability clause. If an indemnification provision is listed in such a carve-out, extract the carve-out language.
- If multiple separate spans satisfy this condition, output each on a new line.
- If no such explicit uncapping language exists, output NO_ANSWER and set marked_impossible=true.

Remember: The output must be the exact contract text, never 'Yes' or 'No'.""",
}
