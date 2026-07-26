"""Autoresearch candidate — category: Covenant Not to Sue."""

CATEGORY_SYSTEM_PROMPTS = {
    "Covenant Not to Sue": """Category: Covenant Not to Sue
Category description: Is a party restricted from contesting the validity of the counterparty’s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?
Instructions:
- Identify all clauses that constitute a covenant not to sue: any contractual provision under which a party promises not to bring a lawsuit, demand, or challenge, or where bringing such a challenge is made a ground for termination, default, or penalty.
- Specifically look for language like ‘shall not issue any challenge’, ‘undertakes not to assert any claim’, ‘covenant not to sue’, ‘shall not contest’, and similar phrases, even if the clause relates to the contract itself (e.g., prohibiting challenges to termination).
- Also include provisions in termination, intellectual property, and default sections that make challenging the validity or enforceability of IP a ground for termination or breach.
- Before outputting a candidate, verify whether the restriction is limited solely to claims that are directly related to the contract’s performance (e.g., those explicitly limited to ‘claims in connection with the manufacture, advertising, sale, or distribution’ of goods, or to ‘breach of patent rights in the Products’ where those products are the subject of the agreement). If the clause only restricts claims that are inherently connected to the contract’s subject matter and does not also restrict challenges to IP ownership, then do NOT extract it; instead, treat it as not qualifying under this category.
- However, if a clause unconditionally forbids a challenge or claim, even if the subject (like termination) is related to the contract, extract it.
- Output each distinct qualifying span on a new line, preserving internal line breaks. If a covenant is part of an enumerated list with an introductory sentence, output the full list item and the preceding introductory sentence.
- If no clause meets these criteria, output NO_ANSWER and set marked_impossible to true.
- Do NOT output ‘Yes’ or ‘No’.""",
}
