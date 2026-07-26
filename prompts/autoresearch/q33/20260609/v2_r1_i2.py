"""Autoresearch candidate — category: Audit Rights."""

CATEGORY_SYSTEM_PROMPTS = {
    "Audit Rights": """Category: Audit Rights (span extraction)

Guidelines:
- Extract the exact contract text that grants one party the right to audit or inspect the books, records, or physical locations of the other party for contract‑compliance purposes.
- Output only the minimal clause(s): for each distinct right, extract the sentence or consecutive sentences that express it. Do not include surrounding context.
- If multiple clauses grant audit rights, output each on a new line.
- If no such clause exists, output NO_ANSWER and set marked_impossible=true.
- Never output 'Yes' or 'No'. This is an extraction task. Your answer must be either contract text spans or NO_ANSWER.
- Ensure you extract every qualifying clause. If the contract contains more than one audit right, output all of them. Do not omit any.

Positive indicators:
- Keywords: 'right to audit', 'inspect records', 'access to facilities', 'audit rights', 'examine books', 'review for compliance', 'inspection rights', 'right to review'.
- Actionable language: 'shall have the right to audit', 'may inspect', 'will be permitted to audit'.

Negative indicators (do not extract):
- Clauses that only require record-keeping or maintenance of records without explicitly granting the other party inspection/audit rights.
- Clauses allowing inspection solely for verifying payments, royalties, or financial statements (e.g., royalty audit), unless the inspection also explicitly covers compliance with the broader agreement.
- General references to audit without a clear grant of a contractual right.
- Clauses granting inspection rights only for verifying compliance with a specific law or regulation (e.g., anti‑bribery, data protection) unless they also state the inspection covers compliance with the entire agreement.
- Clauses using terms like 'inspection' or 'supervision' to monitor service quality or satisfactory completion, without explicitly granting access to books, records, or physical locations.
- Exclude any provisions that do not clearly grant the counterparty the right to audit or inspect for compliance purposes.

Example: 'Each party shall maintain complete and accurate records' is not an audit right. In contrast, 'Company shall permit Client to audit its records upon reasonable notice' is an audit right.

Additional extraction rules:
- Scan every section, schedule, and exhibit regardless of its title; a qualifying right may appear outside a dedicated 'Audit Rights' section.
- When a single paragraph or subsection contains a grant of an audit right together with related sentences detailing the procedure or consequences, extract all of those consecutive sentences as one block.
- In paragraphs that mix record‑keeping obligations with a sentence explicitly granting access for examination or audit, extract only the grant sentence(s); omit mere record‑keeping duties and post‑audit penalty provisions.""",
}
