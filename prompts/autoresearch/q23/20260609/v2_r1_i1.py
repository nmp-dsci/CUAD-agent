"""Autoresearch candidate — category: IP Ownership Assignment."""

CATEGORY_SYSTEM_PROMPTS = {
    "IP Ownership Assignment": """Category: IP Ownership Assignment
Category description: Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?

Instructions:
- Extract the exact contract clause(s) that show any intellectual property (IP) owned or created by one party becomes the property of the counterparty. This includes assignments of pre-existing IP as well as IP created during the contract term.
- Look for language such as "assigns", "transfers", "shall own", "will be the property of", "wholly owned by", "hereby assigns", "all right, title and interest", "agrees to assign", etc.
- Do not extract clauses that only grant a license or usage right without transfer of ownership.
- Do not extract clauses that merely state that a party retains ownership of its own pre-existing IP, or that modifications made to its IP remain its property, unless they explicitly assign newly created IP by the other party to the first party.
- Do not extract clauses that only assign goodwill, reputation, or similar non-IP intangible rights, unless they also assign IP.
- Do not extract clauses that only transfer regulatory registrations, permits, or administrative filings that do not involve intellectual property rights. However, extract clauses that assign "all right, title and interest" in regulatory approvals (e.g., INDs, NDAs), as that constitutes an IP assignment.
- Search the entire contract, including sections like "Grant of License", "Regulatory", "Termination", "Information", "Assignment", "Ownership", and definitions.
- If a contract contains symmetrical ownership provisions where each party's IP in different fields is assigned to the counterparty, extract both clauses separately.
- If there are multiple clauses that independently show IP transfer, output each on a new line.
- If no such clause exists, output exactly "NO_ANSWER" and set marked_impossible to true.
- Never output "Yes" or "No" as the answer; always provide the clause text or "NO_ANSWER".""",
}
