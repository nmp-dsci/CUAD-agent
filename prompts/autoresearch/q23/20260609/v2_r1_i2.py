"""Autoresearch candidate — category: IP Ownership Assignment."""

CATEGORY_SYSTEM_PROMPTS = {
    "IP Ownership Assignment": """Category: IP Ownership Assignment
Category description: Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?

Instructions:
- Extract the exact contract clause(s) that show any intellectual property (IP) owned or created by one party becomes the property of the counterparty. This includes assignments of pre-existing IP as well as IP created during the contract term.
- Look for language such as "assigns", "transfers", "shall own", "will be the property of", "wholly owned by", "hereby assigns", "all right, title and interest", "agrees to assign", etc. Additionally, include clauses stating that IP rights "automatically revert", "shall vest in", or "shall be assigned back to" a party upon termination or another specified event, as these constitute an assignment of ownership.
- Do not extract clauses that only grant a license or usage right without transfer of ownership.
- Do not extract clauses that merely state that a party retains ownership of its own pre-existing IP, or that modifications made to its IP remain its property, unless they explicitly assign newly created IP by the other party to the first party. For example, a clause stating "Any intellectual property rights created by changes made by Party A to Party B's materials are the sole property of Party B" is not an assignment of IP from Party A to Party B. Similarly, a clause that "improvements to Party A's Baseline IP are solely the property of Party A" is not an assignment unless it also assigns IP created by Party B to Party A.
- Do not extract clauses that assign only goodwill, reputation, or similar non-IP intangible rights, even if they use phrases like "hereby assigns all goodwill and all other rights developed in connection with trademark use." Only extract such clauses if they also explicitly assign intellectual property rights (e.g., "all right, title and interest in the patents, copyrights, trademarks").
- Do not extract clauses that transfer only regulatory registrations, permits, or administrative filings (e.g., import licenses, health registrations) unless those filings are explicitly defined as Intellectual Property in the contract. If a clause assigns "all right, title and interest" in regulatory approvals together with associated patents, data, or other IP rights, include the entire clause.
- Search the entire contract, including all sections, exhibits, schedules, and attachments. Pay special attention to sections like "Grant of License", "Regulatory", "Termination", "Information", "Assignment", "Ownership", definitions, and any exhibit titled "Assignment" or "Form of Assignment".
- If a contract contains symmetrical ownership provisions where each party's IP in different fields is assigned to the counterparty, read the entire section and extract every separate assignment statement, even if they appear as consecutive subclauses. Output both clauses separately.
- If there are multiple clauses that independently show IP transfer, output each on a new line, even if they appear in the same paragraph. Do not merge them.
- If no such clause exists, output exactly "NO_ANSWER" and set marked_impossible to true.
- Never output "Yes" or "No" as the answer; always provide the clause text or "NO_ANSWER".""",
}
