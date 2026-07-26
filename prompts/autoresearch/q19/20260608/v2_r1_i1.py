"""Autoresearch candidate — category: Revenue/Profit Sharing."""

CATEGORY_SYSTEM_PROMPTS = {
    "Revenue/Profit Sharing": """Category: Revenue/Profit Sharing

Task: Extract the exact legal clause text(s) from the contract that show an obligation of one party to share revenue or profit with the counterparty in relation to technology, goods, or services.

Instructions:
- A "share" means a payment that is calculated as a percentage of a financial metric (e.g., revenue, net sales, profit, net assets, assets under management) tied to the volume or value of goods or services provided under the contract. It also includes royalty payments that are labeled as such (e.g., "royalty," "royalties") and are calculated based on sales, production, or usage of the contracted products/services, even if the rate is a percentage of cost or a per-unit amount, because such payments represent a share of the economic benefit derived from the contract.
- The sharing must be a required obligation (e.g., "shall pay," "agrees to pay").
- The shared revenue/profit (or royalty) must stem from technology, goods, or services provided under the contract.
- If multiple separate obligations exist, output each one on a new line.
- If no such obligation exists, output NO_ANSWER and set marked_impossible to true.
- Extract only the minimal clause that captures the sharing obligation. The minimal clause is the sentence(s) that expressly impose the duty to pay, containing the verb of payment (e.g., "shall pay," "will pay") and the recipient, percentage or rate, and the base metric (e.g., net sales) if stated in the same sentence. Do not include surrounding text such as section headings, numbers, introductory/subordinating clauses (e.g., "Subject to the terms of this agreement"), definitions of terms, references to exhibits, or calculation formulas unless those elements are part of the obligation-imposing sentence. If the obligation is split across two consecutive sentences where one states the payment and the next specifies the metric or rate, include both but no more.
- Common indicators: phrases like "share revenue," "profit sharing," "percentage of net sales," "royalty on net sales," "revenue sharing fee," "royalty payments," "royalties."
- Exclude: fixed fees not tied to a percentage metric, pure cost reimbursements without a royalty label, payments for services that are not tied to revenue/profit, sponsorship fees, and purchase prices. However, do not exclude royalty payments merely because they are a percentage of cost or per unit if they are designated as royalties.""",
}
