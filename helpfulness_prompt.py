HELPFULNESS_RATER_TEMPLATE = """
## Primary Goal
You are an expert evaluator judging the "helpfulness" of an AI assistant's intermediate responses. The assistant is designed to think and respond in interleaved steps (<think><answer><think><answer>). Your task is to determine if a given intermediate answer provides tangible value to the user, or if the assistant should have continued thinking before providing that response.
## Core Principle
An intermediate response is HELPFUL if it provides a self-contained, usable piece of information that the user can understand and act on without waiting for a future revision. It should feel like a deliberate, incremental delivery of the final answer.
An intermediate response is NOT HELPFUL if it's a "work-in-progress" artifact, a trivial fragment, or a low-quality draft that the AI intends to replace or substantially revise later. It should not feel like the user is watching the AI "type and delete."
## Inputs
### User Query:
{question}
### Previous Turns:
{context}
### AI Intermediate Response:
{predicted_answer}
## Evaluation Criteria
Carefully evaluate the AI Intermediate Response based on the criteria below.
### A response is HELPFUL (Decision: TRUE) if it:
Delivers a Complete Sub-Part: Provides a finished, polished part of the final answer.
Example: For a request of "10 suggestions," responding with "Here are the first 3 fully-formed suggestions: [1], [2], [3]." is HELPFUL.
Asks a Necessary Clarifying Question: Poses a question to the user that is essential for refining the final answer.
Example: "To give you better backpacking ideas, could you tell me your budget and experience level?" is HELPFUL.
States a Concrete, Informative Action: Announces a specific action (like a tool use or search query) that informs the user about the process and manages expectations.
Example: "I'm now searching for pet-friendly trails within a 2-hour drive of NYC that are accessible by train." is HELPFUL.
Summarizes Findings and Presents a Choice: Synthesizes information gathered so far and offers the user a decision point.
Example: "I found 5 options, but none are open in the winter. Should I look for summer options instead, or give you the 5 I found?" is HELPFUL.
### A response is NOT HELPFUL (Decision: FALSE) if it:
Is a Trivial Fragment: Consists of a single word, an incomplete sentence, or conversational filler that adds no new information.
Example: "Okay, I can help with that." is NOT HELPFUL.
Example: "Here are some" (and then stops) is NOT HELPFUL.
Is "Drafting in Public": Presents incomplete fragments from multiple different parts of the final answer, instead of finishing one part completely. The response feels like a low-quality skeleton or a rough draft that the user cannot yet use.
Example: For a request to write a three-paragraph email, responding with one rough sentence for each of the three paragraphs is NOT HELPFUL. The model should have finished the first paragraph completely.
Is a Vague or Redundant Statement of Intent: Repeats the user's request or states a vague plan without providing new information or a concrete action.
Example: "I will now give you 10 suggestions for a backpacking trip." is NOT HELPFUL.
Example: "Thinking about it..." is NOT HELPFUL.
Merely Corrects a Previous Minor Error: Fixes a typo or grammar mistake from its own previous turn without adding substantive new information. Corrections should be integrated into the next helpful chunk.
## Final Check
Ask yourself: "Could the user gain real value from this response right now, or is it just noise they have to wait through?" If it delivers real, incremental progress, it's TRUE. Otherwise, it's FALSE.
## Output Format
Respond with exactly one line in this format:
Decision: TRUE
or
Decision: FALSE
Please proceed with the evaluation.
Decision: """