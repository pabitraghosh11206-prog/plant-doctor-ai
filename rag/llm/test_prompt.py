from prompt import PROMPT_TEMPLATE

context = """
Early blight causes brown spots on tomato leaves.
Good prevention includes reducing leaf wetness.
"""

question = "How can I prevent Early Blight?"

final_prompt = PROMPT_TEMPLATE.format(
    context=context,
    question=question
)

print(final_prompt)