from huggingface_hub import InferenceClient
from my_config import HF_API_KEY

client = InferenceClient(api_key=HF_API_KEY)

user_question = input("Введите запрос для ИИ: ")

try:
    first_prompt = f"""
Ответь на запрос пользователя кратко, понятно и по существу.

Запрос пользователя:
{user_question}
"""

    first_response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
        messages=[
            {
                "role": "user",
                "content": first_prompt
            }
        ],
        max_tokens=500
    )

    first_answer = first_response.choices[0].message.content

    improve_prompt = f"""
Улучши ответ другой модели.

Сделай ответ:
- более ясным
- более полным
- более структурированным
- без фактических ошибок
- без лишней воды

Запрос пользователя:
{user_question}

Первый ответ:
{first_answer}
"""

    second_response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
        messages=[
            {
                "role": "user",
                "content": improve_prompt
            }
        ],
        max_tokens=700
    )

    improved_answer = second_response.choices[0].message.content

    print("\n--- ИСХОДНЫЙ ЗАПРОС ---")
    print(user_question)

    print("\n--- ПЕРВЫЙ ПРОМПТ ---")
    print(first_prompt)

    print("\n--- ПЕРВЫЙ ОТВЕТ ---")
    print(first_answer)

    print("\n--- ПРОМПТ ДЛЯ УЛУЧШЕНИЯ ---")
    print(improve_prompt)

    print("\n--- УЛУЧШЕННЫЙ ОТВЕТ ---")
    print(improved_answer)

except Exception as e:
    print("\nПроизошла ошибка:")
    print(e)
