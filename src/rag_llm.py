import ollama
from disease_info import disease_info

def retrieve_disease_info(disease_name):
    if disease_name in disease_info:
        return disease_info[disease_name]

    normalized_name = disease_name.lower().replace("_", " ").replace("___", " - ")

    for key, value in disease_info.items():
        normalized_key = key.lower().replace("_", " ").replace("___", " - ")

        if normalized_name in normalized_key or normalized_key in normalized_name:
            return value

    return None

def generate_disease_explanation(disease_name):
    info = retrieve_disease_info(disease_name)

    if info:
        context = f"""
Disease: {disease_name}

Description:
{info.get("description", "")}

Treatment:
{info.get("treatment", "")}

Prevention:
{info.get("prevention", "")}
"""
    else:
        context = f"""
Disease: {disease_name}

No additional information was found in the project knowledge base.
"""

    prompt = f"""
You are a plant disease assistant.

Use the following retrieved knowledge base information to explain the disease.

{context}

Give a simple and clear explanation.

Include:
1. What the disease is
2. Symptoms
3. Treatment
4. Prevention

Do not invent information that is not supported by the retrieved knowledge.
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Unable to generate AI explanation: {str(e)}"