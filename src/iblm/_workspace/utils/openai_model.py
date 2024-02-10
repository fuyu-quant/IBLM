from openai import OpenAI

client = OpenAI()

def _openai_model(model_name, prompt, seed):
    response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            seed=seed
            )

    return response.choices[0].message.content
