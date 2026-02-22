# import requests


# API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/d1aae4e11203568e67b580609a23f625/ai/v1"
# headers = {"Authorization": "Bearer i94gxxjqVtdTYOAMwAzO0i08AVo1WEOI-wV1ttCm"}


# def run(model, inputs):
#     input = { "messages": inputs }
#     response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
#     return response.json()


# inputs = [
#     { "role": "system", "content": "You are a friendly assistan that helps write stories" },
#     { "role": "user", "content": "Write a short story about a llama that goes on a journey to find an orange cloud "}
# ];
# output = run("@cf/zai-org/glm-4.7-flash", inputs)
# print(output)

from openai import OpenAI

client = OpenAI(
    api_key='i94gxxjqVtdTYOAMwAzO0i08AVo1WEOI-wV1ttCm',
    base_url="https://api.cloudflare.com/client/v4/accounts/d1aae4e11203568e67b580609a23f625/ai/v1"
)

response = client.chat.completions.create(
    model="@cf/zai-org/glm-4.7-flash",
    messages=[
        { "role": "system", "content": "You are a friendly assistan that helps write stories" },
        { "role": "user", "content": "Write a short story about a llama that goes on a journey to find an orange cloud "}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='', flush=True)