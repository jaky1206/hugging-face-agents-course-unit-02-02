import os

from dotenv import load_dotenv

load_dotenv()

from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=100,
    api_key=os.getenv("OPENAI_API_KEY"),
)

print(llm.complete("Hello, how are you?"))
# I am good, how can I help you today?
