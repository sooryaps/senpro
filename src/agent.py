import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from vector_store import query_chunks, lookup_by_id

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def search_documents(query: str) -> str:
    """Search QA documents semantically for relevant information."""
    results = query_chunks(query, top_k=3)
    return "\n\n".join([f"[Score: {score:.2f}] {chunk}" for chunk, score in results])

search_tool = {
    "name": "search_documents",
    "description": "Semantically search QA release notes, bug logs, and test cases for information relevant to a question.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"]
    }
}

lookup_tool = {
    "name": "lookup_by_id",
    "description": "Look up an exact bug or test case by its ID (e.g. 'BUG-447', 'TC-108') when the user references a specific ID directly.",
    "parameters": {
        "type": "object",
        "properties": {"item_id": {"type": "string", "description": "The exact ID, e.g. BUG-447"}},
        "required": ["item_id"]
    }
}

tools = types.Tool(function_declarations=[search_tool, lookup_tool])
config = types.GenerateContentConfig(tools=[tools])

SYSTEM_PROMPT = """You are Sentinel, a QA assistant with access to two tools: search_documents
for open-ended questions, and lookup_by_id when the user mentions a specific BUG- or TC- ID.
Choose ONE tool, call it ONCE, then immediately give your final answer using its results.
Do not call another tool unless the first result was completely empty or irrelevant.
Only answer using tool results — never guess. Cite the specific ID in your answer."""

AVAILABLE_FUNCTIONS = {
    "search_documents": search_documents,
    "lookup_by_id": lookup_by_id
}

def run_agent(user_message: str, max_turns: int = 3) -> str:
    """Run the agent in a loop, letting the model call tools until it gives a final text answer."""
    contents = [
        types.Content(role="user", parts=[types.Part(text=f"{SYSTEM_PROMPT}\n\nUser question: {user_message}")])
    ]

    for turn in range(max_turns):
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=contents,
            config=config
        )

        part = response.candidates[0].content.parts[0]

        if part.function_call:
            fn_name = part.function_call.name
            fn_args = dict(part.function_call.args)
            print(f"[Agent chose tool: {fn_name}({fn_args})]")

            result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)

            contents.append(response.candidates[0].content)
            contents.append(types.Content(
                role="user",
                parts=[types.Part(function_response=types.FunctionResponse(name=fn_name, response={"result": result}))]
            ))
        else:
            return part.text

    return "Agent reached max turns without a final answer."
    """Run the agent: let the model decide which tool to call, execute it, and get a final answer."""
    contents = [
        types.Content(role="user", parts=[types.Part(text=f"{SYSTEM_PROMPT}\n\nUser question: {user_message}")])
    ]

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=contents,
        config=config
    )

    part = response.candidates[0].content.parts[0]

    if part.function_call:
        fn_name = part.function_call.name
        fn_args = dict(part.function_call.args)
        print(f"[Agent chose tool: {fn_name}({fn_args})]")

        result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)

        contents.append(response.candidates[0].content)
        contents.append(types.Content(
            role="user",
            parts=[types.Part(function_response=types.FunctionResponse(name=fn_name, response={"result": result}))]
        ))

        final_response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=contents,
            config=config
        )
        return final_response.text

    return part.text
import time
if __name__ == "__main__":
    print(run_agent("why did the availability filter test fail?"))
    print("\n---\n")
    print(run_agent("what's the status of BUG-441?"))