# student_schema_prompt.py
# pip install langchain-google-genai python-dotenv

import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1) Keep schema separate (JSON Schema format with multiple types)
json_schema = {
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "age": { "type": "integer", "minimum": 0, "maximum": 120 },
    "roll_no": { "type": "string" },
    "active": { "type": "boolean" },
    "gpa": { "type": "number", "minimum": 0, "maximum": 10 },
    "attendance": { "type": "number", "minimum": 0, "maximum": 100 },
    "subjects": {
      "type": "object",
      "properties": {
        "maths": { "type": "string", "enum": ["Ms Sonakshi"] },
        "eng":   { "type": "string", "enum": ["Mr Ramesh"] }
      },
      "required": ["maths", "eng"],
      "additionalProperties": False
    },
    "address": {
      "type": "object",
      "properties": {
        "city":  { "type": "string" },
        "state": { "type": "string" }
      },
      "required": ["city", "state"],
      "additionalProperties": True
    },
    "clubs": { "type": "array", "items": { "type": "string" } },
    "enrolled_on": { "type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" }
  },
  "required": ["name","age","subjects","roll_no","active","gpa","attendance","enrolled_on"],
  "additionalProperties": False
}

# 2) Ask user to describe the student (these details are fed to the LLM)
user_description = input("Describe the student (e.g., name, age, city, interests): ").strip()

# 3) Build a minimal, strict prompt
schema_str = json.dumps(json_schema, indent=2)
prompt = f"""
Return exactly ONE JSON object that strictly conforms to the JSON Schema below.
Rules:
- Output JSON only (no markdown, no backticks, no comments).
- Use double quotes for all keys and string values.
- If the user description provides a value, use it; otherwise infer a plausible value.
- Keep teachers fixed as per schema.

User description:
{user_description}

JSON Schema:
{schema_str}
"""

# 4) Call Gemini and print the JSON
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
msg = llm.invoke(prompt)
text = (msg.content or "").strip()

# Try to pretty-print if valid JSON, or print raw on failure
try:
    obj = json.loads(text)
    print(json.dumps(obj, indent=2, ensure_ascii=False))
except Exception:
    print(text)
