from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()

# Accept multiple common names and normalize
hf_token = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

if not hf_token:
    raise EnvironmentError(
        "Hugging Face token not found. Set HUGGINGFACE_HUB_TOKEN (preferred) or another recognized var."
    )


os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token  # some integrations read this

# Optional: persist credentials for huggingface_hub cache/downloads
try:
    login(token=hf_token, add_to_git_credential=False)
except Exception:
    pass  # non-interactive envs may skip

# Create the LangChain LLM (pass token to model load)
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"token": hf_token},  # used by Transformers >=4.35
    pipeline_kwargs={"max_new_tokens": 500, "temperature": 0.5},
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("Write a poem about a lonely computer.")
print(result.content)
