from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
import anthropic
from src.main import answer_question
from src.retrieval import get_chroma_client
import time


load_dotenv()

chroma_client = get_chroma_client()
anthropic_client = anthropic.Anthropic()

file_path = "data/test_docs/Monarch_Butterfly_Migration.pdf"
file_bytes = Path(file_path).read_bytes()
filename = Path(file_path).name
time.perf_counter()
start = time.perf_counter()
answer = answer_question(file_bytes, filename, "what is the life cycle of Monarch Butterfly ? ", chroma_client, anthropic_client)
end = time.perf_counter()

print(answer)
print(f"\n--- Took {end - start:.2f} seconds ---")