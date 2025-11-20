import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import time

LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda"

DOCUMENTS = [
    "The Nsight Systems tool is used for timeline analysis and CPU-GPU interactions.",
    "Nsight Compute is a kernel-level profiler for optimizing specific CUDA instructions.",
    "CUDA Streams allow concurrent execution of kernels on the same GPU.",
    "Warp divergence happens when threads in a warp follow different execution paths.",
    "Shared memory is a small, user-managed cache on the Streaming Multiprocessor (SM).",
    "Global memory is large but has high latency compared to L1/L2 caches.",
    "Tensor Cores are specialized hardware units for mixed-precision matrix multiplication.",
    "Occupancy refers to the ratio of active warps to the maximum number of warps supported.",
    "NVTX (NVIDIA Tools Extension) allows developers to annotate their timeline.",
    "Memory coalescing ensures that threads access global memory in contiguous chunks."
]

class RAGAgent:
    def __init__(self):
        print(f"Loading Embedding Model ({EMBED_MODEL_ID})...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
        
        print(f"Loading LLM ({LLM_MODEL_ID})...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        print("Indexing Knowledge Base...")
        with torch.no_grad():
            self.doc_embeddings = self.embed_model.encode(
                DOCUMENTS, 
                convert_to_tensor=True, 
                device=DEVICE
            )
            
    def retrieve(self, query):
        nvtx.range_push("RAG: Retrieval (BERT + MatMul)")
        nvtx.range_push("Embed Query")
        query_vec = self.embed_model.encode(query, convert_to_tensor=True, device=DEVICE)
        nvtx.range_pop()
        nvtx.range_push("Vector Search")
        scores = torch.matmul(query_vec, self.doc_embeddings.T)
        top_k_indices = torch.topk(scores, k=2).indices
        nvtx.range_pop()
        retrieved_docs = [DOCUMENTS[idx] for idx in top_k_indices]
        torch.cuda.synchronize() 
        nvtx.range_pop()
        return retrieved_docs

    def generate(self, query, context_docs):

        nvtx.range_push("RAG: Generation (Qwen3)")

        context_str = "\n".join([f"- {d}" for d in context_docs])
        messages = [
            {"role": "system", "content": "You are a helpful CUDA expert."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = self.llm.generate(
                model_inputs.input_ids,
                max_new_tokens=30, # Keep it short for cleaner profiling
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        torch.cuda.synchronize()
        nvtx.range_pop()
        return response

    def run(self, user_query):
        nvtx.range_push(f"Query: {user_query}")
        
        print(f"\nUser: {user_query}")
        
        docs = self.retrieve(user_query)
        print(f"Retrieved: {docs}")

        answer = self.generate(user_query, docs)
        print(f"Agent: {answer}")
        
        nvtx.range_pop()

if __name__ == "__main__":
    agent = RAGAgent()
    
    print("\n--- Warmup ---")
    agent.run("What is Nsight Systems?")

    print("\n--- STARTING PROFILE CAPTURE ---")
    torch.cuda.synchronize()
    
    nvtx.range_push("Profile Session")
    agent.run("How do I fix warp divergence?")
    nvtx.range_pop()
    
    print("\nDone.")