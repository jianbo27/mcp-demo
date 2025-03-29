import asyncio
import json
import os
import io
from typing import List, Dict, Any, Tuple, Optional
import aiohttp
import argparse
from tqdm import tqdm
import pandas as pd
import re
from datasets import load_dataset

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ========== CONFIG ==========
OLLAMA_MODEL = "llama3.1:8b"
MCP_SERVER_COMMAND = "npx"
MCP_SERVER_ARGS = ["-y", "@modelcontextprotocol/server-sequential-thinking"]

SYSTEM_PROMPT = """You are a helpful assistant capable of using external tools.

# Tools

{tools}

# Notes:
- Use tools when relevant.
- Respond naturally, blending tool results into conversation.
"""

# Standardized prompt format for GPQA as per the paper
GPQA_PROMPT_TEMPLATE = """What is the correct answer to this question: {question}

Choices:
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}

Format your response as follows: "The correct answer is (insert answer here)"."""

# ========== MCP CLIENT ==========
class MCPClient:
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session = None
        self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()

    async def get_available_tools(self) -> List[Any]:
        tools = await self.session.list_tools()
        return tools.tools

    def call_tool(self, tool_name: str):
        async def callable(*args, **kwargs):
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return response.content[0].text
        return callable

# ========== OLLAMA CLIENT ==========
class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    async def generate_response(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.host}/api/generate", json=payload) as resp:
                    if resp.status != 200:
                        raise Exception(await resp.text())
                    data = await resp.json()
                    return data.get("response", "[No response from model]").strip()
        except Exception as e:
            return f"[Ollama Error: {e}]"

# ========== GPQA DATASET ==========
class GPQADataset:
    def __init__(self, dataset_name: str = "prompt-leaderboard/gpqa-100"):
        """Load the GPQA dataset from Hugging Face"""
        self.dataset_name = dataset_name
        self.dataset = None
        self.source = None
    
    def load(self):
        """Load the dataset from Hugging Face"""
        try:
            print(f"Loading GPQA dataset from Hugging Face: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name, split="test")
            self.source = f"Hugging Face: {self.dataset_name}"
            print(f"Successfully loaded {len(self.dataset)} samples")
            
            # Get dataset features/column names
            features = self.dataset.features
            print(f"Dataset features: {list(features.keys())}")
            
            # Debug - print first item
            if len(self.dataset) > 0:
                print(f"First sample: {json.dumps(self.dataset[0], indent=2)}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        return self
        
    def get_samples(self, limit: Optional[int] = None) -> List[Dict]:
        """Return the dataset samples reformatted for evaluation, optionally limited to a specific count"""
        if self.dataset is None:
            self.load()
        
        samples = []
        for i, item in enumerate(self.dataset):
            try:
                # Extract the question
                question = item.get("Question", "")
                
                # Extract the four options - format: A is always the correct answer
                # In GPQA format they are stored as 'Correct Answer' and 'Incorrect Answer X'
                correct_answer = item.get("Correct Answer", "")
                incorrect_1 = item.get("Incorrect Answer 1", "")
                incorrect_2 = item.get("Incorrect Answer 2", "")
                incorrect_3 = item.get("Incorrect Answer 3", "")
                
                # Skip if any required field is missing
                if not question or not correct_answer or not incorrect_1 or not incorrect_2 or not incorrect_3:
                    missing_fields = []
                    if not question: missing_fields.append("Question")
                    if not correct_answer: missing_fields.append("Correct Answer") 
                    if not incorrect_1: missing_fields.append("Incorrect Answer 1")
                    if not incorrect_2: missing_fields.append("Incorrect Answer 2")
                    if not incorrect_3: missing_fields.append("Incorrect Answer 3")
                    print(f"Skipping item {i}: Missing required fields - {', '.join(missing_fields)}")
                    continue
                
                # Create a unique ID for the sample
                question_id = item.get("Record ID", i)
                
                # Shuffle options to create A, B, C, D format
                # For simplicity, we'll fix the correct answer as A and randomize in the future
                sample = {
                    "question_id": question_id,
                    "question": question,
                    "options": {
                        "A": correct_answer,
                        "B": incorrect_1,
                        "C": incorrect_2,
                        "D": incorrect_3
                    },
                    "correct_answer": "A"  # Always A in this setup
                }
                
                samples.append(sample)
                
                # Print first sample for debugging
                if i == 0:
                    print(f"First processed sample: {json.dumps(sample, indent=2)}")
                
                if limit and len(samples) >= limit:
                    break
                    
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                print(f"Item details: {json.dumps(item, indent=2)}")
                continue
        
        print(f"Successfully processed {len(samples)} valid samples")
        return samples

# ========== ANSWER EXTRACTION ==========
def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract the multiple choice answer (A, B, C, or D) from the model response"""
    # Look for the standardized format "The correct answer is X"
    pattern = r"(?:the correct answer is|correct answer is|answer is|the answer is)[^\w]*(A|B|C|D)"
    matches = re.findall(pattern, response, re.IGNORECASE)
    
    if matches:
        return matches[0].upper()
    
    # Try other common patterns
    patterns = [
        r"(?:answer:|choice is|option is|I choose|I select|choose|select)[^\w]*(A|B|C|D)",
        r"(?:option|choice)[^\w]*(A|B|C|D)[^\w]*(?:is correct|is the answer)",
        r"(A|B|C|D)[^\w]*(?:is the correct answer|is the answer)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()
    
    # If clear patterns fail, look for any standalone A, B, C, or D
    response = response.upper()
    options = ["A", "B", "C", "D"]
    for option in options:
        if re.search(r'\b' + option + r'\b', response):
            return option
    
    return None

# ========== AGENT EVALUATION ==========
async def evaluate_gpqa_sample(
    sample: Dict, 
    tools: Dict[str, Any], 
    ollama: OllamaClient
) -> Tuple[str, str, bool]:
    """Evaluate a single GPQA sample and return predicted answer and correctness"""
    
    # Format the prompt with the standardized GPQA template
    prompt = GPQA_PROMPT_TEMPLATE.format(
        question=sample["question"],
        option_a=sample["options"]["A"],
        option_b=sample["options"]["B"],
        option_c=sample["options"]["C"],
        option_d=sample["options"]["D"]
    )
    
    # Format the full system prompt
    tool_descriptions = "\n- ".join([
        f"{t['name']}: {t['schema']['function']['description']}" for t in tools.values()
    ]) if tools else "No tools available"
    
    full_prompt = SYSTEM_PROMPT.format(tools=tool_descriptions) + f"\n\nUser: {prompt}"
    
    # Get the model's response
    response = await ollama.generate_response(full_prompt)
    
    # Extract the answer
    predicted_answer = extract_answer_from_response(response)
    correct_answer = sample["correct_answer"]
    
    # Check if the prediction is correct
    is_correct = False
    if predicted_answer and predicted_answer == correct_answer:
        is_correct = True
    
    return predicted_answer, response, is_correct

# ========== MAIN EVALUATION LOOP ==========
async def run_gpqa_evaluation(
    ollama_model: str,
    dataset_name: str = "prompt-leaderboard/gpqa-100",
    num_samples: Optional[int] = None,
    output_file: str = "gpqa_results.json"
):
    """Run the GPQA evaluation and save results to a file"""
    
    # Load the GPQA dataset
    try:
        dataset = GPQADataset(dataset_name).load()
        samples = dataset.get_samples(limit=num_samples)
        if not samples:
            print("No valid samples found in the dataset!")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    print(f"Starting evaluation of {len(samples)} GPQA samples")
    
    # Initialize results container
    results = {
        "model": ollama_model,
        "dataset": {
            "source": dataset.source
        },
        "samples": []
    }

    # Initialize clients
    server_params = StdioServerParameters(
        command=MCP_SERVER_COMMAND,
        args=MCP_SERVER_ARGS
    )
    ollama = OllamaClient(model=ollama_model)
    
    # Run the evaluation
    correct_count = 0
    evaluated_count = 0
    
    async with MCPClient(server_params) as mcp_client:
        # Get available tools
        mcp_tools = await mcp_client.get_available_tools()
        tools = {
            tool.name: {
                "name": tool.name,
                "callable": mcp_client.call_tool(tool.name),
                "schema": {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
            }
            for tool in mcp_tools
        }
        
        print(f"Available tools: {[tool for tool in tools.keys()]}")
        
        # Process each sample
        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            question_id = sample["question_id"]
            correct_answer = sample["correct_answer"]
            
            # Evaluate the sample
            predicted_answer, response, is_correct = await evaluate_gpqa_sample(
                sample, tools, ollama
            )
            
            # Only count valid evaluations
            evaluated_count += 1
            if is_correct:
                correct_count += 1
            
            # Save the result
            sample_result = {
                "question_id": question_id,
                "question": sample["question"],
                "options": sample["options"],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "model_response": response,
                "is_correct": is_correct
            }
            results["samples"].append(sample_result)
            
            # Print progress
            acc_so_far = correct_count / max(evaluated_count, 1)  # Avoid division by zero
            print(f"Sample {i+1}/{len(samples)}: Question ID {question_id} - {'Correct' if is_correct else 'Incorrect'} (Acc so far: {acc_so_far:.2f})")
    
    # Calculate final accuracy
    accuracy = correct_count / evaluated_count if evaluated_count > 0 else 0
    results["metrics"] = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "evaluated_count": evaluated_count,
        "total_samples": len(samples)
    }
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete: {correct_count}/{evaluated_count} correct (Accuracy: {accuracy:.4f})")
    print(f"Results saved to {output_file}")
    
    return results

# ========== COMMAND LINE INTERFACE ==========
def main():
    parser = argparse.ArgumentParser(description="Evaluate an Ollama model on the GPQA benchmark")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama model to use")
    parser.add_argument("--dataset", type=str, default="prompt-leaderboard/gpqa-100", 
                        help="Hugging Face dataset path (default: prompt-leaderboard/gpqa-100)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (None for all)")
    parser.add_argument("--output", type=str, default="gpqa_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    asyncio.run(run_gpqa_evaluation(
        ollama_model=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        output_file=args.output
    ))

if __name__ == "__main__":
    main()