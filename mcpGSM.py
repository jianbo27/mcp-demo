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

SYSTEM_PROMPT = """You are a helpful assistant capable of solving math problems and using external tools.

# Tools

{tools}

# Notes:
- Solve the math problem step by step.
- Use tools when relevant for calculations or reasoning.
- Provide your final answer clearly at the end.
- Express your final answer as a single number or value.
"""

# Standardized prompt format for GSM8K 
GSM8K_PROMPT_TEMPLATE = """Solve the following math problem step by step:

{question}

Your answer should be a single number or value. First solve the problem step by step, then provide your final answer in the format "The answer is: [answer]"."""

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

# ========== GSM8K DATASET ==========
class GSM8KDataset:
    def __init__(self, dataset_name: str = "gsm8k", dataset_config: str = "main"):
        """Load the GSM8K dataset from Hugging Face"""
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset = None
        self.source = None
    
    def load(self, split: str = "test"):
        """Load the dataset from Hugging Face"""
        try:
            print(f"Loading GSM8K dataset from Hugging Face: {self.dataset_name}/{self.dataset_config}")
            self.dataset = load_dataset(self.dataset_name, self.dataset_config, split=split)
            self.source = f"Hugging Face: {self.dataset_name}/{self.dataset_config}"
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
                # Extract the question and answer
                question = item.get("question", "")
                answer_with_work = item.get("answer", "")
                
                # GSM8K format has the final answer at the end after the working
                # We need to extract the final numeric answer
                final_answer = self._extract_final_answer(answer_with_work)
                
                # Skip if any required field is missing
                if not question or final_answer is None:
                    missing_fields = []
                    if not question: missing_fields.append("question")
                    if final_answer is None: missing_fields.append("answer")
                    print(f"Skipping item {i}: Missing required fields - {', '.join(missing_fields)}")
                    continue
                
                # Create a unique ID for the sample
                question_id = i
                
                sample = {
                    "question_id": question_id,
                    "question": question,
                    "answer_with_work": answer_with_work,
                    "final_answer": final_answer
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
    
    def _extract_final_answer(self, answer_with_work: str) -> Optional[str]:
        """Extract the final numeric answer from the answer string"""
        # In GSM8K, the final answer is typically the last number in the string,
        # sometimes preceded by "The answer is" or similar
        
        # First try to find "The answer is X" pattern
        match = re.search(r"The answer is[^\d.-]*(-?\d*\.?\d+)", answer_with_work, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Then try to find "= X" pattern at the end
        match = re.search(r"=\s*(-?\d*\.?\d+)(?:\s*\.)?$", answer_with_work)
        if match:
            return match.group(1).strip()
        
        # Finally, try to extract the last number in the text
        matches = re.findall(r"(-?\d*\.?\d+)", answer_with_work)
        if matches:
            return matches[-1].strip()
        
        return None

# ========== ANSWER EXTRACTION ==========
def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract the numeric answer from the model response"""
    # Look for "The answer is: X" pattern
    pattern = r"The answer is:?\s*(-?\d*\.?\d+)"
    matches = re.findall(pattern, response, re.IGNORECASE)
    
    if matches:
        return matches[-1].strip()  # Take the last match as the final answer
    
    # Try other common patterns
    patterns = [
        r"(?:answer|result)(?:\s+is)?:?\s*(-?\d*\.?\d+)",
        r"=\s*(-?\d*\.?\d+)(?:\s*\.)?$",
        r"(?:equals|equal to)\s*(-?\d*\.?\d+)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()  # Take the last match as the final answer
    
    # As a last resort, take the last number in the response
    matches = re.findall(r"(-?\d*\.?\d+)", response)
    if matches:
        return matches[-1].strip()
    
    return None

# ========== AGENT EVALUATION ==========
async def evaluate_gsm8k_sample(
    sample: Dict, 
    tools: Dict[str, Any], 
    ollama: OllamaClient
) -> Tuple[str, str, bool]:
    """Evaluate a single GSM8K sample and return predicted answer and correctness"""
    
    # Format the prompt with the standardized GSM8K template
    prompt = GSM8K_PROMPT_TEMPLATE.format(
        question=sample["question"]
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
    correct_answer = sample["final_answer"]
    
    # Check if the prediction is correct
    # For GSM8K, we need to compare numbers numerically, not just string comparison
    is_correct = False
    if predicted_answer:
        try:
            # Convert to float for numeric comparison
            predicted_float = float(predicted_answer)
            correct_float = float(correct_answer)
            
            # Using exact match for GSM8K (could also consider approximate match with tolerance)
            is_correct = predicted_float == correct_float
            
        except (ValueError, TypeError):
            # If conversion fails, fall back to string comparison
            is_correct = predicted_answer == correct_answer
    
    return predicted_answer, response, is_correct

# ========== MAIN EVALUATION LOOP ==========
async def run_gsm8k_evaluation(
    ollama_model: str,
    dataset_name: str = "gsm8k",
    dataset_config: str = "main",
    dataset_split: str = "test",
    num_samples: Optional[int] = None,
    output_file: str = "gsm8k_results.json"
):
    """Run the GSM8K evaluation and save results to a file"""
    
    # Load the GSM8K dataset
    try:
        dataset = GSM8KDataset(dataset_name, dataset_config).load(split=dataset_split)
        samples = dataset.get_samples(limit=num_samples)
        if not samples:
            print("No valid samples found in the dataset!")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    print(f"Starting evaluation of {len(samples)} GSM8K samples")
    
    # Initialize results container
    results = {
        "model": ollama_model,
        "dataset": {
            "source": dataset.source,
            "config": dataset_config,
            "split": dataset_split
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
            correct_answer = sample["final_answer"]
            
            # Evaluate the sample
            predicted_answer, response, is_correct = await evaluate_gsm8k_sample(
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
    parser = argparse.ArgumentParser(description="Evaluate an Ollama model on the GSM8K benchmark")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama model to use")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Hugging Face dataset name")
    parser.add_argument("--config", type=str, default="main", help="Dataset configuration")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (None for all)")
    parser.add_argument("--output", type=str, default="gsm8k_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    asyncio.run(run_gsm8k_evaluation(
        ollama_model=args.model,
        dataset_name=args.dataset,
        dataset_config=args.config,
        dataset_split=args.split,
        num_samples=args.num_samples,
        output_file=args.output
    ))

if __name__ == "__main__":
    main()