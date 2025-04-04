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
MCP_SERVER_COMMAND = "node"
MCP_SERVER_ARGS = ["/home/kafka/servers/src/mcts/dist/index.js"]

SYSTEM_PROMPT = """You are a helpful assistant capable of solving complex mathematical problems and using external tools.

# Tools

{tools}

# Notes:
- Break down complex math problems into simpler steps.
- Use tools when relevant for calculations or reasoning.
- Show your work clearly for each step.
- Ensure your calculations are accurate.
- The AIME requires numerical answers, usually integers.
- For AIME problems, the answer is always an integer between 0 and 999, inclusive.
- Provide your final answer clearly at the end.
"""

# Standardized prompt format for AIME
AIME_PROMPT_TEMPLATE = """Solve the following AIME (American Invitational Mathematics Examination) problem step by step:

Problem: {problem}

Think through this problem step by step. Show all your work and calculations.
Remember that for AIME problems, the final answer is always an integer between 0 and 999, inclusive.
After you've solved the problem, provide your final answer in the format "The answer is: [your answer]"."""

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

    async def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Call a tool with the given arguments and return the response"""
        response = await self.session.call_tool(tool_name, arguments=arguments)
        return response.content[0].text

# ========== OLLAMA CLIENT ==========
class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    async def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.host}/api/generate", json=payload) as resp:
                    if resp.status != 200:
                        raise Exception(await resp.text())
                    data = await resp.json()
                    return data.get("response", "[No response from model]").strip()
        except Exception as e:
            return f"[Ollama Error: {e}]"

# ========== AIME DATASET ==========
class AIMEDataset:
    def __init__(self, dataset_name: str = "di-zhang-fdu/AIME_1983_2024", dataset_config: str = None, target_year: int = 2023):
        """Load the AIME dataset from Hugging Face"""
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset = None
        self.source = None
        self.target_year = target_year
    
    def load(self, split: str = "train"):
        """Load the dataset from Hugging Face"""
        try:
            print(f"Loading AIME dataset from Hugging Face: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name, split=split)
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
        filtered_count = 0
        
        for i, item in enumerate(self.dataset):
            try:
                # Get year and filter for target year
                year = item.get("Year", None)
                
                # Skip if not from the target year
                if year != self.target_year:
                    continue
                
                filtered_count += 1
                
                # Extract the problem and answer
                problem = item.get("Question", "")
                answer = item.get("Answer", None)
                
                # Skip if any required field is missing
                if not problem or answer is None:
                    missing_fields = []
                    if not problem: missing_fields.append("problem")
                    if answer is None: missing_fields.append("answer")
                    print(f"Skipping item {i}: Missing required fields - {', '.join(missing_fields)}")
                    continue
                
                # Create a unique ID for the sample
                problem_id = item.get("id", i)
                
                # Get problem number if available
                number = item.get("number", None)
                
                sample = {
                    "problem_id": problem_id,
                    "problem": problem,
                    "correct_answer": answer,
                    "year": year,
                    "number": number
                }
                
                samples.append(sample)
                
                # Print first sample for debugging
                if len(samples) == 1:
                    print(f"First processed sample: {json.dumps(sample, indent=2)}")
                
                if limit and len(samples) >= limit:
                    break
                    
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                print(f"Item details: {json.dumps(item, indent=2)}")
                continue
        
        print(f"Found {filtered_count} problems from AIME {self.target_year}")
        print(f"Successfully processed {len(samples)} valid samples")
        return samples

# ========== ANSWER EXTRACTION ==========
def extract_answer_from_response(response: str) -> Optional[int]:
    """Extract the numeric answer from the model response"""
    # Look for "The answer is: [number]" pattern
    pattern = r"The answer is:?\s*(\d+)"
    matches = re.findall(pattern, response, re.IGNORECASE)
    
    if matches:
        # Take the last match as the final answer
        try:
            return int(matches[-1])
        except ValueError:
            pass
    
    # Try other common patterns
    patterns = [
        r"(?:final answer|answer|conclusion)(?:\s+is)?:?\s*(\d+)",
        r"(?:therefore|thus|hence|so)(?:\s+the\s+answer\s+is)?:?\s*(\d+)",
        r"(?:=|equals)\s*(\d+)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                continue
    
    # As a last resort, find all numbers in the text and take the last one
    all_numbers = re.findall(r"\b(\d+)\b", response)
    if all_numbers:
        try:
            return int(all_numbers[-1])
        except ValueError:
            pass
    
    # If we can't determine, return None
    return None

# ========== MCTS INTEGRATION ==========
async def use_mcts_thinking(mcp_client, thoughts: List[Dict]) -> List[Dict]:
    """
    Helper function to run MCTS thinking with automated iterations
    
    Args:
        mcp_client: The MCP client instance
        thoughts: List of existing thoughts to process
        
    Returns:
        List of results from each thought processing
    """
    results = []
    
    for thought in thoughts:
        # Ensure the thought has the required fields
        if not all(k in thought for k in ["thought", "thoughtNumber", "totalThoughts", "nextThoughtNeeded"]):
            print(f"Warning: Skipping invalid thought: {thought}")
            continue
            
        # Call the MCTS thinking tool
        try:
            result_json = await mcp_client.call_tool("mctsthinking", thought)
            # Parse the JSON result
            result = json.loads(result_json)
            results.append(result)
            
            # Print some info about the MCTS progress
            print(f"Processed thought {thought['thoughtNumber']}/{thought['totalThoughts']} - " +
                  f"Tree stats: {result.get('treeStats', {}).get('nodes', 0)} nodes, " +
                  f"Confidence: {result.get('confidence', 0):.2f}")
                  
        except Exception as e:
            print(f"Error processing thought with MCTS: {e}")
            results.append({"error": str(e)})
    
    return results

# ========== EVALUATION FUNCTIONS ==========
async def evaluate_aime_sample(
    sample: Dict, 
    mcp_client: MCPClient,
    ollama: OllamaClient
) -> Tuple[Optional[int], str, bool]:
    """Evaluate a single AIME sample and return predicted answer and correctness"""
    
    # Format the prompt with the standardized AIME template
    prompt = AIME_PROMPT_TEMPLATE.format(
        problem=sample["problem"]
    )
    
    # Get available tools
    mcp_tools = await mcp_client.get_available_tools()
    tool_descriptions = "\n- ".join([
        f"{t.name}: {t.description}" for t in mcp_tools
    ]) if mcp_tools else "No tools available"
    
    # Format the full system prompt
    system_prompt = SYSTEM_PROMPT.format(tools=tool_descriptions)
    
    # Get the model's response
    response = await ollama.generate_response(prompt, system_prompt=system_prompt)
    
    # Extract thoughts from the response for MCTS processing
    thoughts = extract_thoughts_from_response(response)
    
    # Process thoughts with MCTS if any were extracted
    if thoughts:
        print(f"Extracted {len(thoughts)} thoughts for MCTS processing")
        mcts_results = await use_mcts_thinking(mcp_client, thoughts)
        
        # Optionally, you could use the MCTS results to refine the answer
        # For simplicity, we'll just log them for now
        print(f"MCTS processing complete - {len(mcts_results)} results")
    else:
        print("No thoughts extracted for MCTS processing")
    
    # Extract the answer from the original response
    predicted_answer = extract_answer_from_response(response)
    correct_answer = sample["correct_answer"]
    
    # Try to convert correct_answer to int if it's not already
    if not isinstance(correct_answer, int):
        try:
            correct_answer = int(correct_answer)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert correct_answer '{correct_answer}' to integer for problem {sample['problem_id']}")
            return predicted_answer, response, False
    
    # Check if the prediction is correct
    is_correct = False
    if predicted_answer is not None:
        is_correct = predicted_answer == correct_answer
    
    return predicted_answer, response, is_correct

def extract_thoughts_from_response(response: str) -> List[Dict]:
    """
    Extract thinking steps from the model's response to process with MCTS
    
    This is a simple implementation - you might want to refine this based on your model's output format
    """
    thoughts = []
    
    # Split response into paragraphs
    paragraphs = response.split('\n\n')
    
    # Filter out paragraphs that seem to be thinking steps
    thinking_steps = [p for p in paragraphs if len(p.strip()) > 20 and not p.strip().startswith("The answer is")]
    
    # Convert to MCTS format
    total_thoughts = len(thinking_steps)
    for i, step in enumerate(thinking_steps):
        thoughts.append({
            "thought": step.strip(),
            "thoughtNumber": i + 1,
            "totalThoughts": total_thoughts,
            "nextThoughtNeeded": i < total_thoughts - 1,
            "confidence": 0.7  # Default confidence
        })
    
    return thoughts

# ========== MAIN EVALUATION LOOP ==========
async def run_aime_evaluation(
    ollama_model: str,
    target_year: int = 2023,
    dataset_name: str = "di-zhang-fdu/AIME_1983_2024",
    dataset_config: str = None,
    dataset_split: str = "train",
    num_samples: Optional[int] = None,
    output_file: str = "aime_2023_mcp_results.json"
):
    """Run the AIME evaluation and save results to a file"""
    
    # Load the AIME dataset
    try:
        dataset = AIMEDataset(dataset_name, dataset_config, target_year=target_year).load(split=dataset_split)
        samples = dataset.get_samples(limit=num_samples)
        if not samples:
            print(f"No valid samples found for AIME {target_year}!")
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    print(f"Starting evaluation of {len(samples)} AIME samples")
    
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

    # Initialize Ollama client
    ollama = OllamaClient(model=ollama_model)
    
    # Initialize MCP server
    server_params = StdioServerParameters(
        command=MCP_SERVER_COMMAND,
        args=MCP_SERVER_ARGS
    )
    
    # Run the evaluation
    correct_count = 0
    evaluated_count = 0
    
    # Use a single MCP client for the entire evaluation process
    async with MCPClient(server_params) as mcp_client:
        # Process each sample
        for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
            problem_id = sample["problem_id"]
            correct_answer = sample["correct_answer"]
            
            # Evaluate the sample
            predicted_answer, response, is_correct = await evaluate_aime_sample(
                sample, mcp_client, ollama
            )
            
            # Only count valid evaluations
            evaluated_count += 1
            if is_correct:
                correct_count += 1
            
            # Save the result
            sample_result = {
                "problem_id": problem_id,
                "problem": sample["problem"],
                "year": sample.get("year"),
                "number": sample.get("number"),
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "model_response": response,
                "is_correct": is_correct
            }
            results["samples"].append(sample_result)
            
            # Print progress
            acc_so_far = correct_count / max(evaluated_count, 1)  # Avoid division by zero
            print(f"Sample {i+1}/{len(samples)}: Problem ID {problem_id} - {'Correct' if is_correct else 'Incorrect'} (Acc so far: {acc_so_far:.2f})")
    
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
    parser = argparse.ArgumentParser(description="Evaluate an Ollama model on the AIME benchmark with MCP tools")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama model to use")
    parser.add_argument("--year", type=int, default=2023, help="AIME year to evaluate (default: 2023)")
    parser.add_argument("--dataset", type=str, default="di-zhang-fdu/AIME_1983_2024", help="Hugging Face dataset name")
    parser.add_argument("--config", type=str, default=None, help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (usually 'train' for AIME)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (None for all)")
    parser.add_argument("--output", type=str, default="aime_2023_mcp_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    asyncio.run(run_aime_evaluation(
        ollama_model=args.model,
        target_year=args.year,
        dataset_name=args.dataset,
        dataset_config=args.config,
        dataset_split=args.split,
        num_samples=args.num_samples,
        output_file=args.output
    ))

if __name__ == "__main__":
    main()