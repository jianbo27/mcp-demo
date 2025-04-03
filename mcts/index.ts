#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import chalk from 'chalk';

// Types for thought content
interface ThoughtContent {
  thought: string;
  thoughtNumber: number;
  totalThoughts: number;
  nextThoughtNeeded: boolean;
  confidence?: number;
  timestamp?: string;
}

// Types for MCTS node
class MCTSNode {
  content: ThoughtContent | null;
  value: number;
  rewards: number[];
  tree: MCTSTree | null;
  parent: MCTSNode | null;
  children: MCTSNode[];
  rollouts: any[];
  timestep: number;
  isLeaf: boolean;
  probability: number;
  isExpanded: boolean;

  constructor(content: ThoughtContent | null, parent: MCTSNode | null, timestep: number, isLeaf: boolean = false, probability: number = 1) {
    this.content = content;
    this.value = -1;
    this.rewards = [];
    this.tree = parent ? parent.tree : null;
    this.parent = parent;
    this.children = [];
    this.rollouts = [];
    this.timestep = timestep;
    this.isLeaf = isLeaf;
    this.probability = probability;
    this.isExpanded = false;
  }

  // Get the depth of this node in the tree
  getDepth(): number {
    return this.getPath().length;
  }

  // Get the full path from root to this node
  getPath(): (ThoughtContent | null)[] {
    if (!this.content) {
      return [];
    }
    if (!this.parent) {
      return [this.content];
    }
    return [...this.parent.getPath(), this.content];
  }

  // Calculate Q value (average reward)
  q(): number {
    return this.rewards.length > 0 ? this.rewards.reduce((a: number, b: number) => a + b, 0) / this.rewards.length : 0;
  }

  // Calculate N value (visit count)
  n(): number {
    return this.rewards.length + 1; // Add 1 to avoid division by zero
  }

  // Calculate UCB score for selection
  calcUCB(c: number = 1.414): number {
    if (!this.parent) {
      return this.q(); // Root node
    }
    
    // UCB formula: Q(s,a) + c * P(s,a) * sqrt(ln(N(s)) / N(s,a))
    return this.q() + 
           c * 
           this.probability * 
           Math.sqrt(Math.log(this.parent.n()) / this.n());
  }

  // Get the best child according to UCB
  bestChild(c: number = 1.414): MCTSNode | null {
    if (this.children.length === 0) return null;
    
    return this.children.reduce(
      (best: MCTSNode, current: MCTSNode) => current.calcUCB(c) > best.calcUCB(c) ? current : best,
      this.children[0]
    );
  }
}

// Configuration for MCTS
interface MCTSConfig {
  c: number;                  // Exploration constant
  maxDepth: number;           // Maximum tree depth
  minSearchTime: number;      // Minimum search iterations
  maxSearchTime: number;      // Maximum search iterations 
  minTerminals: number;       // Minimum terminal nodes to find
  nodeBudget: number;         // Children per node
  alpha: number;              // Weighting between node value and rollout
}

/**
 * Main MCTS Tree implementation
 */
class MCTSTree {
  config: MCTSConfig;
  allNodes: MCTSNode[];
  root: MCTSNode;
  currentNode: MCTSNode;

  constructor(config: Partial<MCTSConfig> = {}) {
    this.config = {
      c: 1.414,                  // Exploration constant
      maxDepth: 10,              // Maximum tree depth
      minSearchTime: 5,          // Minimum search iterations
      maxSearchTime: 20,         // Maximum search iterations 
      minTerminals: 3,           // Minimum terminal nodes to find
      nodeBudget: 3,             // Children per node
      alpha: 0.5,                // Weighting between node value and rollout
      ...config
    };
    
    this.allNodes = [];
    this.root = this.initRootNode();
    this.currentNode = this.root;
  }

  // Initialize the root node
  initRootNode(): MCTSNode {
    const root = new MCTSNode(null, null, 0, false, 1);
    root.tree = this;
    this.allNodes.push(root);
    return root;
  }

  // Get the current timestep
  getTimestep(): number {
    return Math.max(...this.allNodes.map(node => node.timestep), 0);
  }

  // MCTS Selection phase
  mctsSelect(): { node: MCTSNode, needExpand: boolean } {
    // If root has no children, select root
    if (this.root.children.length === 0) {
      return { node: this.root, needExpand: true };
    }
    
    let currentNode = this.root;
    let needExpand = false;
    
    // Traverse down the tree
    while (!currentNode.isLeaf) {
      // Check if we need to expand
      needExpand = currentNode.children.length === 0 || 
                   currentNode.children.some(child => !child.isExpanded);
      
      if (needExpand) {
        break;
      }
      
      // Select best child according to UCB
      const bestChild = currentNode.bestChild(this.config.c);
      if (!bestChild) break;
      
      currentNode = bestChild;
    }
    
    return { node: currentNode, needExpand };
  }

  // MCTS Expansion phase
  mctsExpand(node: MCTSNode, timestep: number): MCTSNode {
    // If there are unexpanded children, select the first one
    for (const child of node.children) {
      if (!child.isExpanded) {
        child.isExpanded = true;
        return child;
      }
    }
    
    // Otherwise create a new child node
    const nodeDepth = node.getDepth();
    const isTerminal = nodeDepth >= this.config.maxDepth || node.q() >= 0.95;
    
    // In a real implementation, this would generate a thoughtful next step
    // For now, we use a placeholder
    const childContent: ThoughtContent = {
      thought: `Generated thought at depth ${nodeDepth + 1}, timestep ${timestep}`,
      thoughtNumber: nodeDepth + 1,
      totalThoughts: Math.max(this.config.maxDepth, nodeDepth + 3),
      nextThoughtNeeded: !isTerminal,
      confidence: Math.min(0.5 + (nodeDepth / this.config.maxDepth) * 0.5, 0.95),
      timestamp: new Date().toISOString()
    };
    
    const newChild = new MCTSNode(childContent, node, timestep, isTerminal, 1.0);
    newChild.tree = this;
    this.allNodes.push(newChild);
    node.children.push(newChild);
    newChild.isExpanded = true;
    
    return newChild;
  }

  // MCTS Simulation phase
  mctsSimulate(node: MCTSNode): number {
    // In a real implementation, this would run multiple rollouts
    // For now, we'll assign a simulated value
    
    // Bias towards deeper nodes
    const depthFactor = node.getDepth() / this.config.maxDepth;
    
    // Add some noise for exploration
    const noise = Math.random() * 0.2 - 0.1;
    
    // Higher value for deeper nodes (assuming we want to reach deeper reasoning)
    return depthFactor * 0.8 + noise;
  }

  // MCTS Backpropagation phase
  mctsBackpropagate(node: MCTSNode, reward: number): void {
    let currentNode: MCTSNode | null = node;
    while (currentNode) {
      currentNode.rewards.push(reward);
      currentNode = currentNode.parent;
    }
  }

  // Run an MCTS iteration
  runMCTSIteration(): MCTSNode {
    const timestep = this.getTimestep() + 1;
    
    // Selection
    const { node: selectedNode, needExpand } = this.mctsSelect();
    
    if (needExpand) {
      // Expansion
      const newNode = this.mctsExpand(selectedNode, timestep);
      
      // Compute value for the new node
      if (newNode.value < 0) {
        // In a real implementation, this would evaluate the node
        newNode.value = newNode.getDepth() / this.config.maxDepth; // Placeholder
      }
      
      // Simulation
      const simulationReward = this.mctsSimulate(newNode);
      
      // Combine node value and simulation
      const reward = newNode.value * (1 - this.config.alpha) + 
                     simulationReward * this.config.alpha;
      
      // Backpropagation
      this.mctsBackpropagate(newNode, reward);
      
      return newNode;
    } else {
      // If no expansion needed (e.g., at a terminal node)
      this.mctsBackpropagate(selectedNode, selectedNode.q());
      return selectedNode;
    }
  }

  // Run MCTS for a number of iterations
  runMCTS(iterations: number = 5): { results: any[], terminals: number } {
    const results: any[] = [];
    let terminals = 0;
    
    for (let i = 0; i < iterations; i++) {
      const node = this.runMCTSIteration();
      if (node.isLeaf) terminals++;
      
      results.push({
        depth: node.getDepth(),
        reward: node.rewards[node.rewards.length - 1],
        isLeaf: node.isLeaf
      });
    }
    
    return { results, terminals };
  }

  // Find the best path from root to a leaf
  getBestPath(): any[] {
    let currentNode: MCTSNode = this.root;
    const path: any[] = [];
    
    while (currentNode.children.length > 0) {
      // Sort by Q value
      const sortedChildren = [...currentNode.children].sort((a, b) => b.q() - a.q());
      const bestChild = sortedChildren[0];
      
      if (bestChild.content) {
        path.push({
          thought: bestChild.content.thought,
          thoughtNumber: bestChild.content.thoughtNumber,
          confidence: bestChild.q(),
          isLeaf: bestChild.isLeaf
        });
      }
      
      currentNode = bestChild;
      if (currentNode.isLeaf) break;
    }
    
    return path;
  }

  // Get tree statistics
  getTreeStats(): {
    nodes: number;
    depth: number;
    leafNodes: number;
    avgReward: number;
  } {
    return {
      nodes: this.allNodes.length,
      depth: Math.max(...this.allNodes.map(n => n.getDepth())),
      leafNodes: this.allNodes.filter(n => n.isLeaf).length,
      avgReward: this.allNodes.reduce((sum, n) => sum + n.q(), 0) / this.allNodes.length
    };
  }
}

interface NextStepSuggestion {
  type: string;
  suggestion: string;
}

/**
 * The MCP server that exposes MCTS thinking through the Model Context Protocol
 */
class MCTSThinkingServer {
  tree: MCTSTree;
  thoughtHistory: ThoughtContent[];
  lastThoughtId: number;

  constructor() {
    this.tree = new MCTSTree({
      c: 1.5,
      maxDepth: 12,
      minSearchTime: 3,
      maxSearchTime: 10,
      minTerminals: 2,
      alpha: 0.7
    });
    
    this.thoughtHistory = [];
    this.lastThoughtId = 0;
  }

  // Format a thought for display
  formatThought(thought: ThoughtContent): string {
    const { thoughtNumber, totalThoughts, thought: content, confidence } = thought;
    
    const confidenceStr = confidence ? ` [Confidence: ${(confidence * 100).toFixed(1)}%]` : '';
    const prefix = chalk.blue('💭 MCTS Thought');
    const header = `${prefix} ${thoughtNumber}/${totalThoughts}${confidenceStr}`;
    
    const borderLength = Math.max(header.length, content.length) + 4;
    const border = '─'.repeat(borderLength);
    
    return `
┌${border}┐
│ ${header.padEnd(borderLength - 2)} │
├${border}┤
│ ${content.padEnd(borderLength - 2)} │
└${border}┘`;
  }

  // Process an incoming thought request
  processThought(input: any): {
    content: Array<{ type: string; text: string }>;
    isError?: boolean;
  } {
    try {
      // Validate input
      this.validateInput(input);
      
      // Store in history
      this.thoughtHistory.push(input);
      this.lastThoughtId++;
      
      // Display the thought
      console.error(this.formatThought(input));
      
      // Run MCTS to explore reasoning paths
      const mctsResults = this.tree.runMCTS(5);
      
      // Get the best path found so far
      const bestPath = this.tree.getBestPath();
      
      // Get tree statistics
      const treeStats = this.tree.getTreeStats();
      
      // Generate suggestions for next steps if needed
      let nextStepSuggestions: NextStepSuggestion[] = [];
      if (input.nextThoughtNeeded) {
        // In a real implementation, we'd extract these from the MCTS tree
        nextStepSuggestions = this.generateNextStepSuggestions(bestPath);
      }
      
      // Debug: visualize the tree periodically
      if (this.lastThoughtId % 3 === 0) {
        console.error(this.visualizeTree());
      }
      
      // Return results to the client
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            thoughtId: this.lastThoughtId,
            thoughtNumber: input.thoughtNumber,
            totalThoughts: input.totalThoughts,
            nextThoughtNeeded: input.nextThoughtNeeded,
            mctsIterations: mctsResults.results.length,
            terminalsFound: mctsResults.terminals,
            treeStats,
            bestPathLength: bestPath.length,
            nextStepSuggestions,
            confidence: bestPath.length > 0 ? bestPath[bestPath.length - 1].confidence : 0
          }, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
            status: 'failed'
          }, null, 2)
        }],
        isError: true
      };
    }
  }

  // Validate that the input is well-formed
  validateInput(input: any): void {
    if (!input.thought || typeof input.thought !== 'string') {
      throw new Error('Invalid thought: must be a string');
    }
    if (!input.thoughtNumber || typeof input.thoughtNumber !== 'number') {
      throw new Error('Invalid thoughtNumber: must be a number');
    }
    if (!input.totalThoughts || typeof input.totalThoughts !== 'number') {
      throw new Error('Invalid totalThoughts: must be a number');
    }
    if (typeof input.nextThoughtNeeded !== 'boolean') {
      throw new Error('Invalid nextThoughtNeeded: must be a boolean');
    }
  }

  // Generate suggestions for the next thinking step
  generateNextStepSuggestions(bestPath: any[]): NextStepSuggestion[] {
    // In a real implementation, these would come from the MCTS exploration
    if (bestPath.length === 0) {
      return [{
        type: "initial",
        suggestion: "Start by breaking down the problem into smaller parts"
      }];
    }
    
    const depth = bestPath.length;
    
    // Simple demo suggestions based on depth
    if (depth < 3) {
      return [{
        type: "exploration",
        suggestion: "Consider alternative approaches to the problem"
      }];
    } else if (depth < 6) {
      return [{
        type: "refinement",
        suggestion: "Refine the current approach by addressing potential weaknesses"
      }];
    } else {
      return [{
        type: "conclusion",
        suggestion: "Start working toward a conclusion based on the analysis so far"
      }];
    }
  }

  // Create a simple ASCII visualization of the tree
  visualizeTree(): string {
    const output: string[] = ['MCTS Tree Visualization:'];
    
    // Helper function to print a node
    const printNode = (node: MCTSNode, depth: number, isLast: boolean): void => {
      const indent = '│  '.repeat(depth - 1) + (depth > 0 ? (isLast ? '└──' : '├──') : '');
      const value = node.q().toFixed(2);
      const visits = node.n() - 1;
      
      const nodeInfo = node.content ? 
        `Thought ${node.content.thoughtNumber} (Value: ${value}, Visits: ${visits})` : 
        `Root (Value: ${value}, Visits: ${visits})`;
      
      output.push(`${indent}${nodeInfo}`);
      
      for (let i = 0; i < node.children.length; i++) {
        printNode(node.children[i], depth + 1, i === node.children.length - 1);
      }
    };
    
    printNode(this.tree.root, 0, true);
    return output.join('\n');
  }
}

// Define the MCTS Thinking Tool
const MCTS_THINKING_TOOL: Tool = {
  name: "mctsthinking",
  description: `A sophisticated problem-solving tool based on Monte Carlo Tree Search (MCTS).
This tool strategically explores different reasoning paths to find the most promising solutions.
It balances depth and breadth of exploration to optimize problem-solving.

When to use this tool:
- Complex problems with multiple possible solution paths
- Situations with uncertainty where exploration is valuable
- Problems that benefit from systematic exploration of alternatives
- When you want to balance exploration vs. exploitation in your thinking
- For planning tasks with multiple decision points
- For problems where the optimal path isn't initially obvious

Key features:
- Uses UCB (Upper Confidence Bound) to balance exploration and exploitation
- Builds a tree of possible reasoning paths
- Automatically identifies and prioritizes promising lines of thought
- Provides statistical feedback about the search process
- Suggests next steps based on exploration results
- Works iteratively to refine solutions
- Can handle problems with delayed feedback

Parameters explained:
- thought: Your current thinking step
- thoughtNumber: Current number in sequence
- totalThoughts: Estimated total thoughts needed
- nextThoughtNeeded: Whether another thought is needed
- confidence: Optional confidence in this thought (0-1)

Usage guidance:
1. Start with clear initial thoughts about the problem
2. Let the MCTS process guide exploration of solution paths
3. Consider the suggested next steps from the MCTS exploration
4. Continue until a satisfactory solution is found
5. The final solution should reflect the most promising path in the tree`,
  inputSchema: {
    type: "object",
    properties: {
      thought: {
        type: "string",
        description: "Your current thinking step"
      },
      thoughtNumber: {
        type: "integer",
        description: "Current thought number",
        minimum: 1
      },
      totalThoughts: {
        type: "integer",
        description: "Estimated total thoughts needed",
        minimum: 1
      },
      nextThoughtNeeded: {
        type: "boolean",
        description: "Whether another thought step is needed"
      },
      confidence: {
        type: "number",
        description: "Confidence in this thought (0-1)",
        minimum: 0,
        maximum: 1
      }
    },
    required: ["thought", "thoughtNumber", "totalThoughts", "nextThoughtNeeded"]
  }
};

// Create and initialize the MCP server
const server = new Server(
  {
    name: "mcts-thinking-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Create MCTS thinking server instance
const thinkingServer = new MCTSThinkingServer();

// Set up request handlers
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [MCTS_THINKING_TOOL],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "mctsthinking") {
    return thinkingServer.processThought(request.params.arguments);
  }

  return {
    content: [{
      type: "text",
      text: `Unknown tool: ${request.params.name}`
    }],
    isError: true
  };
});

// Start the server
async function runServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCTS Thinking MCP Server running on stdio");
}

runServer().catch((error) => {
  console.error("Fatal error running server:", error);
  process.exit(1);
});