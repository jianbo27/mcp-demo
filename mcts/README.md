# MCTS Thinking MCP Server

An MCP server implementation that provides a tool for strategic problem-solving through Monte Carlo Tree Search (MCTS), systematically exploring and evaluating different reasoning paths.

## Features

- Strategically explore multiple solution paths simultaneously
- Balance exploration of new ideas with exploitation of promising ones
- Automatically prioritize the most promising lines of reasoning
- Build a complete tree of possible reasoning approaches
- Receive statistical feedback on exploration progress
- Visualize the reasoning tree for deeper insight

## Tool

### mctsthinking

Facilitates an advanced thinking process using Monte Carlo Tree Search for problem-solving and decision-making.

**Inputs:**
- `thought` (string): Your current thinking step
- `thoughtNumber` (integer): Current thought number
- `totalThoughts` (integer): Estimated total thoughts needed
- `nextThoughtNeeded` (boolean): Whether another thought step is needed
- `confidence` (number, optional): Confidence in this thought (0-1)

## Usage

The MCTS Thinking tool is designed for:
- Complex problems with multiple possible solution paths
- Situations with uncertainty where exploration is valuable
- Problems that benefit from systematic exploration of alternatives
- Planning tasks with multiple decision points
- Problems where the optimal path isn't initially obvious
- When you want to balance exploration vs. exploitation in your thinking

## Configuration

### Usage with Claude Desktop

Add this to your `claude_desktop_config.json`:

#### npx

```json
{
  "mcpServers": {
    "mcts-thinking": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/mcts"
      ]
    }
  }
}
```

#### docker

```json
{
  "mcpServers": {
    "mctsthinking": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "mcp/mctsthinking"
      ]
    }
  }
}
```

## Building

Docker:

```bash
docker build -t mcp/mctsthinking -f src/mctsthinking/Dockerfile .
```

## How It Works

The MCTS Thinking server uses the following process:
1. **Selection**: Starting from the root, it selects nodes using UCB until reaching a leaf or unexpanded node
2. **Expansion**: Creates new thought nodes as children of the selected node
3. **Simulation**: Estimates the value of new paths through rollouts
4. **Backpropagation**: Updates node statistics based on simulation results

This approach provides a sophisticated balance between exploring new ideas and focusing on the most promising ones, leading to more effective problem-solving.

## License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.