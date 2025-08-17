# AI Agent for Editing Extremely Large Jupyter Notebooks

This directory contains a complete AI agent that can efficiently process and edit extremely large Jupyter notebooks. The agent is built using the 7 foundational building blocks of AI agents and can handle notebooks with hundreds of cells through intelligent chunking and processing strategies.

## Features

### 🤖 Core Capabilities
- **Memory-efficient processing** of large notebooks through intelligent chunking
- **AI-powered understanding** of notebook content and structure
- **Intelligent editing capabilities** with validation and error handling
- **Maintains notebook integrity** and proper formatting
- **Supports various editing operations** (cell modification, insertion, deletion, etc.)

### 🏗️ Architecture (7 Building Blocks)

1. **Intelligence** (`intelligence.py`) - AI understanding and generation using LLMs
2. **Memory** (`memory.py`) - Context persistence across interactions
3. **Tools** (`tools.py`) - Concrete notebook operations and file handling
4. **Validation** (`notebook_utils.py`) - Ensuring notebook integrity
5. **Control** (`agent.py`) - Decision making and process flow
6. **Recovery** - Error handling and rollback mechanisms
7. **Feedback** - Learning from edit results and user interactions

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-openai-api-key"
```

### Basic Usage

```python
from notebook_editor import NotebookEditorAgent

# Initialize the agent
agent = NotebookEditorAgent()

# Start editing session
session = agent.start_editing_session(
    notebook_path="large_notebook.ipynb",
    user_goals=["Add documentation", "Improve code organization"]
)

# Analyze the notebook
analysis = agent.analyze_notebook(detailed=True)

# Generate and apply edits
edits = agent.edit_notebook(
    user_request="Add markdown documentation before each major code section",
    auto_apply=True
)

# Save the edited notebook
agent.save_notebook(output_path="edited_notebook.ipynb")

# End session
agent.end_session()
```

### Run the Example

```bash
cd agents/notebook-editor
python example.py
```

## Key Components

### NotebookChunker
Handles splitting large notebooks into manageable chunks for processing:
- Configurable chunk size limits
- Content-aware chunking strategies
- Memory usage estimation

### NotebookIntelligence
AI-powered analysis and editing suggestions:
- Structured analysis using Pydantic models
- Context-aware edit suggestions
- Code explanation and documentation generation

### NotebookMemory
Persistent memory system for maintaining context:
- Session management and history
- Conversation tracking
- Caching of analysis results

### NotebookTools
File operations and notebook manipulations:
- Safe notebook loading and saving
- Atomic operations with backup support
- Various cell operations (modify, insert, delete, etc.)

## Supported Edit Operations

- **MODIFY_CELL** - Change existing cell content
- **INSERT_CELL** - Add new cells at specific positions
- **DELETE_CELL** - Remove cells
- **MOVE_CELL** - Reorder cells
- **SPLIT_CELL** - Split large cells into smaller ones
- **MERGE_CELLS** - Combine related cells
- **ADD_MARKDOWN** - Add documentation cells
- **UPDATE_METADATA** - Modify notebook or cell metadata

## Processing Strategies

The agent automatically selects optimal processing strategies based on notebook size:

- **Full Notebook Processing** - For small notebooks (<50 cells)
- **Chunked Processing** - For medium notebooks (50-200 cells)
- **Chunked Processing with Caching** - For large notebooks (>200 cells)

## Memory Management

- **Intelligent Chunking** - Splits large notebooks while preserving context
- **Analysis Caching** - Caches AI analysis results to avoid redundant processing
- **Session Persistence** - Maintains state across interactions
- **Backup System** - Automatic backups before modifications

## Error Handling & Recovery

- **Validation** - Comprehensive notebook and edit request validation
- **Atomic Operations** - Safe file operations with rollback capability
- **Backup System** - Automatic backups before any modifications
- **Error Recovery** - Graceful handling of failures with detailed error reporting

## Configuration Options

```python
agent = NotebookEditorAgent(
    model="gpt-4o",                    # OpenAI model to use
    max_cells_per_chunk=50,            # Maximum cells per processing chunk
    enable_backups=True,               # Enable automatic backups
    memory_storage_path="./memory"     # Path for persistent memory storage
)
```

## Example Workflows

### 1. Large Notebook Analysis
```python
# Analyze a 500-cell notebook
analysis = agent.analyze_notebook(detailed=True)
print(f"Processing strategy: {analysis['overview']['processing_strategy']}")
print(f"Recommendations: {analysis['recommendations']}")
```

### 2. Automated Documentation
```python
# Add documentation throughout a notebook
edits = agent.edit_notebook(
    user_request="Add markdown cells explaining each major code section",
    auto_apply=True
)
```

### 3. Code Organization
```python
# Improve code structure
edits = agent.edit_notebook(
    user_request="Split large cells into smaller, focused cells and add comments",
    auto_apply=False  # Review suggestions first
)

# Review and apply specific edits
for suggestion in edits['suggestions']:
    if suggestion['priority'] == 'high':
        edit_request = EditRequest(...)
        agent.apply_edit(edit_request)
```

## Performance Considerations

- **Chunking** - Large notebooks are processed in chunks to manage memory usage
- **Caching** - AI analysis results are cached to avoid redundant API calls
- **Lazy Loading** - Only relevant chunks are analyzed when needed
- **Batch Operations** - Multiple edits can be batched for efficiency

## Integration with Existing Workflows

The agent can be integrated into various workflows:
- **CI/CD Pipelines** - Automated notebook quality checks
- **Jupyter Extensions** - Direct integration with Jupyter Lab
- **Batch Processing** - Processing multiple notebooks
- **API Services** - Notebook editing as a service

## Extending the Agent

The modular architecture makes it easy to extend:

```python
# Custom validation
class CustomValidator(NotebookValidator):
    def validate_custom_rules(self, notebook):
        # Your custom validation logic
        pass

# Custom intelligence
class CustomIntelligence(NotebookIntelligence):
    def custom_analysis(self, chunk):
        # Your custom analysis logic
        pass
```

## Troubleshooting

### Common Issues

1. **Memory Errors** - Reduce `max_cells_per_chunk` parameter
2. **API Rate Limits** - Implement rate limiting or use caching
3. **Large Outputs** - Consider preprocessing to remove large outputs
4. **Validation Errors** - Check notebook format and fix issues before processing

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

agent = NotebookEditorAgent(...)
```

## Contributing

1. Follow the existing architectural patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure compatibility with the 7 building blocks architecture

## License

This project is part of the AI Cookbook and follows the repository's license terms.