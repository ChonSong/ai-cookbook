#!/usr/bin/env python3
"""
Demonstration of the Notebook Editor Agent (without OpenAI API)

This script demonstrates the core functionality of the notebook editor agent
without requiring an OpenAI API key, focusing on the non-AI components.
"""

import sys
from pathlib import Path
import nbformat

# Add the current directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from notebook_utils import (
    NotebookChunker, NotebookValidator, EditRequest, EditOperation
)
from memory import NotebookMemory
from tools import NotebookTools


def create_large_sample_notebook():
    """Create a realistic large notebook for demonstration."""
    print("📝 Creating large sample notebook...")
    
    nb = nbformat.v4.new_notebook()
    cells = []
    
    # Add title and introduction
    cells.append(nbformat.v4.new_markdown_cell("""# Large Data Science Project
    
This notebook demonstrates various data science techniques and analyses.
It contains multiple sections with code, visualizations, and documentation.

## Table of Contents
1. Data Loading and Exploration
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Results and Conclusions"""))
    
    # Add import section
    cells.append(nbformat.v4.new_markdown_cell("## 1. Data Loading and Exploration"))
    cells.append(nbformat.v4.new_code_cell("""# Standard data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')"""))
    
    # Add many data exploration cells
    for i in range(15):
        if i % 5 == 0:
            cells.append(nbformat.v4.new_markdown_cell(f"### Dataset {i//5 + 1} Analysis"))
        
        cells.append(nbformat.v4.new_code_cell(f"""# Dataset {i+1} loading and initial exploration
data_{i} = pd.DataFrame({{
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'feature_3': np.random.randint(0, 10, 1000),
    'target': np.random.randint(0, 2, 1000)
}})

print(f"Dataset {i+1} shape: {{data_{i}.shape}}")
print(f"Missing values: {{data_{i}.isnull().sum().sum()}}")
print(f"Target distribution: {{data_{i}['target'].value_counts()}}")

# Basic statistics
print("\\nBasic Statistics:")
print(data_{i}.describe())"""))
        
        # Add visualization cells
        if i % 3 == 0:
            cells.append(nbformat.v4.new_code_cell(f"""# Visualization for dataset {i+1}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Dataset {i+1} Visualizations')

# Distribution plots
data_{i}['feature_1'].hist(ax=axes[0,0], bins=30)
axes[0,0].set_title('Feature 1 Distribution')

data_{i}['feature_2'].hist(ax=axes[0,1], bins=30)
axes[0,1].set_title('Feature 2 Distribution')

# Scatter plot
axes[1,0].scatter(data_{i}['feature_1'], data_{i}['feature_2'], 
                 c=data_{i}['target'], alpha=0.6)
axes[1,0].set_title('Feature 1 vs Feature 2')

# Box plot
data_{i}.boxplot(column=['feature_1', 'feature_2'], ax=axes[1,1])
axes[1,1].set_title('Feature Distributions')

plt.tight_layout()
plt.show()"""))
    
    # Add preprocessing section
    cells.append(nbformat.v4.new_markdown_cell("## 2. Data Preprocessing"))
    
    for i in range(10):
        cells.append(nbformat.v4.new_code_cell(f"""# Preprocessing step {i+1}
# Handle missing values and outliers
processed_data_{i} = data_{i % 15}.copy()

# Remove outliers using IQR method
Q1 = processed_data_{i}['feature_1'].quantile(0.25)
Q3 = processed_data_{i}['feature_1'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

processed_data_{i} = processed_data_{i}[
    (processed_data_{i}['feature_1'] >= lower_bound) & 
    (processed_data_{i}['feature_1'] <= upper_bound)
]

print(f"Removed {{len(data_{i % 15}) - len(processed_data_{i})}} outliers")
print(f"Final shape: {{processed_data_{i}.shape}}")"""))
    
    # Add model training section
    cells.append(nbformat.v4.new_markdown_cell("## 3. Model Training and Evaluation"))
    
    for i in range(8):
        model_type = ['RandomForest', 'GradientBoosting', 'LogisticRegression'][i % 3]
        cells.append(nbformat.v4.new_code_cell(f"""# {model_type} Model {i+1}
# Prepare data for training
X = processed_data_{i % 10}[['feature_1', 'feature_2', 'feature_3']]
y = processed_data_{i % 10}['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 + {i}
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
if '{model_type}' == 'RandomForest':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
elif '{model_type}' == 'GradientBoosting':
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
else:
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

# Make predictions
if '{model_type}' == 'LogisticRegression':
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
else:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
print(f"{model_type} Model {i+1} Results:")
print(f"Accuracy: {{(y_pred == y_test).mean():.3f}}")
print(f"AUC-ROC: {{roc_auc_score(y_test, y_prob):.3f}}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))"""))
    
    # Add results section
    cells.append(nbformat.v4.new_markdown_cell("""## 4. Results and Conclusions

### Key Findings:
- Multiple datasets were analyzed with various preprocessing techniques
- Different machine learning models were trained and evaluated
- Performance metrics were calculated for each model

### Next Steps:
- Hyperparameter tuning
- Feature selection
- Ensemble methods
- Production deployment

### Notes:
This analysis demonstrates a comprehensive machine learning pipeline
with proper data preprocessing, model training, and evaluation."""))
    
    nb.cells = cells
    
    # Save the notebook
    sample_path = current_dir / "large_sample_notebook.ipynb"
    with open(sample_path, 'w') as f:
        nbformat.write(nb, f)
    
    print(f"   ✅ Created large notebook with {len(cells)} cells")
    print(f"   📁 Saved to: {sample_path}")
    
    return str(sample_path)


def demonstrate_chunking(notebook_path):
    """Demonstrate notebook chunking functionality."""
    print("\n🔪 Demonstrating Notebook Chunking...")
    
    # Load notebook
    tools = NotebookTools(backup_enabled=False)
    notebook = tools.load_notebook(notebook_path)
    
    print(f"   📊 Original notebook: {len(notebook.cells)} cells")
    
    # Test different chunk sizes
    for chunk_size in [10, 20, 50]:
        chunker = NotebookChunker(max_cells_per_chunk=chunk_size)
        chunks = chunker.chunk_notebook(notebook)
        
        print(f"   📦 Chunk size {chunk_size}: {len(chunks)} chunks")
        
        # Show chunk details
        for i, chunk in enumerate(chunks):
            print(f"      Chunk {i+1}: cells {chunk.start_index}-{chunk.end_index} ({chunk.size} cells)")
    
    # Memory estimation
    chunker = NotebookChunker()
    memory_estimate = chunker.estimate_memory_usage(notebook)
    print(f"\n   🧠 Memory Analysis:")
    print(f"      Total cells: {memory_estimate['total_cells']}")
    print(f"      Content length: {memory_estimate['total_content_length']:,} characters")
    print(f"      Processing complexity: {memory_estimate['processing_complexity']}")
    print(f"      Recommended chunks: {memory_estimate['estimated_chunks']}")


def demonstrate_validation(notebook_path):
    """Demonstrate notebook validation functionality."""
    print("\n✅ Demonstrating Notebook Validation...")
    
    # Load notebook
    tools = NotebookTools(backup_enabled=False)
    notebook = tools.load_notebook(notebook_path)
    
    # Validate notebook
    validator = NotebookValidator()
    validation_result = validator.validate_notebook(notebook)
    
    print(f"   📋 Validation Results:")
    print(f"      Valid: {validation_result['is_valid']}")
    print(f"      Cell count: {validation_result['cell_count']}")
    
    if validation_result['issues']:
        print(f"      Issues: {len(validation_result['issues'])}")
        for issue in validation_result['issues']:
            print(f"        - {issue}")
    
    if validation_result['warnings']:
        print(f"      Warnings: {len(validation_result['warnings'])}")
        for warning in validation_result['warnings']:
            print(f"        - {warning}")
    
    # Test edit request validation
    print(f"\n   🔍 Testing Edit Request Validation:")
    
    # Valid edit request
    valid_edit = EditRequest(
        operation=EditOperation.MODIFY_CELL,
        target_index=0,
        content="# Modified Title\n\nThis title has been modified."
    )
    
    valid_result = validator.validate_edit_request(valid_edit, len(notebook.cells))
    print(f"      Valid edit request: {valid_result['is_valid']}")
    
    # Invalid edit request
    invalid_edit = EditRequest(
        operation=EditOperation.MODIFY_CELL,
        target_index=999,  # Out of range
        content="Some content"
    )
    
    invalid_result = validator.validate_edit_request(invalid_edit, len(notebook.cells))
    print(f"      Invalid edit request: {invalid_result['is_valid']}")
    if invalid_result['issues']:
        for issue in invalid_result['issues']:
            print(f"        - {issue}")


def demonstrate_tools(notebook_path):
    """Demonstrate notebook tools functionality."""
    print("\n🔧 Demonstrating Notebook Tools...")
    
    # Load notebook
    tools = NotebookTools(backup_enabled=True)
    notebook = tools.load_notebook(notebook_path)
    
    # Get statistics
    stats = tools.get_notebook_statistics(notebook)
    print(f"   📊 Notebook Statistics:")
    print(f"      Total cells: {stats['total_cells']}")
    print(f"      Cell types: {stats['cell_types']}")
    print(f"      Total lines: {stats['total_lines']:,}")
    print(f"      Total characters: {stats['total_characters']:,}")
    print(f"      Has outputs: {stats['has_outputs']}")
    
    # Extract code cells
    code_cells = tools.extract_code_cells(notebook)
    print(f"      Code cells: {len(code_cells)}")
    
    # Extract markdown cells
    markdown_cells = tools.extract_markdown_cells(notebook)
    print(f"      Markdown cells: {len(markdown_cells)}")
    
    # Demonstrate editing operations
    print(f"\n   ✏️  Testing Edit Operations:")
    
    # Test modify cell
    original_content = notebook.cells[0].source
    modify_request = EditRequest(
        operation=EditOperation.MODIFY_CELL,
        target_index=0,
        content="# Modified Large Data Science Project\n\nThis notebook has been modified by the AI agent."
    )
    
    modify_result = tools.apply_edit(notebook, modify_request)
    print(f"      Modify cell: {modify_result['success']}")
    
    # Test insert cell
    insert_request = EditRequest(
        operation=EditOperation.INSERT_CELL,
        target_index=1,
        content="## Agent Modification Note\n\nThis cell was inserted by the notebook editor agent.",
        cell_type="markdown"
    )
    
    insert_result = tools.apply_edit(notebook, insert_request)
    print(f"      Insert cell: {insert_result['success']}")
    
    # Save modified notebook
    modified_path = str(Path(notebook_path).with_suffix('.modified.ipynb'))
    save_result = tools.save_notebook(notebook, modified_path)
    print(f"      Save notebook: {save_result['success']}")
    
    if save_result['success']:
        print(f"        Saved to: {save_result['notebook_path']}")
        if save_result['backup_path']:
            print(f"        Backup: {save_result['backup_path']}")
    
    # Restore original content
    notebook.cells[0].source = original_content


def demonstrate_memory():
    """Demonstrate memory system functionality."""
    print("\n🧠 Demonstrating Memory System...")
    
    # Initialize memory
    memory = NotebookMemory(storage_path="./demo_memory")
    
    # Start session
    session_id = memory.start_session(
        notebook_path="large_sample_notebook.ipynb",
        user_goals=["Add documentation", "Improve code quality", "Optimize performance"]
    )
    
    print(f"   📝 Started session: {session_id}")
    
    # Add conversation history
    memory.add_conversation(
        user_input="Please add more documentation to this notebook",
        agent_response="I'll add markdown cells with explanations before each major section",
        context={"operation": "add_documentation"}
    )
    
    memory.add_conversation(
        user_input="Can you improve the code organization?",
        agent_response="I'll split large cells and add better comments",
        context={"operation": "improve_organization"}
    )
    
    # Record edit operations
    edit_request = EditRequest(
        operation=EditOperation.ADD_MARKDOWN,
        target_index=1,
        content="## Documentation added by AI agent"
    )
    
    memory.record_edit(edit_request, {"success": True, "description": "Added documentation"})
    
    # Update session context
    memory.update_session_context("cells_modified", 3)
    memory.update_session_context("documentation_added", True)
    
    # Get context for LLM
    context = memory.get_context_for_llm()
    print(f"   💭 Session Context Preview:")
    print(f"      {context[:200]}...")
    
    # Get session status
    if memory.current_session:
        print(f"   📊 Session Info:")
        print(f"      User goals: {len(memory.current_session.user_goals)} goals")
        print(f"      Edit history: {len(memory.current_session.edit_history)} edits")
        print(f"      Conversation: {len(memory.conversation_history)} exchanges")
    
    # End session
    summary = memory.end_session()
    print(f"   🏁 Session ended:")
    print(f"      Duration: {summary.get('duration', 0):.1f} seconds")
    print(f"      Total edits: {summary.get('total_edits', 0)}")


def main():
    """Main demonstration function."""
    print("🤖 Notebook Editor Agent Demonstration")
    print("=" * 60)
    print("This demo shows the core functionality without requiring OpenAI API")
    print("=" * 60)
    
    try:
        # Create sample notebook
        notebook_path = create_large_sample_notebook()
        
        # Demonstrate each component
        demonstrate_chunking(notebook_path)
        demonstrate_validation(notebook_path)
        demonstrate_tools(notebook_path)
        demonstrate_memory()
        
        print("\n🎉 Demonstration completed successfully!")
        print("\nFiles created:")
        print(f"   • {notebook_path}")
        print(f"   • {Path(notebook_path).with_suffix('.modified.ipynb')}")
        print("\nTo use with AI features, set OPENAI_API_KEY and run example.py")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()