#!/usr/bin/env python3
"""
Example usage of the Notebook Editor Agent

This script demonstrates how to use the AI agent to edit large Jupyter notebooks.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import our agent
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from notebook_editor import NotebookEditorAgent, EditOperation, EditRequest


def create_sample_large_notebook():
    """Create a sample large notebook for testing."""
    import nbformat
    
    # Create a new notebook
    nb = nbformat.v4.new_notebook()
    
    # Add various types of cells to make it realistic
    cells = []
    
    # Title cell
    cells.append(nbformat.v4.new_markdown_cell("# Large Data Science Notebook\n\nThis is a sample large notebook for testing the AI editor."))
    
    # Import cells
    cells.append(nbformat.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report"""))
    
    # Data loading and exploration cells
    for i in range(20):
        if i % 3 == 0:
            cells.append(nbformat.v4.new_markdown_cell(f"## Section {i//3 + 1}: Data Analysis Step {i+1}"))
        
        if i % 2 == 0:
            cells.append(nbformat.v4.new_code_cell(f"""# Data processing step {i+1}
data_{i} = pd.DataFrame(np.random.randn(1000, 5), columns=['A', 'B', 'C', 'D', 'E'])
print(f"Data shape: {{data_{i}.shape}}")
data_{i}.head()"""))
        else:
            cells.append(nbformat.v4.new_code_cell(f"""# Analysis step {i+1}
plt.figure(figsize=(10, 6))
plt.plot(data_{i-1}.mean())
plt.title(f'Mean values for dataset {i}')
plt.show()"""))
    
    # Add some complex analysis cells
    cells.append(nbformat.v4.new_markdown_cell("## Machine Learning Analysis"))
    
    cells.append(nbformat.v4.new_code_cell("""# Prepare data for ML
X = data_0.drop('E', axis=1)
y = (data_0['E'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))"""))
    
    # Add more cells to make it large
    for i in range(30):
        cells.append(nbformat.v4.new_code_cell(f"""# Additional analysis {i+1}
result_{i} = np.random.randn(100)
print(f"Mean: {{np.mean(result_{i}):.3f}}, Std: {{np.std(result_{i}):.3f}}")"""))
    
    # Add conclusion
    cells.append(nbformat.v4.new_markdown_cell("## Conclusion\n\nThis notebook demonstrates various data analysis techniques."))
    
    nb.cells = cells
    
    # Save the notebook
    sample_path = current_dir / "sample_large_notebook.ipynb"
    with open(sample_path, 'w') as f:
        nbformat.write(nb, f)
    
    return str(sample_path)


def main():
    """Main example function."""
    print("🤖 Notebook Editor Agent Example")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: No OPENAI_API_KEY found in environment variables.")
        print("   Set your API key to use AI features:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Create a sample large notebook
        print("📝 Creating sample large notebook...")
        notebook_path = create_sample_large_notebook()
        print(f"   Created: {notebook_path}")
        
        # Initialize the agent
        print("\n🚀 Initializing Notebook Editor Agent...")
        agent = NotebookEditorAgent(
            model="gpt-4o-mini",  # Use cheaper model for demo
            max_cells_per_chunk=10,  # Smaller chunks for demo
            enable_backups=True
        )
        
        # Start editing session
        print("\n📖 Starting editing session...")
        session_result = agent.start_editing_session(
            notebook_path=notebook_path,
            user_goals=["Add better documentation", "Improve code organization", "Add error handling"]
        )
        
        if not session_result["success"]:
            print(f"❌ Failed to start session: {session_result['error']}")
            return
        
        print(f"✅ Session started: {session_result['session_id']}")
        print(f"   Notebook has {session_result['notebook_stats']['total_cells']} cells")
        print(f"   Split into {session_result['chunk_count']} chunks")
        
        # Analyze the notebook
        print("\n🔍 Analyzing notebook...")
        analysis_result = agent.analyze_notebook(detailed=True)
        
        if analysis_result["success"]:
            overview = analysis_result["overview"]
            print(f"   Processing strategy: {overview['processing_strategy']}")
            print(f"   Complexity: {overview['memory_estimate']['processing_complexity']}")
            
            if analysis_result["recommendations"]:
                print("   Recommendations:")
                for rec in analysis_result["recommendations"]:
                    print(f"   • {rec}")
        
        # Generate edit suggestions
        print("\n✏️  Generating edit suggestions...")
        edit_result = agent.edit_notebook(
            user_request="Add markdown documentation before each major code section and improve code comments",
            auto_apply=True  # Automatically apply high-priority suggestions
        )
        
        if edit_result["success"]:
            print(f"   Generated {len(edit_result['suggestions'])} suggestions")
            print(f"   Applied {len(edit_result['applied_edits'])} edits automatically")
            
            # Show applied edits
            for edit in edit_result["applied_edits"]:
                if edit["result"]["success"]:
                    operation = edit["suggestion"]["operation"]
                    reasoning = edit["suggestion"]["reasoning"]
                    print(f"   ✅ {operation}: {reasoning[:80]}...")
        
        # Save the edited notebook
        print("\n💾 Saving edited notebook...")
        save_result = agent.save_notebook(output_path=str(Path(notebook_path).with_suffix('.edited.ipynb')))
        
        if save_result["success"]:
            print(f"   ✅ Saved to: {save_result['notebook_path']}")
            if save_result["backup_path"]:
                print(f"   📋 Backup created: {save_result['backup_path']}")
        
        # Get session status
        print("\n📊 Session Status:")
        status = agent.get_session_status()
        print(f"   Session ID: {status['session_id']}")
        print(f"   Edits applied: {status['edit_count']}")
        print(f"   User goals: {', '.join(status['user_goals'])}")
        
        # End session
        print("\n🏁 Ending session...")
        summary = agent.end_session()
        print(f"   Session duration: {summary.get('duration', 0):.1f} seconds")
        print(f"   Total edits: {summary.get('total_edits', 0)}")
        
        print("\n🎉 Example completed successfully!")
        print(f"\nFiles created:")
        print(f"   • Original: {notebook_path}")
        print(f"   • Edited: {Path(notebook_path).with_suffix('.edited.ipynb')}")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()