#!/usr/bin/env python3
"""
Demo script to visualize and explain the Resume Scoring workflow.
Run this script to generate visual representations of the workflow
using both Graphviz and Mermaid, and see a detailed explanation
of its components.
"""

from services.workflow import ResumeWorkflow
from services.workflow.visualizer import (
    WorkflowVisualizer,
    MermaidWorkflowVisualizer,
    print_workflow_summary
)

def main():
    # Create workflow instance
    workflow = ResumeWorkflow()
    
    # Print textual summary
    print_workflow_summary(workflow)
    
    # Generate and save Graphviz visualization
    print("\nGenerating Graphviz workflow visualization...")
    graph_file = WorkflowVisualizer.save_graph(workflow)
    print(f"Graphviz workflow graph saved to: {graph_file}")
    print("The graph should open automatically in your default image viewer.")
    
    # Generate Mermaid visualization
    print("\nGenerating Mermaid workflow visualization...")
    print("Copy the following Mermaid code to https://mermaid.live or a Markdown editor:")
    print("\n" + MermaidWorkflowVisualizer.create_mermaid_diagram(workflow) + "\n")
    
    # Try to display in IPython if available
    try:
        from IPython.display import display, Markdown
        print("IPython detected - attempting to display diagram directly...")
        MermaidWorkflowVisualizer.display_diagram(workflow)
    except ImportError:
        print("Note: For interactive visualization, run this in a Jupyter notebook")

if __name__ == "__main__":
    main() 