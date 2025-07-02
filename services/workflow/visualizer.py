from typing import Dict, List, Set, Tuple
import graphviz
from IPython.display import display, Markdown
from .graph import ResumeWorkflow

class WorkflowVisualizer:
    """Utility class for visualizing the resume scoring workflow using Graphviz"""
    
    @staticmethod
    def create_graph(workflow: ResumeWorkflow) -> graphviz.Digraph:
        """Create a graphviz visualization of the workflow"""
        dot = graphviz.Digraph(comment='Resume Scoring Workflow')
        dot.attr(rankdir='TB')  # Top to bottom layout
        
        # Node styling
        dot.attr('node', shape='box', style='rounded')
        
        # Add nodes
        nodes = {
            "parse": "Resume Parser\n(PDF/Doc → Text)",
            "embed": "Text Embedder\n(Text → Vectors)",
            "similarity": "Similarity Scorer\n(Cosine Similarity)",
            "skills": "Technical Skills\nEvaluator",
            "culture": "Cultural Fit\nEvaluator",
            "combine": "Score Combiner",
            "feedback": "Feedback\nAnalyzer"
        }
        
        # Add nodes with different colors based on type
        for node_id, label in nodes.items():
            color = WorkflowVisualizer._get_node_color(node_id)
            dot.node(node_id, label, color=color)
        
        # Add regular edges
        edges = [
            ("parse", "embed"),
            ("embed", "similarity"),
            ("similarity", "skills"),
            ("skills", "culture"),
            ("culture", "combine"),
            ("combine", "feedback")
        ]
        for src, dst in edges:
            dot.edge(src, dst)
        
        # Add feedback loop back to skills evaluation
        dot.edge("feedback", "skills", "refine", color="blue", style="dashed")
        
        # Add end state
        dot.node("end", "End", shape="doublecircle")
        dot.edge("feedback", "end", "complete", color="green")
        
        return dot

    @staticmethod
    def _get_node_color(node_id: str) -> str:
        """Get color for different types of nodes"""
        colors = {
            "parse": "lightblue",
            "embed": "lightgreen",
            "similarity": "lightgreen",
            "skills": "pink",
            "culture": "pink",
            "combine": "lightyellow",
            "feedback": "lightgrey"
        }
        return colors.get(node_id, "white")

    @staticmethod
    def save_graph(workflow: ResumeWorkflow, filename: str = "workflow_graph") -> str:
        """Save the workflow visualization to a file"""
        dot = WorkflowVisualizer.create_graph(workflow)
        dot.render(filename, view=True, format='png', cleanup=True)
        return f"{filename}.png"

class MermaidWorkflowVisualizer:
    """Utility class for visualizing the resume scoring workflow using Mermaid"""
    
    @staticmethod
    def get_node_style(node_id: str) -> str:
        """Get Mermaid style for different node types"""
        styles = {
            "parse": "style parse fill:#d0e8ff",
            "embed": "style embed fill:#d0ffd0",
            "similarity": "style similarity fill:#d0ffd0",
            "skills": "style skills fill:#ffd0d0",
            "culture": "style culture fill:#ffd0d0",
            "combine": "style combine fill:#ffffd0",
            "feedback": "style feedback fill:#f0f0f0",
            "end": "style end fill:#e0e0e0"
        }
        return styles.get(node_id, "")

    @staticmethod
    def create_mermaid_diagram(workflow: ResumeWorkflow) -> str:
        """Create a Mermaid diagram representation of the workflow"""
        mermaid_code = [
            "```mermaid",
            "graph TB",
            "    %% Node definitions",
            "    parse[\"Resume Parser<br/>(PDF/Doc → Text)\"]",
            "    embed[\"Text Embedder<br/>(Text → Vectors)\"]",
            "    similarity[\"Similarity Scorer<br/>(Cosine)\"]",
            "    skills[\"Technical Skills<br/>Evaluator\"]",
            "    culture[\"Cultural Fit<br/>Evaluator\"]",
            "    combine[\"Score<br/>Combiner\"]",
            "    feedback[\"Feedback<br/>Analyzer\"]",
            "    end((End))",
            "",
            "    %% Node styles",
            "    classDef default stroke:#333,stroke-width:2px;",
            "    " + "\n    ".join([MermaidWorkflowVisualizer.get_node_style(node) for node in 
                                  ["parse", "embed", "similarity", "skills", "culture", "combine", "feedback", "end"]]),
            "",
            "    %% Regular edges",
            "    parse --> embed",
            "    embed --> similarity",
            "    similarity --> skills",
            "    skills --> culture",
            "    culture --> combine",
            "    combine --> feedback",
            "",
            "    %% Conditional edges",
            "    feedback -. \"refine\" .-> skills",
            "    feedback -- \"complete\" --> end",
            "```"
        ]
        return "\n".join(mermaid_code)

    @staticmethod
    def display_diagram(workflow: ResumeWorkflow) -> None:
        """Display the Mermaid diagram in a Jupyter notebook"""
        mermaid_diagram = MermaidWorkflowVisualizer.create_mermaid_diagram(workflow)
        display(Markdown(mermaid_diagram))

def print_workflow_summary(workflow: ResumeWorkflow) -> None:
    """Print a textual summary of the workflow"""
    print("Resume Scoring Workflow Summary")
    print("==============================")
    print("\nComponents:")
    print("1. Resume Parser (PDFResumeParser)")
    print("   - Extracts text from PDF/DOC files")
    
    print("\n2. Embedding Components:")
    print("   - Text Embedder: Converts text to vectors")
    print("   - Similarity Scorer: Computes cosine similarity")
    
    print("\n3. LLM-based Evaluators:")
    print("   - Technical Skills Evaluator")
    print("   - Cultural Fit Evaluator")
    print("   - Feedback Analyzer")
    
    print("\nWorkflow Steps:")
    print("1. Parse Resume & Job Description")
    print("2. Generate Embeddings")
    print("3. Compute Similarity Score")
    print("4. Evaluate Technical Skills")
    print("5. Evaluate Cultural Fit")
    print("6. Combine Scores")
    print("7. Get Feedback")
    print("8. Either:")
    print("   - Loop back to skills evaluation for refinement with feedback")
    print("   - End if satisfied or max iterations reached")
    
    print("\nScoring Weights:")
    print("- Technical Skills: 40%")
    print("- Cultural Fit: 30%")
    print("- Embedding Similarity: 30%") 