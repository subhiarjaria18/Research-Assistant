from typing import List, Dict, Any
import json
import re
from llm_manager import LLMManager
from document_processor import DocumentProcessor
from paper_search import PaperSearcher

class ResearchAgent:
    def __init__(self, llm_provider: str = "together", model: str = None):
        self.llm = LLMManager(provider=llm_provider, model=model)
        self.doc_processor = DocumentProcessor()
        self.paper_searcher = PaperSearcher()
        
    def summarize_paper(self, paper_text: str, max_length: int = 500) -> str:
        """Generate a comprehensive summary of the research paper"""
        prompt = f"""
        Please provide a comprehensive summary of this research paper in {max_length} words or less.
        
        Focus on:
        1. Main research question/objective
        2. Key methodology used
        3. Primary findings/results
        4. Significance of the work
        5. Main conclusions
        
        Paper text:
        {paper_text[:4000]}...
        
        Summary:
        """
        
        return self.llm.generate_text(prompt, max_tokens=max_length//2)
    
    def extract_citations(self, paper_text: str) -> Dict[str, List[str]]:
        """Extract and categorize citations from the paper"""
        citations = self.doc_processor.extract_citations(paper_text)
        
        prompt = f"""
        Analyze the following citations and categorize them. Return a JSON object with:
        - "in_text_citations": List of in-text citations found
        - "reference_list": List of full references (if available)
        - "citation_count": Total number of citations
        
        Citations found: {citations[:50]}  # Limit for prompt size
        
        Paper text excerpt:
        {paper_text[-2000:]}  # Look at end where references usually are
        
        Return only valid JSON:
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=1000)
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "in_text_citations": citations,
            "reference_list": [],
            "citation_count": len(citations)
        }
    
    def extract_objective(self, paper_text: str) -> str:
        """Extract the main objective/research question"""
        prompt = f"""
        Extract the main research objective, research question, or hypothesis from this paper.
        Be specific and concise. If multiple objectives exist, list them clearly.
        
        Paper text:
        {paper_text[:3000]}
        
        Research Objective(s):
        """
        
        return self.llm.generate_text(prompt, max_tokens=300)
    
    def extract_introduction(self, paper_text: str) -> str:
        """Extract and summarize the introduction section"""
        sections = self.doc_processor.extract_sections(paper_text)
        intro_text = sections.get('introduction', '')
        
        if not intro_text:
            # Try to find introduction in full text
            intro_match = re.search(r'(?i)introduction\s*(.{500,2000})', paper_text, re.DOTALL)
            intro_text = intro_match.group(1) if intro_match else paper_text[:1500]
        
        prompt = f"""
        Summarize the introduction section of this research paper. Include:
        1. Background and context
        2. Problem statement
        3. Research gap identified
        4. Proposed solution/approach
        
        Introduction text:
        {intro_text}
        
        Introduction Summary:
        """
        
        return self.llm.generate_text(prompt, max_tokens=400)
    
    def extract_methodology(self, paper_text: str) -> str:
        """Extract and summarize the methodology section"""
        sections = self.doc_processor.extract_sections(paper_text)
        method_text = sections.get('methodology', '')
        
        if not method_text:
            # Try alternative patterns
            method_match = re.search(r'(?i)(?:methodology|methods?|approach)\s*(.{500,2000})', paper_text, re.DOTALL)
            method_text = method_match.group(1) if method_match else ""
        
        if not method_text:
            return "Methodology section not clearly identified in the paper."
        
        prompt = f"""
        Summarize the methodology section of this research paper. Include:
        1. Research design/approach
        2. Data collection methods
        3. Analysis techniques
        4. Tools and technologies used
        5. Experimental setup (if applicable)
        
        Methodology text:
        {method_text}
        
        Methodology Summary:
        """
        
        return self.llm.generate_text(prompt, max_tokens=400)
    
    def extract_results(self, paper_text: str) -> str:
        """Extract and summarize the results section"""
        sections = self.doc_processor.extract_sections(paper_text)
        results_text = sections.get('results', '')
        
        if not results_text:
            results_match = re.search(r'(?i)(?:results|findings)\s*(.{500,2000})', paper_text, re.DOTALL)
            results_text = results_match.group(1) if results_match else ""
        
        if not results_text:
            return "Results section not clearly identified in the paper."
        
        prompt = f"""
        Summarize the results section of this research paper. Include:
        1. Key findings
        2. Statistical results (if any)
        3. Performance metrics
        4. Observations
        5. Data analysis outcomes
        
        Results text:
        {results_text}
        
        Results Summary:
        """
        
        return self.llm.generate_text(prompt, max_tokens=400)
    
    def identify_research_gap(self, paper_text: str) -> str:
        """Identify research gaps mentioned in the paper"""
        prompt = f"""
        Identify and summarize the research gaps mentioned in this paper. Look for:
        1. Limitations of current work
        2. Areas for future research
        3. Unsolved problems
        4. Gaps in existing literature
        5. Suggestions for improvement
        
        Paper text:
        {paper_text[:4000]}
        
        Research Gaps:
        """
        
        return self.llm.generate_text(prompt, max_tokens=350)
    
    def find_similar_papers(self, paper_title: str, paper_text: str = None) -> List[Dict]:
        """Find similar papers based on title and content"""
        try:
            # Extract key terms from title and abstract
            key_terms = self.paper_searcher.extract_key_terms(paper_title)
            if paper_text:
                sections = self.doc_processor.extract_sections(paper_text)
                abstract = sections.get('abstract', '')
                if abstract:
                    key_terms.extend(self.paper_searcher.extract_key_terms(abstract))
            
            # Search for similar papers
            similar_papers = self.paper_searcher.search_papers(key_terms[:10])  # Limit key terms
            
            return similar_papers
        except Exception as e:
            print(f"Error finding similar papers: {e}")
            return []
    
    def analyze_paper(self, paper_path: str) -> Dict[str, Any]:
        """Complete analysis of a research paper"""
        try:
            # Extract text from PDF
            paper_text = self.doc_processor.extract_text_from_pdf(paper_path)
            paper_title = paper_path.split('/')[-1].replace('.pdf', '')
            
            # Store in vector database
            collection_name, sections, citations = self.doc_processor.store_document(paper_path)
            
            # Generate all analyses
            analysis = {
                'title': paper_title,
                'summary': self.summarize_paper(paper_text),
                'citations': self.extract_citations(paper_text),
                'objective': self.extract_objective(paper_text),
                'introduction': self.extract_introduction(paper_text),
                'methodology': self.extract_methodology(paper_text),
                'results': self.extract_results(paper_text),
                'research_gap': self.identify_research_gap(paper_text),
                'similar_papers': self.find_similar_papers(paper_title, paper_text),
                'sections': sections,
                'collection_name': collection_name
            }
            
            return analysis
            
        except Exception as e:
            return {'error': f"Error analyzing paper: {str(e)}"}
    
    def query_paper(self, query: str, collection_name: str = None) -> str:
        """Query the paper using RAG"""
        # Search relevant chunks
        search_results = self.doc_processor.search_documents(query, collection_name)
        
        if not search_results:
            return "No relevant information found in the paper."
        
        # Prepare context from search results
        context = "\n\n".join([result['content'] for result in search_results[:3]])
        
        prompt = f"""
        Based on the following context from the research paper, answer the question:
        
        Question: {query}
        
        Context:
        {context}
        
        Answer:
        """
        
        return self.llm.generate_text(prompt, max_tokens=500)