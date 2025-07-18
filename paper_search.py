import arxiv
import requests
from typing import List, Dict, Any
import re
from scholarly import scholarly
import time

class PaperSearcher:
    def __init__(self):
        self.arxiv_client = arxiv.Client()
        
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for search"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cannot', 'using', 'based', 'approach', 'method', 'paper', 'study', 'research', 'analysis', 'application', 'system', 'model', 'framework', 'algorithm', 'technique'}
        
        # Extract words (remove punctuation and convert to lowercase)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        key_terms = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return unique terms, prioritizing longer ones
        return list(dict.fromkeys(sorted(key_terms, key=len, reverse=True)))
    
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search ArXiv for papers"""
        papers = []
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in self.arxiv_client.results(search):
                papers.append({
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'url': paper.entry_id,
                    'published': paper.published.strftime('%Y-%m-%d'),
                    'source': 'ArXiv',
                    'categories': [cat for cat in paper.categories]
                })
        except Exception as e:
            print(f"Error searching ArXiv: {e}")
        
        return papers
    
    def search_google_scholar(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Google Scholar for papers"""
        papers = []
        try:
            search_query = scholarly.search_pubs(query)
            
            for i, paper in enumerate(search_query):
                if i >= max_results:
                    break
                    
                try:
                    papers.append({
                        'title': paper.get('title', 'Unknown Title'),
                        'authors': paper.get('author', []),
                        'abstract': paper.get('abstract', 'No abstract available'),
                        'url': paper.get('eprint_url', paper.get('pub_url', '')),
                        'published': str(paper.get('year', 'Unknown')),
                        'source': 'Google Scholar',
                        'citations': paper.get('num_citations', 0),
                        'venue': paper.get('venue', 'Unknown Venue')
                    })
                    
                    # Add delay to avoid rate limiting
                    time.sleep(1)
                except Exception as e:
                    print(f"Error processing paper from Google Scholar: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
        
        return papers
    
    def search_semantic_scholar(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Semantic Scholar for papers"""
        papers = []
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'title,authors,abstract,year,url,citationCount,venue,publicationTypes'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                for paper in data.get('data', []):
                    papers.append({
                        'title': paper.get('title', 'Unknown Title'),
                        'authors': [author.get('name', 'Unknown') for author in paper.get('authors', [])],
                        'abstract': paper.get('abstract', 'No abstract available'),
                        'url': paper.get('url', ''),
                        'published': str(paper.get('year', 'Unknown')),
                        'source': 'Semantic Scholar',
                        'citations': paper.get('citationCount', 0),
                        'venue': paper.get('venue', 'Unknown Venue')
                    })
        except Exception as e:
            print(f"Error searching Semantic Scholar: {e}")
        
        return papers
    
    def search_papers(self, key_terms: List[str], max_results: int = 10) -> List[Dict]:
        """Search for papers using multiple sources"""
        # Create search query from key terms
        query = ' '.join(key_terms[:5])  # Use top 5 key terms
        
        all_papers = []
        
        # Search ArXiv
        arxiv_papers = self.search_arxiv(query, max_results // 3)
        all_papers.extend(arxiv_papers)
        
        # Search Semantic Scholar
        semantic_papers = self.search_semantic_scholar(query, max_results // 3)
        all_papers.extend(semantic_papers)
        
        # Search Google Scholar (with rate limiting)
        try:
            scholar_papers = self.search_google_scholar(query, max_results // 3)
            all_papers.extend(scholar_papers)
        except Exception as e:
            print(f"Google Scholar search failed: {e}")
        
        # Remove duplicates based on title similarity
        unique_papers = []
        seen_titles = set()
        
        for paper in all_papers:
            title_lower = paper['title'].lower()
            # Simple duplicate detection
            is_duplicate = any(
                self._similarity_score(title_lower, seen_title) > 0.8 
                for seen_title in seen_titles
            )
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(title_lower)
        
        # Sort by relevance (citations, recency, etc.)
        unique_papers.sort(key=lambda x: (
            x.get('citations', 0),
            int(x.get('published', '0').split('-')[0]) if x.get('published', '0') != 'Unknown' else 0
        ), reverse=True)
        
        return unique_papers[:max_results]
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Simple similarity score between two strings"""
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0