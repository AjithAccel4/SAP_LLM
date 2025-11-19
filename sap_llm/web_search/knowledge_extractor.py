"""
Knowledge Extractor for Search Results.

Extracts structured knowledge from search results and prepares it for
integration with knowledge bases and prompt markup systems.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeEntry:
    """Structured knowledge entry extracted from search result."""

    def __init__(
        self,
        content: str,
        source_url: str,
        source_type: str,
        title: str,
        trust_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize knowledge entry.

        Args:
            content: Extracted content
            source_url: Source URL
            source_type: Type of source (api_doc, tutorial, forum, etc.)
            title: Entry title
            trust_score: Trust score (0.0-1.0)
            metadata: Additional metadata
        """
        self.content = content
        self.source_url = source_url
        self.source_type = source_type
        self.title = title
        self.trust_score = trust_score
        self.metadata = metadata or {}
        self.extracted_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "title": self.title,
            "trust_score": self.trust_score,
            "metadata": self.metadata,
            "extracted_at": self.extracted_at.isoformat()
        }


class KnowledgeExtractor:
    """
    Extract structured knowledge from search results.

    Features:
    - Content fetching and parsing
    - Schema extraction for APIs
    - Documentation structure extraction
    - Code snippet extraction
    - Metadata enrichment
    - Deduplication

    Example:
        >>> extractor = KnowledgeExtractor()
        >>> entries = extractor.extract_from_results(search_results)
        >>> for entry in entries:
        ...     print(entry.title, entry.trust_score)
    """

    # API-related patterns
    API_PATTERNS = {
        "endpoint": re.compile(r'(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-{}]+)'),
        "parameter": re.compile(r'(\w+)\s*:\s*(\w+)'),
        "response_code": re.compile(r'\b(200|201|204|400|401|403|404|500)\b')
    }

    # Code block patterns
    CODE_PATTERNS = {
        "json": re.compile(r'```json\n(.*?)\n```', re.DOTALL),
        "python": re.compile(r'```python\n(.*?)\n```', re.DOTALL),
        "javascript": re.compile(r'```(?:js|javascript)\n(.*?)\n```', re.DOTALL),
        "sql": re.compile(r'```sql\n(.*?)\n```', re.DOTALL)
    }

    def __init__(
        self,
        fetch_full_content: bool = False,
        max_content_length: int = 10000,
        timeout: float = 10.0
    ):
        """
        Initialize knowledge extractor.

        Args:
            fetch_full_content: Whether to fetch full page content
            max_content_length: Maximum content length to extract
            timeout: Request timeout for fetching content
        """
        self.fetch_full_content = fetch_full_content
        self.max_content_length = max_content_length
        self.timeout = timeout

        # Statistics
        self.stats = {
            "total_extracted": 0,
            "api_docs_extracted": 0,
            "tutorials_extracted": 0,
            "forum_posts_extracted": 0,
            "fetch_errors": 0
        }

        logger.info(
            f"KnowledgeExtractor initialized "
            f"(fetch_full_content={fetch_full_content})"
        )

    def extract_from_results(
        self,
        results: List[Dict[str, Any]],
        min_trust_score: float = 0.6
    ) -> List[KnowledgeEntry]:
        """
        Extract knowledge entries from search results.

        Args:
            results: List of search results
            min_trust_score: Minimum trust score to extract

        Returns:
            List of knowledge entries
        """
        knowledge_entries = []

        for result in results:
            trust_score = result.get("trust_score", 0.5)

            # Skip low-trust results
            if trust_score < min_trust_score:
                continue

            try:
                # Determine source type
                source_type = self._determine_source_type(result)

                # Extract based on source type
                if source_type == "api_documentation":
                    entry = self._extract_api_documentation(result)
                    self.stats["api_docs_extracted"] += 1
                elif source_type == "tutorial":
                    entry = self._extract_tutorial(result)
                    self.stats["tutorials_extracted"] += 1
                elif source_type == "forum":
                    entry = self._extract_forum_post(result)
                    self.stats["forum_posts_extracted"] += 1
                else:
                    entry = self._extract_general_knowledge(result)

                if entry:
                    knowledge_entries.append(entry)
                    self.stats["total_extracted"] += 1

            except Exception as e:
                logger.warning(f"Failed to extract knowledge from {result.get('url')}: {e}")
                self.stats["fetch_errors"] += 1

        logger.info(
            f"Extracted {len(knowledge_entries)} knowledge entries from "
            f"{len(results)} results"
        )

        return knowledge_entries

    def _determine_source_type(self, result: Dict[str, Any]) -> str:
        """Determine the type of source from result metadata."""
        url = result.get("url", "").lower()
        title = result.get("title", "").lower()

        # Check metadata first
        metadata = result.get("trust_metadata", {})
        if "content_type" in metadata:
            return metadata["content_type"]

        # Fallback to URL/title analysis
        if "/api/" in url or "api reference" in title:
            return "api_documentation"
        elif "/tutorial/" in url or "tutorial" in title or "getting started" in title:
            return "tutorial"
        elif "/question" in url or "/answers/" in url or "community" in url:
            return "forum"
        elif "help.sap.com" in url:
            return "official_help"
        else:
            return "general"

    def _extract_api_documentation(
        self,
        result: Dict[str, Any]
    ) -> Optional[KnowledgeEntry]:
        """
        Extract API documentation.

        Args:
            result: Search result

        Returns:
            Knowledge entry or None
        """
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        content_parts = []
        metadata = {
            "endpoints": [],
            "parameters": [],
            "response_codes": []
        }

        # Use snippet as base content
        content_parts.append(snippet)

        # Fetch full content if enabled
        if self.fetch_full_content:
            full_content = self._fetch_page_content(url)
            if full_content:
                content_parts.append(full_content[:self.max_content_length])

                # Extract API-specific info
                metadata["endpoints"] = self._extract_endpoints(full_content)
                metadata["parameters"] = self._extract_parameters(full_content)
                metadata["response_codes"] = self._extract_response_codes(full_content)

        content = "\n\n".join(content_parts)

        return KnowledgeEntry(
            content=content,
            source_url=url,
            source_type="api_documentation",
            title=title,
            trust_score=result.get("trust_score", 0.5),
            metadata=metadata
        )

    def _extract_tutorial(
        self,
        result: Dict[str, Any]
    ) -> Optional[KnowledgeEntry]:
        """
        Extract tutorial content.

        Args:
            result: Search result

        Returns:
            Knowledge entry or None
        """
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        metadata = {
            "code_snippets": [],
            "steps": []
        }

        content_parts = [snippet]

        # Fetch full content if enabled
        if self.fetch_full_content:
            full_content = self._fetch_page_content(url)
            if full_content:
                # Extract code snippets
                metadata["code_snippets"] = self._extract_code_snippets(full_content)

                # Extract numbered steps
                metadata["steps"] = self._extract_steps(full_content)

                content_parts.append(full_content[:self.max_content_length])

        content = "\n\n".join(content_parts)

        return KnowledgeEntry(
            content=content,
            source_url=url,
            source_type="tutorial",
            title=title,
            trust_score=result.get("trust_score", 0.5),
            metadata=metadata
        )

    def _extract_forum_post(
        self,
        result: Dict[str, Any]
    ) -> Optional[KnowledgeEntry]:
        """
        Extract forum post content.

        Args:
            result: Search result

        Returns:
            Knowledge entry or None
        """
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        # Forum posts have lower trust by default
        trust_score = min(result.get("trust_score", 0.5), 0.7)

        metadata = {
            "post_type": "question" if "/question" in url else "discussion",
            "has_accepted_answer": False
        }

        return KnowledgeEntry(
            content=snippet,  # Usually just snippet for forums
            source_url=url,
            source_type="forum",
            title=title,
            trust_score=trust_score,
            metadata=metadata
        )

    def _extract_general_knowledge(
        self,
        result: Dict[str, Any]
    ) -> Optional[KnowledgeEntry]:
        """
        Extract general knowledge.

        Args:
            result: Search result

        Returns:
            Knowledge entry or None
        """
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        return KnowledgeEntry(
            content=snippet,
            source_url=url,
            source_type="general",
            title=title,
            trust_score=result.get("trust_score", 0.5),
            metadata={}
        )

    def _fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch full page content.

        Args:
            url: Page URL

        Returns:
            Extracted text content or None
        """
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return text

        except Exception as e:
            logger.debug(f"Failed to fetch content from {url}: {e}")
            return None

    def _extract_endpoints(self, content: str) -> List[Dict[str, str]]:
        """Extract API endpoints from content."""
        endpoints = []
        matches = self.API_PATTERNS["endpoint"].findall(content)

        for method, path in matches[:10]:  # Limit to 10
            endpoints.append({"method": method, "path": path})

        return endpoints

    def _extract_parameters(self, content: str) -> List[Dict[str, str]]:
        """Extract API parameters from content."""
        parameters = []
        matches = self.API_PATTERNS["parameter"].findall(content)

        for name, param_type in matches[:20]:  # Limit to 20
            parameters.append({"name": name, "type": param_type})

        return parameters

    def _extract_response_codes(self, content: str) -> List[str]:
        """Extract HTTP response codes from content."""
        matches = self.API_PATTERNS["response_code"].findall(content)
        return list(set(matches))

    def _extract_code_snippets(self, content: str) -> List[Dict[str, str]]:
        """Extract code snippets from content."""
        snippets = []

        for language, pattern in self.CODE_PATTERNS.items():
            matches = pattern.findall(content)
            for match in matches[:5]:  # Limit per language
                snippets.append({
                    "language": language,
                    "code": match.strip()
                })

        return snippets

    def _extract_steps(self, content: str) -> List[str]:
        """Extract numbered steps from tutorial content."""
        # Look for numbered lists
        step_pattern = re.compile(r'^\d+\.\s+(.+)$', re.MULTILINE)
        matches = step_pattern.findall(content)

        return matches[:20]  # Limit to 20 steps

    def update_knowledge_base(
        self,
        entries: List[KnowledgeEntry],
        knowledge_base: Any
    ) -> int:
        """
        Update knowledge base with extracted entries.

        Args:
            entries: List of knowledge entries
            knowledge_base: Knowledge base to update (e.g., PMG, vector store)

        Returns:
            Number of entries added
        """
        added_count = 0

        for entry in entries:
            try:
                # Add to knowledge base
                # This is a placeholder - actual implementation depends on KB interface
                if hasattr(knowledge_base, "add_knowledge"):
                    knowledge_base.add_knowledge(entry.to_dict())
                    added_count += 1

            except Exception as e:
                logger.error(f"Failed to add entry to knowledge base: {e}")

        logger.info(f"Added {added_count} entries to knowledge base")
        return added_count

    def export_to_json(
        self,
        entries: List[KnowledgeEntry],
        filepath: str
    ) -> bool:
        """
        Export knowledge entries to JSON file.

        Args:
            entries: List of knowledge entries
            filepath: Output file path

        Returns:
            True if successful
        """
        try:
            import json

            data = {
                "extracted_at": datetime.now().isoformat(),
                "entry_count": len(entries),
                "entries": [entry.to_dict() for entry in entries]
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported {len(entries)} entries to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export entries: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get extraction statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "fetch_full_content": self.fetch_full_content,
            "max_content_length": self.max_content_length
        }
