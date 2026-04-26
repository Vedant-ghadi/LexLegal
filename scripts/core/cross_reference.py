import re
from collections import defaultdict

class CrossRefGraph:
    """In-memory legal cross-reference graph (replaces Neo4j on Kaggle)."""
    
    # Patterns that indicate cross-references in legal text
    XREF_PATTERNS = [
        r'(?:as\s+defined\s+in|set\s+forth\s+in|pursuant\s+to|under|described\s+in'
        r'|referenced\s+in|referred\s+to\s+in|subject\s+to|in\s+accordance\s+with)'
        r'\s+(?:Section|Clause|Article|Paragraph|Subsection|Exhibit|Schedule|Annex)'
        r'\s*(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))*)',
        
        r'(?:Section|Clause|Article|Paragraph|Subsection)\s*'
        r'(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))*)',
        
        r'["\u201c]([A-Z][a-zA-Z\s]+?)["\u201d]\s+(?:has\s+the\s+meaning|shall\s+mean|means)',
    ]
    
    def __init__(self):
        self.nodes = {}           # chunk_id -> chunk_data
        self.edges = defaultdict(list)  # chunk_id -> [(target_chunk_id, rel_type)]
        self.section_map = {}     # (file_path, section_id) -> [chunk_ids]
        self.definition_map = {}  # (file_path, term) -> [chunk_ids]
    
    def add_chunk(self, chunk_id, chunk_data):
        self.nodes[chunk_id] = chunk_data
        text = chunk_data.get('raw_text', chunk_data['text'])
        fp = chunk_data['file_path']
        
        # Detect section headers: "Section 4.2" etc.
        sec_matches = re.findall(
            r'^(?:Section|Clause|Article)\s*(\d+(?:\.\d+)*)', text, re.MULTILINE
        )
        for sec_id in sec_matches:
            key = (fp, sec_id)
            self.section_map.setdefault(key, []).append(chunk_id)
        
        # Detect definitions: "Confidential Information" means...
        def_matches = re.findall(
            r'["\u201c]([A-Z][a-zA-Z\s]{2,40}?)["\u201d]\s+(?:has\s+the\s+meaning|shall\s+mean|means)',
            text
        )
        for term in def_matches:
            key = (fp, term.strip().lower())
            self.definition_map.setdefault(key, []).append(chunk_id)
    
    def build_edges(self):
        """After all chunks added, resolve cross-references into edges."""
        for cid, cdata in self.nodes.items():
            text = cdata.get('raw_text', cdata['text'])
            fp = cdata['file_path']
            
            # Find referenced sections
            for pattern in self.XREF_PATTERNS[:2]:
                refs = re.findall(pattern, text, re.IGNORECASE)
                for ref in refs:
                    key = (fp, ref)
                    if key in self.section_map:
                        for target in self.section_map[key]:
                            if target != cid:
                                self.edges[cid].append((target, 'REFERENCES'))
            
            # Find referenced defined terms
            for (dfp, dterm), def_chunks in self.definition_map.items():
                if dfp == fp and dterm in text.lower():
                    for target in def_chunks:
                        if target != cid:
                            self.edges[cid].append((target, 'USES_DEFINITION'))
    
    def get_related(self, chunk_ids, max_hops=1):
        """Traverse graph from given chunks, return related chunk IDs."""
        visited = set(chunk_ids)
        frontier = list(chunk_ids)
        
        for _ in range(max_hops):
            next_frontier = []
            for cid in frontier:
                for target, rel in self.edges.get(cid, []):
                    if target not in visited:
                        visited.add(target)
                        next_frontier.append(target)
            frontier = next_frontier
        
        # Return only the NEW chunks (not the originals)
        return [cid for cid in visited if cid not in set(chunk_ids)]
    
    def stats(self):
        n_edges = sum(len(v) for v in self.edges.values())
        n_refs = sum(1 for v in self.edges.values() for _, r in v if r == 'REFERENCES')
        n_defs = sum(1 for v in self.edges.values() for _, r in v if r == 'USES_DEFINITION')
        return f'{len(self.nodes)} nodes | {n_edges} edges ({n_refs} refs, {n_defs} defs) | {len(self.section_map)} sections | {len(self.definition_map)} definitions'