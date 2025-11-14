"""
Knowledge Augmentation Engine

Auto-extracts patterns from successful extractions, builds custom dictionaries,
learns field mappings, discovers validation rules, and generates training data
from production usage.
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class PatternExtractor:
    """Extract and learn patterns from successful extractions."""

    def __init__(self):
        self.patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pattern_confidence: Dict[str, float] = {}

    def extract_patterns(
        self,
        field_name: str,
        values: List[str],
        min_occurrences: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Extract regex patterns from successful field extractions.

        Args:
            field_name: Field name
            values: List of successfully extracted values
            min_occurrences: Minimum occurrences to consider pattern

        Returns:
            List of patterns with confidence scores
        """
        if not values:
            return []

        patterns = []

        # Extract common patterns
        # 1. Date patterns
        date_pattern = self._extract_date_pattern(values)
        if date_pattern:
            patterns.append(date_pattern)

        # 2. Number patterns (amounts, IDs)
        number_pattern = self._extract_number_pattern(values)
        if number_pattern:
            patterns.append(number_pattern)

        # 3. Code patterns (e.g., PO numbers, supplier codes)
        code_pattern = self._extract_code_pattern(values)
        if code_pattern:
            patterns.append(code_pattern)

        # 4. Email patterns
        if '@' in ' '.join(values):
            patterns.append({
                "type": "email",
                "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "confidence": 0.9,
            })

        # 5. Phone patterns
        phone_pattern = self._extract_phone_pattern(values)
        if phone_pattern:
            patterns.append(phone_pattern)

        # Store patterns
        self.patterns[field_name].extend(patterns)

        logger.info(f"Extracted {len(patterns)} patterns for {field_name}")
        return patterns

    def _extract_date_pattern(self, values: List[str]) -> Optional[Dict[str, Any]]:
        """Extract date pattern."""
        date_patterns = [
            (r"\d{4}-\d{2}-\d{2}", "YYYY-MM-DD"),
            (r"\d{2}/\d{2}/\d{4}", "MM/DD/YYYY"),
            (r"\d{2}\.\d{2}\.\d{4}", "DD.MM.YYYY"),
            (r"\d{2}-[A-Z]{3}-\d{4}", "DD-MMM-YYYY"),
        ]

        matches = Counter()

        for pattern, format_name in date_patterns:
            count = sum(1 for v in values if re.search(pattern, v))
            if count > 0:
                matches[(pattern, format_name)] = count

        if matches:
            (pattern, format_name), count = matches.most_common(1)[0]
            confidence = count / len(values)

            if confidence > 0.5:
                return {
                    "type": "date",
                    "pattern": pattern,
                    "format": format_name,
                    "confidence": confidence,
                }

        return None

    def _extract_number_pattern(self, values: List[str]) -> Optional[Dict[str, Any]]:
        """Extract number/amount pattern."""
        # Check if values are mostly numeric
        numeric_count = 0
        has_decimals = 0
        has_currency = 0

        for value in values:
            # Remove common currency symbols and separators
            cleaned = re.sub(r'[$€£¥,\s]', '', value)

            if re.match(r'^-?\d+\.?\d*$', cleaned):
                numeric_count += 1

                if '.' in cleaned:
                    has_decimals += 1

            if re.search(r'[$€£¥]', value):
                has_currency += 1

        if numeric_count / len(values) > 0.7:
            # Mostly numeric
            pattern_parts = []

            if has_currency / len(values) > 0.5:
                pattern_parts.append(r'[$€£¥]?')

            pattern_parts.append(r'-?\d{1,3}(?:,?\d{3})*')

            if has_decimals / len(values) > 0.5:
                pattern_parts.append(r'\.\d{2}')

            return {
                "type": "number",
                "pattern": ''.join(pattern_parts),
                "has_decimals": has_decimals > 0,
                "has_currency": has_currency > 0,
                "confidence": numeric_count / len(values),
            }

        return None

    def _extract_code_pattern(self, values: List[str]) -> Optional[Dict[str, Any]]:
        """Extract code pattern (e.g., PO numbers, SKUs)."""
        # Look for alphanumeric patterns with consistent structure
        if not values:
            return None

        # Analyze structure
        structures = []
        for value in values:
            structure = self._get_structure(value)
            structures.append(structure)

        # Find most common structure
        structure_counts = Counter(structures)
        if not structure_counts:
            return None

        common_structure, count = structure_counts.most_common(1)[0]
        confidence = count / len(values)

        if confidence > 0.6:
            # Convert structure to regex
            pattern = self._structure_to_regex(common_structure)

            return {
                "type": "code",
                "pattern": pattern,
                "structure": common_structure,
                "confidence": confidence,
            }

        return None

    def _extract_phone_pattern(self, values: List[str]) -> Optional[Dict[str, Any]]:
        """Extract phone number pattern."""
        phone_patterns = [
            r"\+\d{1,3}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{4}",  # International
            r"\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}",  # (123) 456-7890
            r"\d{3}[\s-]?\d{3}[\s-]?\d{4}",  # 123-456-7890
        ]

        matches = Counter()

        for pattern in phone_patterns:
            count = sum(1 for v in values if re.search(pattern, v))
            if count > 0:
                matches[pattern] = count

        if matches:
            pattern, count = matches.most_common(1)[0]
            confidence = count / len(values)

            if confidence > 0.5:
                return {
                    "type": "phone",
                    "pattern": pattern,
                    "confidence": confidence,
                }

        return None

    def _get_structure(self, value: str) -> str:
        """Get structure representation of value (e.g., 'AAAA-9999' -> 'A4-D4')."""
        structure = []
        prev_type = None
        count = 0

        for char in value:
            if char.isalpha():
                char_type = 'A'
            elif char.isdigit():
                char_type = 'D'
            else:
                char_type = char  # Keep special chars as-is

            if char_type == prev_type and char_type in ['A', 'D']:
                count += 1
            else:
                if prev_type:
                    if prev_type in ['A', 'D']:
                        structure.append(f"{prev_type}{count}")
                    else:
                        structure.append(prev_type)
                prev_type = char_type
                count = 1

        # Append last
        if prev_type:
            if prev_type in ['A', 'D']:
                structure.append(f"{prev_type}{count}")
            else:
                structure.append(prev_type)

        return ''.join(structure)

    def _structure_to_regex(self, structure: str) -> str:
        """Convert structure to regex pattern."""
        pattern = []
        i = 0

        while i < len(structure):
            if structure[i] == 'A':
                # Letters
                i += 1
                count = ''
                while i < len(structure) and structure[i].isdigit():
                    count += structure[i]
                    i += 1
                pattern.append(f"[A-Za-z]{{{count}}}")
            elif structure[i] == 'D':
                # Digits
                i += 1
                count = ''
                while i < len(structure) and structure[i].isdigit():
                    count += structure[i]
                    i += 1
                pattern.append(f"\\d{{{count}}}")
            else:
                # Special character
                pattern.append(re.escape(structure[i]))
                i += 1

        return ''.join(pattern)


class DictionaryBuilder:
    """Build custom dictionaries from processed documents."""

    def __init__(self):
        self.dictionaries: Dict[str, Set[str]] = defaultdict(set)
        self.term_frequencies: Dict[str, Counter] = defaultdict(Counter)
        self.co_occurrences: Dict[str, Counter] = defaultdict(Counter)

    def add_terms(
        self,
        category: str,
        terms: List[str],
        source_doc_id: Optional[str] = None,
    ):
        """
        Add terms to dictionary category.

        Args:
            category: Dictionary category (e.g., 'supplier_names', 'product_codes')
            terms: List of terms to add
            source_doc_id: Optional source document ID
        """
        for term in terms:
            if term and len(term) > 1:  # Filter very short terms
                self.dictionaries[category].add(term)
                self.term_frequencies[category][term] += 1

        logger.debug(f"Added {len(terms)} terms to {category} dictionary")

    def build_supplier_dictionary(
        self,
        supplier_names: List[str],
        supplier_ids: List[str],
    ) -> Dict[str, List[str]]:
        """Build supplier name/ID dictionary."""
        dictionary = {}

        for name, supplier_id in zip(supplier_names, supplier_ids):
            if supplier_id not in dictionary:
                dictionary[supplier_id] = []
            if name and name not in dictionary[supplier_id]:
                dictionary[supplier_id].append(name)

        self.dictionaries['suppliers'] = set(supplier_ids)
        self.dictionaries['supplier_names'] = set(supplier_names)

        logger.info(f"Built supplier dictionary: {len(dictionary)} suppliers")
        return dictionary

    def build_product_dictionary(
        self,
        product_descriptions: List[str],
        product_codes: List[str],
    ) -> Dict[str, str]:
        """Build product description/code dictionary."""
        dictionary = {}

        for desc, code in zip(product_descriptions, product_codes):
            if code and desc:
                dictionary[code] = desc

        self.dictionaries['product_codes'] = set(product_codes)

        logger.info(f"Built product dictionary: {len(dictionary)} products")
        return dictionary

    def get_common_terms(
        self,
        category: str,
        min_frequency: int = 5,
        limit: int = 100,
    ) -> List[Tuple[str, int]]:
        """Get most common terms in category."""
        if category not in self.term_frequencies:
            return []

        common = self.term_frequencies[category].most_common(limit)
        return [(term, count) for term, count in common if count >= min_frequency]

    def find_synonyms(
        self,
        category: str,
        term: str,
        min_co_occurrence: int = 3,
    ) -> List[str]:
        """Find potential synonyms based on co-occurrence."""
        if category not in self.co_occurrences:
            return []

        co_occur = self.co_occurrences[category]
        similar = [
            t for t, count in co_occur.most_common()
            if count >= min_co_occurrence and t != term
        ]

        return similar[:10]


class FieldMappingLearner:
    """Learn field mappings automatically from extractions."""

    def __init__(self):
        self.field_mappings: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.mapping_confidence: Dict[str, float] = {}
        self.source_target_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def learn_mapping(
        self,
        doc_type: str,
        source_field: str,
        target_field: str,
        examples: List[Tuple[Any, Any]],
    ):
        """
        Learn mapping from source to target field.

        Args:
            doc_type: Document type
            source_field: Source field name
            target_field: Target field name (e.g., SAP field)
            examples: List of (source_value, target_value) tuples
        """
        mapping_key = f"{doc_type}:{source_field}:{target_field}"

        # Store examples
        self.source_target_pairs[mapping_key].extend(examples)

        # Analyze mapping type
        mapping_type = self._analyze_mapping_type(examples)

        # Calculate confidence
        confidence = self._calculate_mapping_confidence(examples)

        self.field_mappings[doc_type][f"{source_field}->{target_field}"] = {
            "source": source_field,
            "target": target_field,
            "type": mapping_type,
            "confidence": confidence,
            "examples": examples[:10],  # Store first 10 examples
        }

        self.mapping_confidence[mapping_key] = confidence

        logger.info(
            f"Learned mapping: {source_field} -> {target_field} "
            f"({mapping_type}, confidence: {confidence:.2f})"
        )

    def _analyze_mapping_type(
        self,
        examples: List[Tuple[Any, Any]],
    ) -> str:
        """Analyze type of mapping."""
        if not examples:
            return "unknown"

        # Check if direct mapping (1:1)
        source_values = [ex[0] for ex in examples]
        target_values = [ex[1] for ex in examples]

        unique_sources = len(set(source_values))
        unique_targets = len(set(target_values))

        if unique_sources == unique_targets == len(examples):
            return "direct_1to1"

        # Check if transformation needed
        if all(str(s).upper() == str(t) for s, t in examples):
            return "uppercase"
        elif all(str(s).lower() == str(t) for s, t in examples):
            return "lowercase"

        # Check if lookup/dictionary
        if unique_sources > unique_targets:
            return "many_to_one"
        elif unique_sources < unique_targets:
            return "one_to_many"

        return "complex"

    def _calculate_mapping_confidence(
        self,
        examples: List[Tuple[Any, Any]],
    ) -> float:
        """Calculate confidence in mapping."""
        if not examples:
            return 0.0

        # Simple confidence based on consistency
        # More examples = higher confidence
        confidence = min(len(examples) / 100.0, 0.9)

        return confidence

    def get_mapping(
        self,
        doc_type: str,
        source_field: str,
        target_field: str,
    ) -> Optional[Dict[str, Any]]:
        """Get learned mapping."""
        if doc_type not in self.field_mappings:
            return None

        mapping_key = f"{source_field}->{target_field}"
        return self.field_mappings[doc_type].get(mapping_key)


class ValidationRuleLearner:
    """Discover validation rules from data."""

    def __init__(self):
        self.rules: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def discover_rules(
        self,
        field_name: str,
        values: List[Any],
        valid_flags: List[bool],
    ) -> List[Dict[str, Any]]:
        """
        Discover validation rules from data.

        Args:
            field_name: Field name
            values: Field values
            valid_flags: Whether each value is valid

        Returns:
            List of discovered rules
        """
        discovered_rules = []

        # Separate valid and invalid
        valid_values = [v for v, flag in zip(values, valid_flags) if flag]
        invalid_values = [v for v, flag in zip(values, valid_flags) if not flag]

        # 1. Range rules (for numeric fields)
        range_rule = self._discover_range_rule(field_name, valid_values, invalid_values)
        if range_rule:
            discovered_rules.append(range_rule)

        # 2. Length rules
        length_rule = self._discover_length_rule(field_name, valid_values, invalid_values)
        if length_rule:
            discovered_rules.append(length_rule)

        # 3. Format rules
        format_rule = self._discover_format_rule(field_name, valid_values, invalid_values)
        if format_rule:
            discovered_rules.append(format_rule)

        # 4. Allowed values (for categorical fields)
        allowed_rule = self._discover_allowed_values(field_name, valid_values, invalid_values)
        if allowed_rule:
            discovered_rules.append(allowed_rule)

        # Store rules
        self.rules[field_name].extend(discovered_rules)

        logger.info(f"Discovered {len(discovered_rules)} rules for {field_name}")
        return discovered_rules

    def _discover_range_rule(
        self,
        field_name: str,
        valid_values: List[Any],
        invalid_values: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Discover numeric range rule."""
        # Try to convert to numbers
        valid_nums = []
        for v in valid_values:
            try:
                valid_nums.append(float(v))
            except (ValueError, TypeError):
                pass

        if len(valid_nums) < len(valid_values) * 0.8:
            # Not mostly numeric
            return None

        # Calculate range
        min_val = min(valid_nums)
        max_val = max(valid_nums)

        # Add some margin
        margin = (max_val - min_val) * 0.1
        min_val -= margin
        max_val += margin

        return {
            "type": "range",
            "field": field_name,
            "min": min_val,
            "max": max_val,
            "confidence": 0.8,
        }

    def _discover_length_rule(
        self,
        field_name: str,
        valid_values: List[Any],
        invalid_values: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Discover length rule."""
        valid_lengths = [len(str(v)) for v in valid_values]

        if not valid_lengths:
            return None

        min_len = min(valid_lengths)
        max_len = max(valid_lengths)

        # Check if consistent length
        if min_len == max_len:
            return {
                "type": "exact_length",
                "field": field_name,
                "length": min_len,
                "confidence": 0.9,
            }
        else:
            return {
                "type": "length_range",
                "field": field_name,
                "min_length": min_len,
                "max_length": max_len,
                "confidence": 0.7,
            }

    def _discover_format_rule(
        self,
        field_name: str,
        valid_values: List[Any],
        invalid_values: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Discover format rule using pattern extraction."""
        # Use pattern extractor
        extractor = PatternExtractor()
        patterns = extractor.extract_patterns(
            field_name,
            [str(v) for v in valid_values],
        )

        if patterns:
            # Use most confident pattern
            best_pattern = max(patterns, key=lambda p: p.get('confidence', 0))

            return {
                "type": "format",
                "field": field_name,
                "pattern": best_pattern.get('pattern'),
                "confidence": best_pattern.get('confidence', 0.7),
            }

        return None

    def _discover_allowed_values(
        self,
        field_name: str,
        valid_values: List[Any],
        invalid_values: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Discover allowed values for categorical fields."""
        unique_valid = set(valid_values)

        # Only create rule if limited set of values
        if len(unique_valid) <= 20:
            return {
                "type": "allowed_values",
                "field": field_name,
                "allowed": list(unique_valid),
                "confidence": 0.85,
            }

        return None


class TrainingDataGenerator:
    """Generate training data from production usage."""

    def __init__(self, pmg: ProcessMemoryGraph):
        self.pmg = pmg

    def generate_from_production(
        self,
        doc_type: str,
        days: int = 30,
        min_confidence: float = 0.9,
        max_samples: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Generate training data from production usage.

        Args:
            doc_type: Document type
            days: Days of history to use
            min_confidence: Minimum confidence threshold
            max_samples: Maximum samples to generate

        Returns:
            List of training samples
        """
        logger.info(
            f"Generating training data for {doc_type} "
            f"(last {days} days, min_conf={min_confidence})"
        )

        # Query PMG for high-confidence successful transactions
        samples = self.pmg.find_similar_documents(
            doc_type=doc_type,
            limit=max_samples,
        )

        # Filter by confidence and success
        training_data = []

        for sample in samples:
            confidence = sample.get('confidence', 0.0)

            if confidence >= min_confidence:
                # Convert to training format
                training_sample = {
                    "features": sample,
                    "label": sample.get('doc_type', doc_type),
                    "confidence": confidence,
                    "source": "production",
                }
                training_data.append(training_sample)

        logger.info(f"Generated {len(training_data)} training samples from production")
        return training_data

    def generate_synthetic_data(
        self,
        doc_type: str,
        base_samples: List[Dict[str, Any]],
        num_synthetic: int = 1000,
        noise_level: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic data for data augmentation.

        Args:
            doc_type: Document type
            base_samples: Base samples to augment
            num_synthetic: Number of synthetic samples
            noise_level: Amount of noise to add (0-1)

        Returns:
            Synthetic training samples
        """
        if not base_samples:
            return []

        synthetic_data = []

        for _ in range(num_synthetic):
            # Select random base sample
            base = base_samples[np.random.randint(len(base_samples))]

            # Create synthetic variant
            synthetic = self._create_variant(base, noise_level)
            synthetic_data.append(synthetic)

        logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data

    def _create_variant(
        self,
        base_sample: Dict[str, Any],
        noise_level: float,
    ) -> Dict[str, Any]:
        """Create synthetic variant of base sample."""
        variant = base_sample.copy()

        # Add noise to numeric fields
        for key, value in variant.items():
            if isinstance(value, (int, float)):
                # Add random noise
                noise = np.random.normal(0, noise_level * abs(value))
                variant[key] = value + noise

            elif isinstance(value, str):
                # Slight string variations
                if np.random.random() < noise_level:
                    # Randomly change case
                    if np.random.random() < 0.5:
                        variant[key] = value.upper()
                    else:
                        variant[key] = value.lower()

        return variant


class KnowledgeAugmentationEngine:
    """
    Comprehensive knowledge augmentation engine.

    Combines pattern extraction, dictionary building, field mapping learning,
    validation rule discovery, and training data generation.
    """

    def __init__(self, pmg: ProcessMemoryGraph):
        self.pmg = pmg
        self.pattern_extractor = PatternExtractor()
        self.dictionary_builder = DictionaryBuilder()
        self.field_mapping_learner = FieldMappingLearner()
        self.validation_rule_learner = ValidationRuleLearner()
        self.training_data_generator = TrainingDataGenerator(pmg)

        logger.info("KnowledgeAugmentationEngine initialized")

    def augment_from_successful_extractions(
        self,
        doc_type: str,
        days: int = 7,
        min_confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Augment knowledge from successful extractions.

        Args:
            doc_type: Document type
            days: Days to look back
            min_confidence: Minimum confidence threshold

        Returns:
            Augmentation statistics
        """
        logger.info(f"Augmenting knowledge for {doc_type} (last {days} days)")

        # Get successful extractions from PMG
        successful = self.pmg.find_similar_documents(
            doc_type=doc_type,
            limit=10000,
        )

        # Filter by confidence
        high_confidence = [
            doc for doc in successful
            if doc.get('confidence', 0) >= min_confidence
        ]

        logger.info(f"Found {len(high_confidence)} high-confidence extractions")

        stats = {
            "doc_type": doc_type,
            "samples_analyzed": len(high_confidence),
            "patterns_extracted": 0,
            "dictionary_terms_added": 0,
            "mappings_learned": 0,
            "rules_discovered": 0,
        }

        # Extract patterns for each field
        field_values = defaultdict(list)
        for doc in high_confidence:
            for field, value in doc.items():
                if isinstance(value, str) and value:
                    field_values[field].append(value)

        for field, values in field_values.items():
            patterns = self.pattern_extractor.extract_patterns(field, values)
            stats["patterns_extracted"] += len(patterns)

            # Add to dictionary
            self.dictionary_builder.add_terms(f"{doc_type}:{field}", values)
            stats["dictionary_terms_added"] += len(values)

        logger.info(f"Knowledge augmentation complete: {stats}")
        return stats

    def build_knowledge_base(
        self,
        doc_types: List[str],
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Build comprehensive knowledge base from production data.

        Args:
            doc_types: List of document types
            days: Days of history

        Returns:
            Knowledge base statistics
        """
        logger.info(f"Building knowledge base for {len(doc_types)} document types")

        stats = {
            "doc_types_processed": 0,
            "total_patterns": 0,
            "total_dictionary_terms": 0,
            "total_mappings": 0,
            "total_rules": 0,
        }

        for doc_type in doc_types:
            result = self.augment_from_successful_extractions(doc_type, days)
            stats["doc_types_processed"] += 1
            stats["total_patterns"] += result["patterns_extracted"]
            stats["total_dictionary_terms"] += result["dictionary_terms_added"]
            stats["total_mappings"] += result["mappings_learned"]
            stats["total_rules"] += result["rules_discovered"]

        logger.info(f"Knowledge base built: {stats}")
        return stats

    def get_patterns(self, field_name: str) -> List[Dict[str, Any]]:
        """Get learned patterns for field."""
        return self.pattern_extractor.patterns.get(field_name, [])

    def get_dictionary(self, category: str) -> Set[str]:
        """Get dictionary for category."""
        return self.dictionary_builder.dictionaries.get(category, set())

    def get_field_mapping(
        self,
        doc_type: str,
        source_field: str,
        target_field: str,
    ) -> Optional[Dict[str, Any]]:
        """Get learned field mapping."""
        return self.field_mapping_learner.get_mapping(
            doc_type, source_field, target_field
        )

    def get_validation_rules(self, field_name: str) -> List[Dict[str, Any]]:
        """Get validation rules for field."""
        return self.validation_rule_learner.rules.get(field_name, [])
