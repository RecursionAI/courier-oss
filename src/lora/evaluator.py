import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import Counter
import difflib
from datetime import datetime

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvaluationIssue:
    issue_type: str
    severity: SeverityLevel
    description: str
    details: Optional[List[str]] = None
    confidence: float = 1.0


@dataclass
class EvaluationCriteria:
    name: str
    weight: float
    enabled: bool = True
    manual_score: Optional[float] = None


@dataclass
class EvaluationResult:
    overall_score: float
    criteria_scores: Dict[str, float]
    issues: List[EvaluationIssue]
    metadata: Dict[str, Any]
    timestamp: str


class AIOutputEvaluator:
    """
    Comprehensive AI output evaluation system for comparing generated text
    against reference/control outputs.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Default evaluation criteria
        self.default_criteria = {
            "accuracy": EvaluationCriteria("accuracy", 0.30),
            # "relevance": EvaluationCriteria("relevance", 0.25),
            "completeness": EvaluationCriteria("completeness", 0.20),
            "clarity": EvaluationCriteria("clarity", 0.15),
            # "coherence": EvaluationCriteria("coherence", 0.10)
        }

        # Initialize NLP components if available
        self._init_nlp_components()

    def _init_nlp_components(self):
        """Initialize NLP components based on available libraries."""
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                self.logger.warning("Failed to initialize NLTK components")

        if TRANSFORMERS_AVAILABLE and self.config.get('use_embeddings', False):
            try:
                model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                self.embedding_model = pipeline('feature-extraction', model=model_name)
            except:
                self.logger.warning("Failed to initialize embedding model")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)

        return text

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract various features from text for analysis."""
        text = self.preprocess_text(text)

        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'unique_word_ratio': len(set(text.lower().split())) / len(text.split()) if text.split() else 0,
        }

        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
                features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0

                words = word_tokenize(text.lower())
                features['stopword_ratio'] = len([w for w in words if w in self.stop_words]) / len(
                    words) if words else 0
            except:
                pass

        return features

    def calculate_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate various similarity metrics between two texts."""
        similarities = {}

        # Basic similarity metrics
        similarities['jaccard'] = self._jaccard_similarity(text1, text2)
        similarities['overlap_coefficient'] = self._overlap_coefficient(text1, text2)
        similarities['length_similarity'] = self._length_similarity(text1, text2)
        similarities['structural_similarity'] = self._structural_similarity(text1, text2)

        # Advanced similarity metrics
        if SKLEARN_AVAILABLE:
            similarities['tfidf_cosine'] = self._tfidf_cosine_similarity(text1, text2)

        if NLTK_AVAILABLE:
            similarities['semantic_overlap'] = self._semantic_overlap(text1, text2)

        return similarities

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _overlap_coefficient(self, text1: str, text2: str) -> float:
        """Calculate overlap coefficient between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        min_set = min(len(words1), len(words2))

        return intersection / min_set if min_set > 0 else 0.0

    def _length_similarity(self, text1: str, text2: str) -> float:
        """Calculate length-based similarity."""
        len1, len2 = len(text1.split()), len(text2.split())
        return min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

    def _structural_similarity(self, text1: str, text2: str) -> float:
        """Calculate structural similarity based on sentence patterns."""
        sentences1 = len(re.split(r'[.!?]+', text1.strip()))
        sentences2 = len(re.split(r'[.!?]+', text2.strip()))

        return min(sentences1, sentences2) / max(sentences1, sentences2) if max(sentences1, sentences2) > 0 else 0.0

    def _tfidf_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity."""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0.0

    def _semantic_overlap(self, text1: str, text2: str) -> float:
        """Calculate semantic overlap using lemmatization."""
        try:
            words1 = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text1)
                      if word.lower() not in self.stop_words and word.isalpha()]
            words2 = [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text2)
                      if word.lower() not in self.stop_words and word.isalpha()]

            set1, set2 = set(words1), set(words2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))

            return intersection / union if union > 0 else 0.0
        except:
            return 0.0

    def detect_issues(self, generated: str, control: str) -> List[EvaluationIssue]:
        """Detect potential issues in generated text compared to control."""
        issues = []

        # Length discrepancy check
        gen_words = len(generated.split())
        ctrl_words = len(control.split())
        length_ratio = gen_words / ctrl_words if ctrl_words > 0 else float('inf')

        if length_ratio > 2.0:
            issues.append(EvaluationIssue(
                issue_type="excessive_length",
                severity=SeverityLevel.MEDIUM,
                description=f"Generated text is {length_ratio:.1f}x longer than control",
                confidence=0.9
            ))
        elif length_ratio < 0.5:
            issues.append(EvaluationIssue(
                issue_type="insufficient_length",
                severity=SeverityLevel.MEDIUM,
                description=f"Generated text is {1 / length_ratio:.1f}x shorter than control",
                confidence=0.9
            ))

        # Contradiction detection
        contradictions = self._detect_contradictions(generated, control)
        if contradictions:
            issues.append(EvaluationIssue(
                issue_type="contradictions",
                severity=SeverityLevel.HIGH,
                description=f"Found {len(contradictions)} potential contradictions",
                details=contradictions[:3],  # Show first 3
                confidence=0.7
            ))

        # Missing key information
        missing_info = self._detect_missing_information(generated, control)
        if missing_info:
            issues.append(EvaluationIssue(
                issue_type="missing_information",
                severity=SeverityLevel.MEDIUM,
                description=f"Missing {len(missing_info)} key information pieces",
                details=missing_info[:3],
                confidence=0.8
            ))

        # Repetition detection
        repetitions = self._detect_repetitions(generated)
        if repetitions:
            issues.append(EvaluationIssue(
                issue_type="repetition",
                severity=SeverityLevel.LOW,
                description=f"Found {len(repetitions)} repetitive patterns",
                details=repetitions[:3],
                confidence=0.6
            ))

        return issues

    def _detect_contradictions(self, generated: str, control: str) -> List[str]:
        """Detect potential contradictions between texts."""
        contradictions = []

        # Simple negation-based contradiction detection
        gen_sentences = re.split(r'[.!?]+', generated.lower())
        ctrl_sentences = re.split(r'[.!?]+', control.lower())

        negation_patterns = [r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bcannot\b', r'\bwon\'t\b']

        for gen_sent in gen_sentences:
            for ctrl_sent in ctrl_sentences:
                similarity = difflib.SequenceMatcher(None, gen_sent.strip(), ctrl_sent.strip()).ratio()
                if similarity > 0.6:  # Similar sentences
                    gen_negations = sum(1 for pattern in negation_patterns if re.search(pattern, gen_sent))
                    ctrl_negations = sum(1 for pattern in negation_patterns if re.search(pattern, ctrl_sent))

                    if abs(gen_negations - ctrl_negations) > 0:
                        contradictions.append(f"Generated: {gen_sent.strip()[:100]}...")

        return contradictions

    def _detect_missing_information(self, generated: str, control: str) -> List[str]:
        """Detect missing key information in generated text."""
        missing = []

        # Extract key phrases from control text
        key_phrase_patterns = [
            r'\b(?:is|are|was|were|will be|must be|should be)\s+\w+\b',
            r'\b\d+(?:\.\d+)?\s*(?:%|percent|dollars?|years?|months?|days?)\b',
            r'\b(?:always|never|often|sometimes|rarely)\b',
        ]

        for pattern in key_phrase_patterns:
            ctrl_matches = re.findall(pattern, control, re.IGNORECASE)
            for match in ctrl_matches:
                if match.lower() not in generated.lower():
                    missing.append(match)

        return list(set(missing))  # Remove duplicates

    def _detect_repetitions(self, text: str) -> List[str]:
        """Detect repetitive patterns in text."""
        repetitions = []
        sentences = re.split(r'[.!?]+', text)

        # Check for repeated sentences
        sentence_counts = Counter(sent.strip().lower() for sent in sentences if sent.strip())
        for sentence, count in sentence_counts.items():
            if count > 1 and len(sentence.split()) > 3:
                repetitions.append(f"Repeated {count} times: {sentence[:50]}...")

        # Check for repeated phrases
        words = text.split()
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i + 3])
            remaining_text = ' '.join(words[i + 3:])
            if phrase.lower() in remaining_text.lower():
                repetitions.append(f"Repeated phrase: {phrase}")

        return repetitions

    def calculate_scores(self, generated: str, control: str,
                         criteria: Optional[Dict[str, EvaluationCriteria]] = None) -> Dict[str, float]:
        """Calculate scores for each evaluation criterion."""
        if criteria is None:
            criteria = self.default_criteria

        scores = {}
        similarities = self.calculate_text_similarity(generated, control)
        gen_features = self.extract_features(generated)
        ctrl_features = self.extract_features(control)

        for criterion_name, criterion in criteria.items():
            if not criterion.enabled:
                continue

            if criterion.manual_score is not None:
                scores[criterion_name] = criterion.manual_score
                continue

            # Calculate automated scores based on criterion type
            if criterion_name == "accuracy":
                scores[criterion_name] = self._calculate_accuracy_score(similarities, generated, control)
            elif criterion_name == "relevance":
                scores[criterion_name] = self._calculate_relevance_score(similarities)
            elif criterion_name == "completeness":
                scores[criterion_name] = self._calculate_completeness_score(gen_features, ctrl_features)
            elif criterion_name == "clarity":
                scores[criterion_name] = self._calculate_clarity_score(gen_features)
            elif criterion_name == "coherence":
                scores[criterion_name] = self._calculate_coherence_score(generated, similarities)
            else:
                # Default scoring for custom criteria
                scores[criterion_name] = similarities.get('jaccard', 0.0) * 100

        return scores

    def _calculate_accuracy_score(self, similarities: Dict[str, float], generated: str, control: str) -> float:
        """Calculate accuracy score based on content overlap and contradictions."""
        base_score = (similarities.get('tfidf_cosine', similarities.get('jaccard', 0))) * 100

        # Penalize for contradictions
        contradictions = self._detect_contradictions(generated, control)
        contradiction_penalty = min(len(contradictions) * 10, 30)  # Max 30 point penalty

        return max(0, base_score - contradiction_penalty)

    def _calculate_relevance_score(self, similarities: Dict[str, float]) -> float:
        """Calculate relevance score based on semantic similarity."""
        semantic_score = similarities.get('semantic_overlap', similarities.get('overlap_coefficient', 0))
        return semantic_score * 100

    def _calculate_completeness_score(self, gen_features: Dict, ctrl_features: Dict) -> float:
        """Calculate completeness score based on content coverage."""
        word_ratio = min(gen_features['word_count'] / ctrl_features['word_count'], 1.5) if ctrl_features[
                                                                                               'word_count'] > 0 else 0
        sentence_ratio = min(gen_features['sentence_count'] / ctrl_features['sentence_count'], 1.5) if ctrl_features[
                                                                                                           'sentence_count'] > 0 else 0

        # Penalize if too short, reward if comprehensive but not excessive
        completeness = (word_ratio + sentence_ratio) / 2
        return min(completeness * 100, 100)

    def _calculate_clarity_score(self, features: Dict) -> float:
        """Calculate clarity score based on readability metrics."""
        # Simple readability approximation
        avg_word_length = features['avg_word_length']
        avg_sentence_length = features.get('avg_sentence_length', 15)

        # Penalize overly complex text
        word_penalty = max(0, (avg_word_length - 6) * 5)  # Penalize words longer than 6 chars
        sentence_penalty = max(0, (avg_sentence_length - 20) * 2)  # Penalize sentences longer than 20 words

        base_score = 100
        return max(0, base_score - word_penalty - sentence_penalty)

    def _calculate_coherence_score(self, text: str, similarities: Dict) -> float:
        """Calculate coherence score based on structure and flow."""
        # Check for structural coherence
        sentences = re.split(r'[.!?]+', text.strip())
        if len(sentences) < 2:
            return 80  # Short texts are generally coherent

        # Check for transition words and logical flow
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally',
                            'consequently', 'meanwhile', 'subsequently', 'nevertheless']

        transition_count = sum(1 for word in transition_words if word in text.lower())
        transition_score = min(transition_count * 10, 30)  # Max 30 points from transitions

        # Base coherence from structural similarity
        structure_score = similarities.get('structural_similarity', 0.7) * 70

        return min(structure_score + transition_score, 100)

    def evaluate(self, generated: str, control: str,
                 criteria: Optional[Dict[str, EvaluationCriteria]] = None) -> EvaluationResult:
        """
        Main evaluation method that compares generated text against control text.

        Args:
            generated: The AI-generated text to evaluate
            control: The reference/control text to compare against
            criteria: Optional custom evaluation criteria

        Returns:
            EvaluationResult object containing scores, issues, and metadata
        """
        if criteria is None:
            criteria = self.default_criteria.copy()

        # Calculate individual criterion scores
        scores = self.calculate_scores(generated, control, criteria)

        # Calculate weighted overall score
        total_weight = sum(c.weight for c in criteria.values() if c.enabled)
        if total_weight > 0:
            overall_score = sum(scores[name] * criteria[name].weight
                                for name in scores if criteria[name].enabled) / total_weight
        else:
            overall_score = 0.0

        # Detect issues
        issues = self.detect_issues(generated, control)

        # Prepare metadata
        similarities = self.calculate_text_similarity(generated, control)
        gen_features = self.extract_features(generated)
        ctrl_features = self.extract_features(control)

        metadata = {
            'generated_features': gen_features,
            'control_features': ctrl_features,
            'similarities': similarities,
            'criteria_used': {name: asdict(crit) for name, crit in criteria.items()},
            'evaluation_config': self.config
        }

        return EvaluationResult(
            overall_score=round(overall_score, 2),
            criteria_scores=scores,
            issues=issues,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

    def batch_evaluate(self, pairs: List[Tuple[str, str]],
                       criteria: Optional[Dict[str, EvaluationCriteria]] = None) -> List[EvaluationResult]:
        """Evaluate multiple generated/control text pairs."""
        results = []
        for generated, control in pairs:
            try:
                result = self.evaluate(generated, control, criteria)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating pair: {e}")
                # Create error result
                results.append(EvaluationResult(
                    overall_score=0.0,
                    criteria_scores={},
                    issues=[EvaluationIssue(
                        issue_type="evaluation_error",
                        severity=SeverityLevel.CRITICAL,
                        description=str(e)
                    )],
                    metadata={'error': str(e)},
                    timestamp=datetime.now().isoformat()
                ))
        return results

    def export_results(self, results: List[EvaluationResult], format: str = 'json') -> str:
        """Export evaluation results in specified format."""
        if format.lower() == 'json':
            return json.dumps([asdict(result) for result in results], indent=2, default=str)
        elif format.lower() == 'csv':
            # Simple CSV export for scores
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)

            if results:
                # Header
                header = ['timestamp', 'overall_score'] + list(results[0].criteria_scores.keys()) + ['issue_count']
                writer.writerow(header)

                # Data rows
                for result in results:
                    row = [result.timestamp, result.overall_score]
                    row.extend([result.criteria_scores.get(crit, '') for crit in results[0].criteria_scores.keys()])
                    row.append(len(result.issues))
                    writer.writerow(row)

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = AIOutputEvaluator(config={
        'use_embeddings': False,  # Set to True if you have transformers installed
    })

    # Example texts
    generated_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. It has applications in many fields including healthcare, 
    finance, and autonomous vehicles. Deep learning is a particularly powerful approach.
    """

    control_text = """
    Machine learning is a branch of artificial intelligence that enables computers to 
    learn from data without explicit programming. It's widely used in healthcare, 
    finance, and self-driving cars. Neural networks are a key technology in this field.
    """

    # Run evaluation
    result = evaluator.evaluate(generated_text, control_text)

    # Print results
    print(f"Overall Score: {result.overall_score}/100")
    print("\nCriteria Scores:")
    for criterion, score in result.criteria_scores.items():
        print(f"  {criterion.capitalize()}: {score:.2f}")

    print(f"\nIssues Found: {len(result.issues)}")
    for issue in result.issues:
        print(f"  - {issue.issue_type}: {issue.description}")

    # Export results
    json_output = evaluator.export_results([result])
    print(f"\nJSON Export (first 200 chars): {json_output[:200]}...")