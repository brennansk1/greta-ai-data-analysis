"""
Comprehensive tests for narrative_generation module using pytest.
"""

import pytest
from greta_core.narrative_generation import (
    generate_hypothesis_narrative, generate_summary_narrative,
    generate_insight_narrative, create_report
)


class TestGenerateHypothesisNarrative:
    """Test hypothesis narrative generation functionality."""

    def test_generate_hypothesis_narrative_single_feature(self, sample_hypotheses, feature_names):
        """Test narrative generation for single feature hypothesis."""
        hypothesis = {
            'features': [0],
            'significance': 0.9,
            'effect_size': 0.7,
            'coverage': 0.8,
            'parsimony_penalty': 0.2,
            'fitness': 2.3
        }

        narrative = generate_hypothesis_narrative(hypothesis, feature_names)

        assert isinstance(narrative, str)
        assert 'age' in narrative  # First feature
        assert 'strong' in narrative  # effect_size > 0.5
        assert 'high' in narrative  # significance > 0.8

    def test_generate_hypothesis_narrative_multiple_features(self, sample_hypotheses, feature_names):
        """Test narrative generation for multiple feature hypothesis."""
        hypothesis = sample_hypotheses[0]  # Has features [0, 2]

        narrative = generate_hypothesis_narrative(hypothesis, feature_names)

        assert isinstance(narrative, str)
        assert 'age' in narrative
        assert 'education' in narrative
        assert 'very high' in narrative  # significance > 0.95
        assert 'strong' in narrative  # effect_size > 0.5

    def test_generate_hypothesis_narrative_weak_relationship(self):
        """Test narrative with weak statistical relationship."""
        hypothesis = {
            'features': [1],
            'significance': 0.6,
            'effect_size': 0.2,
            'coverage': 0.3,
            'parsimony_penalty': 0.5,
            'fitness': 0.6
        }
        feature_names = ['A', 'B', 'C']

        narrative = generate_hypothesis_narrative(hypothesis, feature_names)

        assert 'moderate' in narrative or 'low' in narrative
        assert 'weak' in narrative

    def test_generate_hypothesis_narrative_perfect_significance(self):
        """Test narrative with perfect significance."""
        hypothesis = {
            'features': [0],
            'significance': 1.0,
            'effect_size': 0.9,
            'coverage': 0.95,
            'parsimony_penalty': 0.1,
            'fitness': 2.85
        }
        feature_names = ['Perfect']

        narrative = generate_hypothesis_narrative(hypothesis, feature_names)

        assert 'very high' in narrative
        assert 'strong' in narrative

    def test_generate_hypothesis_narrative_edge_cases(self):
        """Test narrative generation edge cases."""
        # Empty features
        hypothesis = {
            'features': [],
            'significance': 0.5,
            'effect_size': 0.5,
            'coverage': 0.5,
            'parsimony_penalty': 0.5,
            'fitness': 0.0
        }
        feature_names = ['A', 'B']

        # Should handle empty features gracefully
        narrative = generate_hypothesis_narrative(hypothesis, feature_names)
        assert isinstance(narrative, str)


class TestGenerateSummaryNarrative:
    """Test summary narrative generation functionality."""

    def test_generate_summary_narrative_multiple_hypotheses(self, sample_hypotheses, feature_names):
        """Test summary generation with multiple hypotheses."""
        summary = generate_summary_narrative(sample_hypotheses, feature_names)

        assert isinstance(summary, str)
        assert 'hypotheses' in summary.lower()
        assert 'strongest' in summary.lower()

    def test_generate_summary_narrative_single_hypothesis(self, feature_names):
        """Test summary generation with single hypothesis."""
        hypotheses = [{
            'features': [0],
            'significance': 0.8,
            'effect_size': 0.6,
            'coverage': 0.7,
            'parsimony_penalty': 0.3,
            'fitness': 1.8
        }]

        summary = generate_summary_narrative(hypotheses, feature_names)

        assert isinstance(summary, str)
        assert '1' in summary  # Number of hypotheses
        assert 'top finding' in summary.lower()

    def test_generate_summary_narrative_empty_list(self, feature_names):
        """Test summary generation with no hypotheses."""
        summary = generate_summary_narrative([], feature_names)

        assert isinstance(summary, str)
        assert 'no significant hypotheses' in summary.lower()

    def test_generate_summary_narrative_average_stats(self, feature_names):
        """Test summary includes average statistics."""
        hypotheses = [
            {
                'features': [0],
                'significance': 0.9,
                'effect_size': 0.8,
                'coverage': 0.7,
                'parsimony_penalty': 0.2,
                'fitness': 2.2
            },
            {
                'features': [1],
                'significance': 0.7,
                'effect_size': 0.4,
                'coverage': 0.5,
                'parsimony_penalty': 0.4,
                'fitness': 1.2
            }
        ]

        summary = generate_summary_narrative(hypotheses, feature_names)

        assert isinstance(summary, str)
        # Should include some statistical summary
        assert len(summary) > 50  # Reasonable length


class TestGenerateInsightNarrative:
    """Test insight narrative generation functionality."""

    def test_generate_insight_narrative_significant(self):
        """Test insight generation for significant results."""
        stat_results = {'p_value': 0.01, 'r_squared': 0.75}
        hypothesis = {
            'effect_size': 0.6,
            'features': [0, 1],
            'significance': 0.9
        }

        insight = generate_insight_narrative(stat_results, hypothesis)

        assert isinstance(insight, str)
        assert 'significant' in insight.lower()
        assert '75%' in insight
        assert 'practical importance' in insight

    def test_generate_insight_narrative_not_significant(self):
        """Test insight generation for non-significant results."""
        stat_results = {'p_value': 0.15, 'r_squared': 0.3}
        hypothesis = {
            'effect_size': 0.2,
            'features': [0],
            'significance': 0.4
        }

        insight = generate_insight_narrative(stat_results, hypothesis)

        assert 'not statistically significant' in insight
        assert '30%' in insight

    def test_generate_insight_narrative_marginal(self):
        """Test insight generation for marginally significant results."""
        stat_results = {'p_value': 0.08, 'r_squared': 0.5}
        hypothesis = {
            'effect_size': 0.4,
            'features': [0],
            'significance': 0.6
        }

        insight = generate_insight_narrative(stat_results, hypothesis)

        assert 'marginally significant' in insight

    def test_generate_insight_narrative_strong_effect(self):
        """Test insight with strong effect size."""
        stat_results = {'p_value': 0.02, 'r_squared': 0.8}
        hypothesis = {
            'effect_size': 0.7,
            'features': [0],
            'significance': 0.8
        }

        insight = generate_insight_narrative(stat_results, hypothesis)

        assert 'effect size' in insight
        assert 'artifact' in insight


class TestCreateReport:
    """Test complete report creation functionality."""

    def test_create_report_basic(self, sample_hypotheses, feature_names):
        """Test basic report creation."""
        stat_results = [
            {'p_value': 0.01, 'r_squared': 0.8},
            {'p_value': 0.05, 'r_squared': 0.6}
        ]

        report = create_report(sample_hypotheses, feature_names, stat_results)

        assert isinstance(report, str)
        assert 'GRETA Analysis Report' in report
        assert 'Data Summary' in report
        assert 'Detailed Findings' in report
        assert 'Recommendations' in report

    def test_create_report_with_metadata(self, sample_hypotheses, feature_names):
        """Test report creation with metadata."""
        stat_results = [{'p_value': 0.02, 'r_squared': 0.7}]

        # Add metadata-like structure
        hypotheses_with_meta = sample_hypotheses[:1]

        report = create_report(hypotheses_with_meta, feature_names, stat_results)

        assert 'GRETA Analysis Report' in report
        assert len(report) > 200  # Substantial report

    def test_create_report_empty_inputs(self):
        """Test report creation with minimal inputs."""
        report = create_report([], [], [])

        assert isinstance(report, str)
        assert 'GRETA Analysis Report' in report

    def test_create_report_high_confidence(self, feature_names):
        """Test report with high confidence hypothesis."""
        hypotheses = [{
            'features': [0],
            'significance': 0.98,
            'effect_size': 0.8,
            'coverage': 0.9,
            'parsimony_penalty': 0.1,
            'fitness': 2.68
        }]
        stat_results = [{'p_value': 0.001, 'r_squared': 0.85}]

        report = create_report(hypotheses, feature_names, stat_results)

        assert 'highly reliable' in report
        assert 'prioritized' in report


# Integration tests
class TestNarrativeGenerationIntegration:
    """Integration tests for narrative generation."""

    def test_full_narrative_pipeline(self, sample_hypotheses, feature_names):
        """Test complete narrative generation pipeline."""
        # Generate individual narratives
        narratives = []
        for hyp in sample_hypotheses:
            narrative = generate_hypothesis_narrative(hyp, feature_names)
            narratives.append(narrative)
            assert isinstance(narrative, str)

        # Generate summary
        summary = generate_summary_narrative(sample_hypotheses, feature_names)
        assert isinstance(summary, str)

        # Generate insights
        stat_results = [{'p_value': 0.01, 'r_squared': 0.8}]
        insight = generate_insight_narrative(stat_results[0], sample_hypotheses[0])
        assert isinstance(insight, str)

        # Create full report
        report = create_report(sample_hypotheses[:1], feature_names, stat_results)
        assert isinstance(report, str)
        assert len(report) > len(summary)

    def test_narrative_consistency(self, sample_hypotheses, feature_names):
        """Test that narratives are consistent and well-formed."""
        for hyp in sample_hypotheses:
            narrative = generate_hypothesis_narrative(hyp, feature_names)

            # Should be proper sentences
            assert narrative.endswith('.')
            assert len(narrative) > 20  # Reasonable length

            # Should contain feature names
            feature_names_in_hyp = [feature_names[i] for i in hyp['features']]
            for fname in feature_names_in_hyp:
                assert fname in narrative