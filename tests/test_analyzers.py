from ticket_router.complexity import ComplexityAnalyzer
from ticket_router.keyword_categorizer import KeywordCategorizer
from ticket_router.queues import QueueAssigner
from ticket_router.schemas import Category
from ticket_router.sentiment import SentimentAnalyzer


class TestKeywordCategorizer:
    def test_billing(self):
        category, _ = KeywordCategorizer().categorize("Charged twice", "I need a refund for the duplicate payment")
        assert category == Category.BILLING

    def test_technical(self):
        category, _ = KeywordCategorizer().categorize("App crash", "The app crashes with an error on upload")
        assert category == Category.TECHNICAL

    def test_subject_outweighs_description(self):
        category, scores = KeywordCategorizer().categorize("Password reset", "it is about my order")
        assert category == Category.ACCOUNT
        assert scores[Category.ACCOUNT] > scores[Category.SHIPPING]

    def test_no_keywords_defaults_to_general(self):
        category, scores = KeywordCategorizer().categorize("Hmm", "zzz qqq")
        assert category == Category.GENERAL
        assert all(s == 0 for s in scores.values())


def test_queue_assignment():
    assigner = QueueAssigner()
    assert assigner.assign(Category.BILLING) == "Billing"
    assert assigner.assign(Category.TECHNICAL) == "Technical Support"


class TestSentiment:
    analyzer = SentimentAnalyzer()

    def test_positive(self):
        assert self.analyzer.analyze("Thanks so much, the team was really helpful!").label == "positive"

    def test_neutral(self):
        assert self.analyzer.analyze("How do I change my shipping address?").label == "neutral"

    def test_very_negative(self):
        result = self.analyzer.analyze("This is TERRIBLE and completely unacceptable!! Worst service ever.")
        assert result.label == "very_negative"
        assert result.score < -0.65

    def test_negation_flips_polarity(self):
        assert self.analyzer.analyze("I am happy").score > 0
        assert self.analyzer.analyze("I am not happy").score < 0

    def test_score_is_bounded(self):
        result = self.analyzer.analyze("awful " * 50)
        assert -1 <= result.score <= 1


class TestComplexity:
    analyzer = ComplexityAnalyzer()

    def test_simple_ticket_is_low(self):
        assert self.analyzer.analyze("Business hours", "What are your hours?").level == "low"

    def test_complex_ticket_is_high(self):
        result = self.analyzer.analyze(
            "API integration broken",
            "Our API integration fails with a 502 error and webhook timeout on the server. The logs show "
            "an SSL certificate exception. I already tried restarting and reinstalled the SDK, this is the "
            "third time I contacted you. Also, my invoice has a wrong charge and I can't log in to my account.",
        )
        assert result.level == "high"
        assert result.factors["technical_depth"] == 1.0
