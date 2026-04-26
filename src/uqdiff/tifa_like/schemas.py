from pydantic import BaseModel


class TifaQuestion(BaseModel):
    """A single TIFA verification question about an image element."""

    element: str
    question: str
    choices: list[str]
    answer: str
    element_type: str


class TifaQuestionSet(BaseModel):
    """A set of TIFA questions generated from a caption."""

    questions: list[TifaQuestion]


class QuestionResult(BaseModel):
    """Result of evaluating a single question against an image."""

    element: str
    question: str
    choices: list[str]
    expected_answer: str
    element_type: str
    free_form_answer: str
    multiple_choice_answer: str
    score: int


class TifaResult(BaseModel):
    """TIFA evaluation result for a single image."""

    score: float
    question_results: list[QuestionResult]
