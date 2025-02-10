import os
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

from utils.ml_logging import get_logger

logger = get_logger()


class PromptManager:
    def __init__(self, template_dir: str = "templates"):
        """
        Initialize the PromptManager with the given template directory.

        Args:
            template_dir (str): The directory containing the Jinja2 templates.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, template_dir)

        self.env = Environment(
            loader=FileSystemLoader(searchpath=template_path), autoescape=False
        )

        templates = self.env.list_templates()
        print(f"Templates found: {templates}")

    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        Render a template with the given context.

        Args:
            template_name (str): The name of the template file.
            **kwargs: The context variables to render the template with.

        Returns:
            str: The rendered template as a string.
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            raise ValueError(f"Error rendering template '{template_name}': {e}")

    def create_prompt_query_classifier_user(self, query: str) -> str:
        """
        Create a user prompt for query classification.

        Args:
            query (str): The user query, e.g. "What is the process for prior authorization for Humira?"

        Returns:
            str: The rendered prompt (query_classifier_user_prompt.jinja) with instructions
                 on classifying the query as 'keyword' or 'semantic'.
        """
        return self.get_prompt(
            "query_classificator_user_prompt.jinja",
            query=query,
        )

    def create_prompt_formulator_user(
        self,
        diagnosis: str,
        medication_or_procedure: str,
        code: str,
        dosage: str,
        duration: str,
        rationale: str,
    ) -> str:
        """
        Create a user prompt for query formulation (using query expansion).

        Args:
            diagnosis (str): Diagnosis / medical justification details.
            medication_or_procedure (str): Name of medication or procedure.
            code (str): Relevant code (ICD-10, CPT, etc.).
            dosage (str): Dosage or plan details.
            duration (str): Duration of treatment.
            rationale (str): Clinical rationale or justification.

        Returns:
            str: The rendered prompt (formulator_user_prompt.jinja) that guides how to
                 construct an optimized search query with synonyms, related terms, etc.
        """
        return self.get_prompt(
            "formulator_user_prompt.jinja",
            diagnosis=diagnosis,
            medication_or_procedure=medication_or_procedure,
            code=code,
            dosage=dosage,
            duration=duration,
            rationale=rationale,
        )

    def create_prompt_evaluator_user(
        self, query: str, search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Create a user prompt for evaluating policy search results.

        Args:
            query (str): The user's query regarding prior authorization (e.g. "What is
                         the prior authorization policy for Epidiolex for LGS?")
            search_results (List[Dict[str, Any]]): A list of retrieved policies, each containing:
                - 'id': Unique identifier
                - 'path': URL or file path
                - 'content': Extracted policy text
                - 'caption': Summary or short description

        Returns:
            str: The rendered prompt (evaluator_user_prompt.jinja) instructing how to
                 evaluate these policies against the query, deduplicate, and form
                 a final JSON-like response.
        """
        return self.get_prompt(
            "evaluator_user_prompt.jinja",
            query=query,
            SearchResults=search_results,
        )
