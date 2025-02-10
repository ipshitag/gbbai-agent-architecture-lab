import logging
import os
from typing import List, Tuple

import requests
from semantic_kernel.functions import kernel_function
from semantic_kernel.utils.logging import setup_logging

# Set up logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)


class ICD10CMPlugin:
    """
    A plugin to retrieve ICD-10-CM codes and descriptions for a given medical term.
    """

    def __init__(self) -> None:
        """
        Initialize the ICD10CMPlugin.
        """
        self.logger = logging.getLogger("ICD10CMPlugin")
        self.base_url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

    @kernel_function(
        name="get_icd10cm_codes",
        description="Retrieve ICD-10-CM codes and descriptions for a given medical term.",
    )
    def get_icd10cm_codes(
        self, term: str, max_results: int = 7
    ) -> List[Tuple[str, str]]:
        """
        Retrieve ICD-10-CM codes and descriptions for a given medical term.

        Parameters:
            term (str): The medical term to search for.
            max_results (int): Maximum number of results to return (default is 7).

        Returns:
            List[Tuple[str, str]]: A list containing tuples of ICD-10-CM codes and their descriptions.
        """
        params = {"sf": "code,name", "terms": term, "maxList": max_results}
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = [(item[0], item[1]) for item in data[3]]
            return results
        else:
            self.logger.error(
                f"API request failed with status code {response.status_code}"
            )
            raise Exception(
                f"API request failed with status code {response.status_code}"
            )
