from typing import Dict, List, Any

import numpy as np
import yaml


class ConfigParser:
    @staticmethod
    def _add_value(categories: Dict, rest: List[str], value):
        key = rest.pop(0)

        if len(rest) > 0 and key not in categories:
            categories[key] = dict()
        elif len(rest) > 0 and not isinstance(categories[key], dict):
            old_val = categories[key]
            categories[key] = dict()
            categories[key]["value"] = old_val

        if len(rest) > 0:
            ConfigParser._add_value(categories[key], rest, value)
        else:
            categories[key] = value

    @staticmethod
    def read(file: str, numpy_mode: bool = False) -> Dict[str, Any]:
        with open(file) as f:
            parameters = yaml.safe_load(f.read())

        categories = dict()

        for item in dict(parameters).items():
            key, value = item

            if isinstance(value, list):
                value = np.array(value) if numpy_mode else value
            elif isinstance(value, dict) and len(value.keys()) == 1 and "value" in value:
                value = value["value"]

            key_split = [x.lower() for x in (key.split("."))]

            ConfigParser._add_value(categories, key_split, value)

        return categories
