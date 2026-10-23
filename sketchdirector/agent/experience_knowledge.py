import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List


class ExperienceKnowledge:
    def __init__(self, kb_path: str = "experience_knowledge.json"):
        self.kb_path = kb_path
        self.knowledge = self._load_or_create()

    def _load_or_create(self) -> Dict[str, Any]:
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    knowledge = json.load(f)

                # Basic structure validation
                if (
                    isinstance(knowledge, dict)
                    and "meta" in knowledge
                    and "experiences" in knowledge
                ):
                    return knowledge

            except (OSError, json.JSONDecodeError) as e:
                print(
                    f"[ExperienceKnowledge] Failed to load knowledge base "
                    f"from '{self.kb_path}': {e}"
                )

        now = datetime.now().isoformat()

        return {
            "meta": {
                "created_at": now,
                "last_updated": now,
                "total_experiences": 0,
            },
            "experiences": [],
        }

    def save(self) -> None:
        self.knowledge["meta"]["last_updated"] = datetime.now().isoformat()
        self.knowledge["meta"]["total_experiences"] = len(
            self.knowledge["experiences"]
        )

        parent_dir = os.path.dirname(self.kb_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(
                self.knowledge,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def add_experience(self, experience: Dict[str, Any]) -> str:
        if not isinstance(experience, dict):
            raise TypeError("experience must be a dictionary.")

        new_experience = deepcopy(experience)

        exp_id = f"exp_{len(self.knowledge['experiences']) + 1}"

        new_experience["id"] = exp_id
        new_experience["created_at"] = datetime.now().isoformat()
        new_experience["verified_count"] = 0
        new_experience["failed_count"] = 0

        self.knowledge["experiences"].append(new_experience)
        self.save()

        return exp_id

    def update_experience(
        self,
        exp_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dictionary.")

        for experience in self.knowledge["experiences"]:
            if experience.get("id") == exp_id:
                experience.update(updates)
                experience["last_updated"] = datetime.now().isoformat()
                self.save()
                return True

        return False

    def get_experiences(self) -> List[Dict[str, Any]]:
        
        return self.knowledge["experiences"]

    def get_formatted_experiences(self) -> str:
        
        experiences = self.knowledge["experiences"]

        if not experiences:
            return "No previous experiences are available."

        lines = [
            "",
            "=== Existing Experiences ===",
            "",
            (
                "The following experiences were summarized from previous "
                "tasks. Use them as references only when they are relevant "
                "to the current scene."
            ),
            "",
        ]

        for index, experience in enumerate(experiences, start=1):
            lines.append(
                f"Experience {index} "
                f"(ID: {experience.get('id', 'unknown')}):"
            )

            lines.append(
                f"  Content: {experience.get('content', '')}"
            )

            lines.append(
                "  Applicable scenario: "
                f"{experience.get('applicable_scenario', 'General')}"
            )

            key_points = experience.get("key_points", [])
            if key_points:
                lines.append("  Key points:")
                for point in key_points:
                    lines.append(f"    - {point}")

            lines.append(
                "  Verified count: "
                f"{experience.get('verified_count', 0)}"
            )

            failed_count = experience.get("failed_count", 0)

            if failed_count > 0:
                lines.append(
                    f"  Previous failures: {failed_count}. "
                    "This experience may require adaptation."
                )

            lines.append("")

        lines.extend(
            [
                (
                    "Note: These experiences were derived from previous "
                    "tasks and may not directly apply to the current scene."
                ),
                (
                    "Evaluate their relevance according to the current "
                    "objects, spatial relationships, and layout requirements."
                ),
            ]
        )

        return "\n".join(lines)