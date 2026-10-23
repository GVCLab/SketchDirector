import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional


class ExperienceMemory:
    def __init__(self, memory_dir="memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        self.session_memory = {
            "operations": [],         
            "object_states": {},       
            "failed_attempts": [],     
            "successful_attempts": []  
        }
        
        self.long_term_memory = self._load_long_term_memory()
        
        self.skill_library = self._load_skill_library()
    
    def _load_long_term_memory(self) -> Dict:
        memory_path = os.path.join(self.memory_dir, "long_term_memory.pkl")
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                return pickle.load(f)
        return {
            "success_patterns": [],     
            "failure_patterns": [],     
            "scene_signatures": [],      
            "skill_usage_stats": {}     
        }
    
    def _load_skill_library(self) -> Dict:
        skill_path = os.path.join(self.memory_dir, "skill_library.json")
        if os.path.exists(skill_path):
            with open(skill_path, 'r') as f:
                return json.load(f)
        return {
            "primitives": [
                "move", "scale", "rotate", "delete", "layered", "modify"
            ],
            "learned_skills": [],      
            "skill_code": {}            
        }
    
    def save_memory(self):
        memory_path = os.path.join(self.memory_dir, "long_term_memory.pkl")
        with open(memory_path, 'wb') as f:
            pickle.dump(self.long_term_memory, f)
        
        skill_path = os.path.join(self.memory_dir, "skill_library.json")
        with open(skill_path, 'w') as f:
            json.dump(self.skill_library, f, indent=2)
    
    def record_operation(self, operation: Dict):
        self.session_memory["operations"].append({
            "timestamp": datetime.now().isoformat(),
            **operation
        })
    
    def record_failure(self, failure: Dict):
        self.session_memory["failed_attempts"].append(failure)
        
        pattern = self._extract_failure_pattern(failure)
        if pattern:
            self.long_term_memory["failure_patterns"].append(pattern)
            self.save_memory()
    
    def record_success(self, success: Dict):
        self.session_memory["successful_attempts"].append(success)
        
        pattern = self._extract_success_pattern(success)
        if pattern:
            self.long_term_memory["success_patterns"].append(pattern)
            self.save_memory()
    
    def _extract_failure_pattern(self, failure: Dict) -> Optional[Dict]:
        return {
            "scene_signature": failure.get("scene_signature", ""),
            "failed_operation": failure.get("operation", ""),
            "error_type": failure.get("error_type", ""),
            "error_step": failure.get("step", -1),
            "recommended_fix": failure.get("recommended_fix", ""),
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_success_pattern(self, success: Dict) -> Optional[Dict]:
        return {
            "scene_signature": success.get("scene_signature", ""),
            "operation_sequence": success.get("operation_sequence", []),
            "key_parameters": success.get("key_parameters", {}),
            "success_score": success.get("score", 1.0),
            "timestamp": datetime.now().isoformat()
        }
    
    def retrieve_similar_experiences(self, scene_signature: str) -> List[Dict]:
        similar = []
        for pattern in self.long_term_memory["success_patterns"]:
            if self._compute_similarity(scene_signature, pattern.get("scene_signature", "")) > 0.6:
                similar.append(pattern)
        return similar
    
    def _compute_similarity(self, sig1: str, sig2: str) -> float:
        if not sig1 or not sig2:
            return 0.0
        words1 = set(sig1.lower().split())
        words2 = set(sig2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
