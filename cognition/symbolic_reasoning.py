from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Generator, List, Optional, Set, Tuple, Union


@dataclass
class Fact:
    predicate: str
    arguments: Tuple[str, ...]
    confidence: float = 1.0
    source: str = "assert"
    timestamp: float = field(default_factory=time.perf_counter)

    def __hash__(self):
        return hash((self.predicate, self.arguments))

    def __eq__(self, other):
        return (
            isinstance(other, Fact)
            and self.predicate == other.predicate
            and self.arguments == other.arguments
        )

    def __repr__(self):
        args_str = ", ".join(self.arguments)
        return f"{self.predicate}({args_str})"

    def matches(self, pattern: "Fact") -> Optional[Dict[str, str]]:
        if self.predicate != pattern.predicate:
            return None
        if len(self.arguments) != len(pattern.arguments):
            return None
        bindings: Dict[str, str] = {}
        for self_arg, pat_arg in zip(self.arguments, pattern.arguments):
            if pat_arg.startswith("?"):
                var = pat_arg
                if var in bindings:
                    if bindings[var] != self_arg:
                        return None
                else:
                    bindings[var] = self_arg
            elif self_arg != pat_arg:
                return None
        return bindings


@dataclass
class Rule:
    head: Fact
    body: List[Fact]
    confidence: float = 1.0
    name: str = ""
    priority: int = 0

    def __repr__(self):
        body_str = " AND ".join(str(f) for f in self.body)
        return f"{body_str} => {self.head}"


class KnowledgeBase:
    def __init__(self):
        self._facts: Set[Fact] = set()
        self._rules: List[Rule] = []
        self._fact_index: Dict[str, List[Fact]] = {}

    def assert_fact(self, predicate: str, *args: str, confidence: float = 1.0, source: str = "user") -> Fact:
        fact = Fact(predicate=predicate, arguments=tuple(args), confidence=confidence, source=source)
        if fact in self._facts:
            for existing in self._facts:
                if existing == fact:
                    existing.confidence = max(existing.confidence, confidence)
                    return existing
        self._facts.add(fact)
        if predicate not in self._fact_index:
            self._fact_index[predicate] = []
        self._fact_index[predicate].append(fact)
        return fact

    def retract_fact(self, predicate: str, *args: str) -> bool:
        target = Fact(predicate=predicate, arguments=tuple(args))
        if target in self._facts:
            self._facts.discard(target)
            if predicate in self._fact_index:
                self._fact_index[predicate] = [
                    f for f in self._fact_index[predicate] if f != target
                ]
            return True
        return False

    def add_rule(
        self,
        head_pred: str, head_args: Tuple[str, ...],
        body: List[Tuple[str, Tuple[str, ...]]],
        confidence: float = 1.0, name: str = "", priority: int = 0,
    ) -> Rule:
        head = Fact(head_pred, head_args)
        body_facts = [Fact(p, a) for p, a in body]
        rule = Rule(head=head, body=body_facts, confidence=confidence, name=name, priority=priority)
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        return rule

    def query(self, predicate: str, *args: str) -> List[Tuple[Fact, Dict[str, str]]]:
        pattern = Fact(predicate, tuple(args))
        results = []
        facts_to_check = self._fact_index.get(predicate, [])
        for fact in facts_to_check:
            bindings = fact.matches(pattern)
            if bindings is not None:
                results.append((fact, bindings))
        return results

    def fact_exists(self, predicate: str, *args: str) -> bool:
        return len(self.query(predicate, *args)) > 0

    def all_facts(self) -> List[Fact]:
        return sorted(self._facts, key=lambda f: f.timestamp)

    def statistics(self) -> dict:
        return {
            "n_facts": len(self._facts),
            "n_rules": len(self._rules),
            "n_predicates": len(self._fact_index),
        }


class InferenceEngine:
    def __init__(self, kb: KnowledgeBase, max_depth: int = 10, max_new_facts: int = 1000):
        self.kb = kb
        self.max_depth = max_depth
        self.max_new_facts = max_new_facts
        self._inference_count: int = 0
        self._derived_facts: List[Fact] = []

    def _apply_bindings(self, fact: Fact, bindings: Dict[str, str]) -> Fact:
        new_args = tuple(
            bindings.get(arg, arg) if arg.startswith("?") else arg
            for arg in fact.arguments
        )
        return Fact(fact.predicate, new_args, source="derived")

    def _match_body(
        self, body: List[Fact], bindings: Dict[str, str], body_idx: int = 0
    ) -> Generator[Dict[str, str], None, None]:
        if body_idx >= len(body):
            yield bindings
            return
        pattern = body[body_idx]
        instantiated = self._apply_bindings(pattern, bindings)
        matches = self.kb.query(instantiated.predicate, *instantiated.arguments)
        for _, new_bindings in matches:
            merged = {**bindings, **new_bindings}
            conflict = any(
                k in bindings and bindings[k] != v
                for k, v in new_bindings.items()
            )
            if not conflict:
                yield from self._match_body(body, merged, body_idx + 1)

    def forward_chain(self) -> List[Fact]:
        new_facts = []
        changed = True
        iterations = 0
        while changed and iterations < self.max_depth and len(new_facts) < self.max_new_facts:
            changed = False
            iterations += 1
            for rule in self.kb._rules:
                for bindings in self._match_body(rule.body, {}):
                    conclusion = self._apply_bindings(rule.head, bindings)
                    conclusion.confidence = rule.confidence
                    if not self.kb.fact_exists(conclusion.predicate, *conclusion.arguments):
                        self.kb.assert_fact(
                            conclusion.predicate, *conclusion.arguments,
                            confidence=conclusion.confidence, source="derived"
                        )
                        new_facts.append(conclusion)
                        changed = True
        self._inference_count += iterations
        self._derived_facts.extend(new_facts)
        return new_facts

    def backward_chain(self, goal: Fact, depth: int = 0) -> List[Dict[str, str]]:
        if depth > self.max_depth:
            return []
        direct = self.kb.query(goal.predicate, *goal.arguments)
        if direct:
            return [b for _, b in direct]
        solutions = []
        for rule in self.kb._rules:
            head_bindings = rule.head.matches(goal)
            if head_bindings is None:
                continue
            for body_bindings in self._match_body(rule.body, head_bindings):
                instantiated_head = self._apply_bindings(rule.head, body_bindings)
                all_body_proved = True
                final_bindings = body_bindings.copy()
                for body_fact in rule.body:
                    inst_body = self._apply_bindings(body_fact, body_bindings)
                    sub_solutions = self.backward_chain(inst_body, depth + 1)
                    if not sub_solutions:
                        all_body_proved = False
                        break
                    final_bindings.update(sub_solutions[0])
                if all_body_proved:
                    solutions.append(final_bindings)
        return solutions

    def explain(self, fact: Fact) -> List[str]:
        explanation = []
        if self.kb.fact_exists(fact.predicate, *fact.arguments):
            for f in self.kb.all_facts():
                if f == fact and f.source == "assert":
                    explanation.append(f"BASE FACT: {fact}")
                    return explanation
            explanation.append(f"DERIVED: {fact}")
            for rule in self.kb._rules:
                for bindings in self._match_body(rule.body, {}):
                    conclusion = self._apply_bindings(rule.head, bindings)
                    if conclusion == fact:
                        explanation.append(f"  VIA RULE: {rule.name or str(rule)}")
                        for bf in rule.body:
                            inst = self._apply_bindings(bf, bindings)
                            explanation.append(f"    REQUIRES: {inst}")
        return explanation


class SymbolicReasoner:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.engine = InferenceEngine(self.kb)
        self._query_history: List[Dict] = []

    def assert_fact(self, predicate: str, *args: str, confidence: float = 1.0) -> Fact:
        return self.kb.assert_fact(predicate, *args, confidence=confidence)

    def add_rule(
        self,
        name: str,
        head: Tuple[str, Tuple[str, ...]],
        body: List[Tuple[str, Tuple[str, ...]]],
        confidence: float = 1.0,
        priority: int = 0,
    ) -> Rule:
        return self.kb.add_rule(
            head[0], head[1], body,
            confidence=confidence, name=name, priority=priority,
        )

    def query(self, predicate: str, *args: str) -> List[Tuple[Fact, Dict[str, str]]]:
        self._query_history.append({"predicate": predicate, "args": args, "time": time.perf_counter()})
        return self.kb.query(predicate, *args)

    def infer(self) -> List[Fact]:
        return self.engine.forward_chain()

    def prove(self, predicate: str, *args: str) -> bool:
        goal = Fact(predicate, tuple(args))
        solutions = self.engine.backward_chain(goal)
        return len(solutions) > 0

    def explain(self, predicate: str, *args: str) -> List[str]:
        return self.engine.explain(Fact(predicate, tuple(args)))

    def integrate_neural_output(
        self,
        labels: List[str],
        probabilities: np.ndarray,
        threshold: float = 0.6,
    ) -> List[Fact]:
        facts = []
        for label, prob in zip(labels, probabilities):
            if prob >= threshold:
                parts = label.split("_")
                pred = parts[0] if parts else label
                args = tuple(parts[1:]) if len(parts) > 1 else ("true",)
                fact = self.assert_fact(pred, *args, confidence=float(prob))
                facts.append(fact)
        return facts

    def statistics(self) -> dict:
        return {
            "kb_stats": self.kb.statistics(),
            "n_inferences": self.engine._inference_count,
            "n_derived_facts": len(self.engine._derived_facts),
            "n_queries": len(self._query_history),
        }