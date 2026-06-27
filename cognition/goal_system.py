from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict


class GoalState(Enum):
    PENDING = auto()
    ACTIVE = auto()
    ACHIEVED = auto()
    FAILED = auto()
    SUSPENDED = auto()
    ABANDONED = auto()


@dataclass
class Goal:
    goal_id: str
    description: str
    utility: float
    priority: float = 1.0
    state: GoalState = GoalState.PENDING
    parent_id: Optional[str] = None
    subgoal_ids: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    success_criteria: Optional[Callable] = None
    created_at: float = field(default_factory=time.perf_counter)
    deadline: Optional[float] = None
    progress: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def urgency(self) -> float:
        if self.deadline is None:
            return 0.0
        time_left = self.deadline - time.perf_counter()
        return 1.0 / max(time_left, 0.001)

    def score(self) -> float:
        return self.utility * self.priority * (1.0 + self.urgency())

    def age_seconds(self) -> float:
        return time.perf_counter() - self.created_at


class MotivationEngine:
    def __init__(self, homeostatic_vars: List[str], set_points: Dict[str, float]):
        self.vars = homeostatic_vars
        self.set_points = set_points
        self._current_values: Dict[str, float] = {v: set_points.get(v, 0.5) for v in homeostatic_vars}
        self._drive_weights: Dict[str, float] = {v: 1.0 for v in homeostatic_vars}

    def update(self, var: str, value: float):
        if var in self._current_values:
            self._current_values[var] = float(np.clip(value, 0.0, 1.0))

    def drive(self, var: str) -> float:
        if var not in self._current_values:
            return 0.0
        deficit = abs(self.set_points.get(var, 0.5) - self._current_values[var])
        return float(deficit * self._drive_weights.get(var, 1.0))

    def total_drive(self) -> float:
        return sum(self.drive(v) for v in self.vars)

    def most_urgent_need(self) -> Optional[str]:
        drives = {v: self.drive(v) for v in self.vars}
        if not drives:
            return None
        return max(drives, key=drives.get)

    def satisfaction_vector(self) -> np.ndarray:
        return np.array([
            1.0 - self.drive(v) for v in self.vars
        ])


class GoalSystem:
    def __init__(self, motivation_engine: Optional[MotivationEngine] = None):
        self._goals: Dict[str, Goal] = {}
        self._goal_counter: int = 0
        self._active_stack: List[str] = []
        self.motivation = motivation_engine
        self._achievement_history: List[Dict] = []
        self._n_cycles: int = 0

    def add_goal(
        self,
        description: str,
        utility: float,
        priority: float = 1.0,
        parent_id: Optional[str] = None,
        preconditions: List[str] = None,
        success_criteria: Optional[Callable] = None,
        deadline: Optional[float] = None,
    ) -> str:
        goal_id = f"g_{self._goal_counter:04d}"
        self._goal_counter += 1
        goal = Goal(
            goal_id=goal_id,
            description=description,
            utility=utility,
            priority=priority,
            parent_id=parent_id,
            preconditions=preconditions or [],
            success_criteria=success_criteria,
            deadline=deadline,
        )
        self._goals[goal_id] = goal
        if parent_id and parent_id in self._goals:
            self._goals[parent_id].subgoal_ids.append(goal_id)
        return goal_id

    def activate(self, goal_id: str) -> bool:
        if goal_id not in self._goals:
            return False
        goal = self._goals[goal_id]
        if not self._preconditions_met(goal):
            return False
        goal.state = GoalState.ACTIVE
        if goal_id not in self._active_stack:
            self._active_stack.append(goal_id)
        return True

    def _preconditions_met(self, goal: Goal) -> bool:
        for prec in goal.preconditions:
            if prec not in self._goals:
                return False
            if self._goals[prec].state != GoalState.ACHIEVED:
                return False
        return True

    def achieve(self, goal_id: str):
        if goal_id not in self._goals:
            return
        goal = self._goals[goal_id]
        goal.state = GoalState.ACHIEVED
        goal.progress = 1.0
        if goal_id in self._active_stack:
            self._active_stack.remove(goal_id)
        self._achievement_history.append({
            "goal_id": goal_id,
            "description": goal.description,
            "utility": goal.utility,
            "time_to_achieve": goal.age_seconds(),
        })
        if goal.parent_id and goal.parent_id in self._goals:
            parent = self._goals[goal.parent_id]
            n_subgoals = len(parent.subgoal_ids)
            n_achieved = sum(
                1 for gid in parent.subgoal_ids
                if gid in self._goals and self._goals[gid].state == GoalState.ACHIEVED
            )
            parent.progress = n_achieved / max(n_subgoals, 1)

    def fail(self, goal_id: str, reason: str = ""):
        if goal_id not in self._goals:
            return
        self._goals[goal_id].state = GoalState.FAILED
        self._goals[goal_id].metadata["failure_reason"] = reason
        if goal_id in self._active_stack:
            self._active_stack.remove(goal_id)

    def select_next(self) -> Optional[Goal]:
        pending = [
            g for g in self._goals.values()
            if g.state == GoalState.PENDING and self._preconditions_met(g)
        ]
        if not pending:
            return None
        return max(pending, key=lambda g: g.score())

    def update_progress(self, goal_id: str, delta: float):
        if goal_id not in self._goals:
            return
        goal = self._goals[goal_id]
        goal.progress = float(np.clip(goal.progress + delta, 0.0, 1.0))
        if goal.success_criteria is not None:
            try:
                if goal.success_criteria():
                    self.achieve(goal_id)
            except Exception:
                pass

    def cycle(self) -> Optional[str]:
        self._n_cycles += 1
        for goal_id in list(self._active_stack):
            goal = self._goals.get(goal_id)
            if goal is None:
                self._active_stack.remove(goal_id)
                continue
            if goal.deadline and time.perf_counter() > goal.deadline:
                self.fail(goal_id, "deadline_exceeded")
        next_goal = self.select_next()
        if next_goal is not None:
            self.activate(next_goal.goal_id)
            return next_goal.goal_id
        return None

    def active_goals(self) -> List[Goal]:
        return [self._goals[gid] for gid in self._active_stack if gid in self._goals]

    def goal_hierarchy(self, root_id: str) -> dict:
        if root_id not in self._goals:
            return {}
        goal = self._goals[root_id]
        return {
            "id": root_id,
            "description": goal.description,
            "state": goal.state.name,
            "progress": goal.progress,
            "score": goal.score(),
            "subgoals": [
                self.goal_hierarchy(sg_id) for sg_id in goal.subgoal_ids
            ],
        }

    def statistics(self) -> dict:
        state_counts = defaultdict(int)
        for g in self._goals.values():
            state_counts[g.state.name] += 1
        return {
            "total_goals": len(self._goals),
            "active": len(self._active_stack),
            "state_distribution": dict(state_counts),
            "n_achieved": len(self._achievement_history),
            "mean_utility_achieved": float(np.mean([a["utility"] for a in self._achievement_history])) if self._achievement_history else 0.0,
            "n_cycles": self._n_cycles,
        }