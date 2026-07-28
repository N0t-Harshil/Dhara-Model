#!/usr/bin/env python3
"""
Dataset creation and management for the Specialized Coding Model.

Provides a curated collection of production-quality Python and JavaScript
coding examples used for fine-tuning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeExample:
    """A single coding example consisting of a problem description and solution."""

    language: str
    problem: str
    solution: str
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Format as an instruction-following prompt for training."""
        return (
            f"### Instruction\n"
            f"Write a {self.language} solution for the following problem:\n"
            f"{self.problem}\n\n"
            f"### Response\n"
            f"{self.solution}"
        )


class SpecializedDataset:
    """Manages the collection of coding examples used for model training.

    Attributes:
        examples: List of all CodeExample instances in the dataset.
    """

    def __init__(self) -> None:
        self.examples: List[CodeExample] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_example(
        self,
        language: str,
        problem: str,
        solution: str,
        difficulty: str = "medium",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Add a single example to the dataset."""
        self.examples.append(
            CodeExample(
                language=language.lower(),
                problem=problem,
                solution=solution,
                difficulty=difficulty,
                tags=tags or [],
            )
        )

    def save_to_json(self, filepath: str | Path) -> None:
        """Persist the dataset to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(ex) for ex in self.examples]
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved %d examples to %s", len(data), filepath)

    @classmethod
    def load_from_json(cls, filepath: str | Path) -> "SpecializedDataset":
        """Load a dataset from a JSON file."""
        filepath = Path(filepath)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        ds = cls()
        for item in data:
            problem = item.get("problem")
            if not problem:
                problem = item.get("instruction") or ""
                inp = item.get("input", "")
                if inp:
                    problem = f"{problem}\n{inp}"
                    
            solution = item.get("solution") or item.get("output", "")
            language = item.get("language", "python")
            difficulty = item.get("difficulty", "medium")
            tags = item.get("tags", [])
            
            ds.examples.append(CodeExample(
                language=language,
                problem=problem,
                solution=solution,
                difficulty=difficulty,
                tags=tags
            ))
        logger.info("Loaded %d examples from %s", len(ds.examples), filepath)
        return ds

    def validate(self) -> bool:
        """Validate that every example has a non-empty problem and solution."""
        ok = True
        for i, ex in enumerate(self.examples):
            if not ex.problem.strip():
                logger.error("Example %d has an empty problem.", i)
                ok = False
            if not ex.solution.strip():
                logger.error("Example %d has an empty solution.", i)
                ok = False
            valid_langs = (
                "python", "rust", "golang", "cpp", "java", "typescript", 
                "javascript", "csharp", "php", "ruby", "swift", "kotlin", 
                "sql", "shell", "r", "scala", "objective-c", "perl", 
                "lua", "haskell", "julia", "zig", "assembly", "dart", "web-design"
            )
            if ex.language.lower() not in valid_langs:
                logger.warning("Example %d has unexpected language '%s'.", i, ex.language)
        logger.info(
            "Validation %s — %d examples checked.",
            "passed" if ok else "FAILED",
            len(self.examples),
        )
        return ok

    def get_by_language(self, language: str) -> List[CodeExample]:
        """Return examples filtered by language."""
        return [ex for ex in self.examples if ex.language == language.lower()]

    def stats(self) -> dict:
        """Return summary statistics for the dataset."""
        langs: dict[str, int] = {}
        for ex in self.examples:
            langs[ex.language] = langs.get(ex.language, 0) + 1
        return {"total": len(self.examples), "by_language": langs}

    # ------------------------------------------------------------------
    # Built-in example sets
    # ------------------------------------------------------------------

    def add_python_examples(self) -> None:
        """Add 10 production-quality Python examples."""

        self.add_example(
            language="python",
            problem="Implement merge sort that sorts a list of integers in ascending order.",
            solution='''\
def merge_sort(arr: list[int]) -> list[int]:
    """Sort *arr* in ascending order using the merge-sort algorithm."""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return _merge(left, right)


def _merge(left: list[int], right: list[int]) -> list[int]:
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
            difficulty="medium",
            tags=["sorting", "divide-and-conquer"],
        )

        self.add_example(
            language="python",
            problem="Implement binary search that returns the index of a target in a sorted list, or -1 if not found.",
            solution='''\
def binary_search(arr: list[int], target: int) -> int:
    """Return the index of *target* in sorted *arr*, or -1 if absent."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
            difficulty="easy",
            tags=["searching", "binary-search"],
        )

        self.add_example(
            language="python",
            problem="Implement a stack data structure with push, pop, peek, and is_empty methods.",
            solution='''\
from __future__ import annotations
from typing import Generic, TypeVar

T = TypeVar("T")


class Stack(Generic[T]):
    """A LIFO stack backed by a Python list."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Stack({self._items!r})"
''',
            difficulty="easy",
            tags=["data-structure", "stack"],
        )

        self.add_example(
            language="python",
            problem="Calculate the nth Fibonacci number efficiently using memoization.",
            solution='''\
from functools import lru_cache


@lru_cache(maxsize=None)
def fibonacci(n: int) -> int:
    """Return the *n*-th Fibonacci number (0-indexed)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
''',
            difficulty="easy",
            tags=["dynamic-programming", "memoization"],
        )

        self.add_example(
            language="python",
            problem="Implement a linked list with insert, delete, search, and display methods.",
            solution='''\
from __future__ import annotations
from typing import Optional


class Node:
    """Single node in a singly-linked list."""

    def __init__(self, data: int, next_node: Optional["Node"] = None) -> None:
        self.data = data
        self.next = next_node


class LinkedList:
    """Singly-linked list with basic operations."""

    def __init__(self) -> None:
        self.head: Optional[Node] = None

    def insert(self, data: int) -> None:
        """Insert *data* at the head of the list."""
        self.head = Node(data, self.head)

    def delete(self, data: int) -> bool:
        """Delete the first occurrence of *data*. Return True if found."""
        prev, curr = None, self.head
        while curr:
            if curr.data == data:
                if prev:
                    prev.next = curr.next
                else:
                    self.head = curr.next
                return True
            prev, curr = curr, curr.next
        return False

    def search(self, data: int) -> bool:
        """Return True if *data* exists in the list."""
        curr = self.head
        while curr:
            if curr.data == data:
                return True
            curr = curr.next
        return False

    def display(self) -> list[int]:
        """Return the list contents as a Python list."""
        result: list[int] = []
        curr = self.head
        while curr:
            result.append(curr.data)
            curr = curr.next
        return result
''',
            difficulty="medium",
            tags=["data-structure", "linked-list"],
        )

        self.add_example(
            language="python",
            problem="Implement BFS (Breadth-First Search) for a graph represented as an adjacency list.",
            solution='''\
from collections import deque


def bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """Return nodes in BFS order starting from *start*.

    Args:
        graph: Adjacency list mapping node → neighbours.
        start: Starting node.
    """
    visited: set[int] = set()
    order: list[int] = []
    queue: deque[int] = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return order
''',
            difficulty="medium",
            tags=["graph", "bfs"],
        )

        self.add_example(
            language="python",
            problem="Implement quicksort that sorts a list of integers in ascending order.",
            solution='''\
import random


def quicksort(arr: list[int]) -> list[int]:
    """Return a new list with *arr* sorted in ascending order (quicksort)."""
    if len(arr) <= 1:
        return list(arr)

    pivot = arr[random.randint(0, len(arr) - 1)]
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]
    return quicksort(less) + equal + quicksort(greater)
''',
            difficulty="medium",
            tags=["sorting", "divide-and-conquer"],
        )

        self.add_example(
            language="python",
            problem="Implement a binary search tree with insert, search, and in-order traversal.",
            solution='''\
from __future__ import annotations
from typing import Optional


class BSTNode:
    """Node in a Binary Search Tree."""

    def __init__(self, val: int) -> None:
        self.val = val
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None


class BinarySearchTree:
    """Binary Search Tree supporting insert, search, and in-order traversal."""

    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None

    def insert(self, val: int) -> None:
        self.root = self._insert(self.root, val)

    def _insert(self, node: Optional[BSTNode], val: int) -> BSTNode:
        if node is None:
            return BSTNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        return node

    def search(self, val: int) -> bool:
        return self._search(self.root, val)

    def _search(self, node: Optional[BSTNode], val: int) -> bool:
        if node is None:
            return False
        if val == node.val:
            return True
        return self._search(node.left if val < node.val else node.right, val)

    def inorder(self) -> list[int]:
        result: list[int] = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: Optional[BSTNode], acc: list[int]) -> None:
        if node:
            self._inorder(node.left, acc)
            acc.append(node.val)
            self._inorder(node.right, acc)
''',
            difficulty="medium",
            tags=["data-structure", "bst"],
        )

        self.add_example(
            language="python",
            problem="Implement Dijkstra's shortest-path algorithm for a weighted graph.",
            solution='''\
import heapq
from typing import Dict, List, Tuple


def dijkstra(
    graph: Dict[int, List[Tuple[int, float]]],
    start: int,
) -> Dict[int, float]:
    """Return shortest distances from *start* to every reachable node.

    Args:
        graph: Adjacency list — node → [(neighbour, weight), ...].
        start: Source node.
    """
    dist: Dict[int, float] = {start: 0.0}
    heap: list[Tuple[float, int]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist
''',
            difficulty="hard",
            tags=["graph", "shortest-path"],
        )

        self.add_example(
            language="python",
            problem="Implement an LRU (Least Recently Used) cache with get and put methods.",
            solution='''\
from collections import OrderedDict
from typing import Optional


class LRUCache:
    """Least-Recently-Used cache with O(1) get and put."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> Optional[int]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"LRUCache(capacity={self.capacity}, size={len(self)})"
''',
            difficulty="hard",
            tags=["data-structure", "cache"],
        )

        logger.info("Added 10 Python examples.")

    def add_javascript_examples(self) -> None:
        """Add 10 production-quality JavaScript examples."""

        self.add_example(
            language="javascript",
            problem="Implement merge sort that sorts an array of numbers in ascending order.",
            solution='''\
/**
 * Sort an array in ascending order using merge sort.
 * @param {number[]} arr
 * @returns {number[]}
 */
function mergeSort(arr) {
  if (arr.length <= 1) return arr;

  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));

  return merge(left, right);
}

function merge(left, right) {
  const result = [];
  let i = 0, j = 0;

  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) {
      result.push(left[i++]);
    } else {
      result.push(right[j++]);
    }
  }

  return result.concat(left.slice(i), right.slice(j));
}
''',
            difficulty="medium",
            tags=["sorting", "divide-and-conquer"],
        )

        self.add_example(
            language="javascript",
            problem="Implement binary search that returns the index of a target in a sorted array, or -1 if not found.",
            solution='''\
/**
 * Binary search for target in a sorted array.
 * @param {number[]} arr - sorted array
 * @param {number} target
 * @returns {number} index or -1
 */
function binarySearch(arr, target) {
  let lo = 0;
  let hi = arr.length - 1;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  return -1;
}
''',
            difficulty="easy",
            tags=["searching", "binary-search"],
        )

        self.add_example(
            language="javascript",
            problem="Implement a stack data structure with push, pop, peek, and isEmpty methods.",
            solution='''\
/**
 * LIFO Stack implementation.
 */
class Stack {
  constructor() {
    this.items = [];
  }

  push(item) {
    this.items.push(item);
  }

  pop() {
    if (this.isEmpty()) {
      throw new Error("Cannot pop from empty stack");
    }
    return this.items.pop();
  }

  peek() {
    if (this.isEmpty()) {
      throw new Error("Cannot peek empty stack");
    }
    return this.items[this.items.length - 1];
  }

  isEmpty() {
    return this.items.length === 0;
  }

  size() {
    return this.items.length;
  }
}
''',
            difficulty="easy",
            tags=["data-structure", "stack"],
        )

        self.add_example(
            language="javascript",
            problem="Calculate the nth Fibonacci number using memoization.",
            solution='''\
/**
 * Return the nth Fibonacci number (0-indexed) using memoization.
 * @param {number} n
 * @returns {number}
 */
function fibonacci(n, memo = {}) {
  if (n < 0) throw new Error("n must be non-negative");
  if (n <= 1) return n;
  if (memo[n] !== undefined) return memo[n];

  memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo);
  return memo[n];
}
''',
            difficulty="easy",
            tags=["dynamic-programming", "memoization"],
        )

        self.add_example(
            language="javascript",
            problem="Implement a linked list with insert, delete, search, and toArray methods.",
            solution='''\
class ListNode {
  constructor(data, next = null) {
    this.data = data;
    this.next = next;
  }
}

class LinkedList {
  constructor() {
    this.head = null;
  }

  /** Insert at head. */
  insert(data) {
    this.head = new ListNode(data, this.head);
  }

  /** Delete first occurrence of data. Returns true if found. */
  delete(data) {
    let prev = null;
    let curr = this.head;

    while (curr) {
      if (curr.data === data) {
        if (prev) {
          prev.next = curr.next;
        } else {
          this.head = curr.next;
        }
        return true;
      }
      prev = curr;
      curr = curr.next;
    }

    return false;
  }

  /** Return true if data exists. */
  search(data) {
    let curr = this.head;
    while (curr) {
      if (curr.data === data) return true;
      curr = curr.next;
    }
    return false;
  }

  /** Convert to an array. */
  toArray() {
    const result = [];
    let curr = this.head;
    while (curr) {
      result.push(curr.data);
      curr = curr.next;
    }
    return result;
  }
}
''',
            difficulty="medium",
            tags=["data-structure", "linked-list"],
        )

        self.add_example(
            language="javascript",
            problem="Implement BFS (Breadth-First Search) for a graph represented as an adjacency list.",
            solution='''\
/**
 * BFS traversal of a graph.
 * @param {Object.<number, number[]>} graph - adjacency list
 * @param {number} start
 * @returns {number[]} nodes in BFS order
 */
function bfs(graph, start) {
  const visited = new Set([start]);
  const queue = [start];
  const order = [];

  while (queue.length > 0) {
    const node = queue.shift();
    order.push(node);

    for (const neighbour of (graph[node] || [])) {
      if (!visited.has(neighbour)) {
        visited.add(neighbour);
        queue.push(neighbour);
      }
    }
  }

  return order;
}
''',
            difficulty="medium",
            tags=["graph", "bfs"],
        )

        self.add_example(
            language="javascript",
            problem="Implement quicksort that sorts an array of numbers in ascending order.",
            solution='''\
/**
 * Sort an array in ascending order using quicksort.
 * @param {number[]} arr
 * @returns {number[]}
 */
function quicksort(arr) {
  if (arr.length <= 1) return [...arr];

  const pivotIndex = Math.floor(Math.random() * arr.length);
  const pivot = arr[pivotIndex];

  const less = arr.filter((x) => x < pivot);
  const equal = arr.filter((x) => x === pivot);
  const greater = arr.filter((x) => x > pivot);

  return [...quicksort(less), ...equal, ...quicksort(greater)];
}
''',
            difficulty="medium",
            tags=["sorting", "divide-and-conquer"],
        )

        self.add_example(
            language="javascript",
            problem="Implement a binary search tree with insert, search, and inorder traversal.",
            solution='''\
class BSTNode {
  constructor(val) {
    this.val = val;
    this.left = null;
    this.right = null;
  }
}

class BinarySearchTree {
  constructor() {
    this.root = null;
  }

  insert(val) {
    this.root = this._insert(this.root, val);
  }

  _insert(node, val) {
    if (!node) return new BSTNode(val);
    if (val < node.val) node.left = this._insert(node.left, val);
    else if (val > node.val) node.right = this._insert(node.right, val);
    return node;
  }

  search(val) {
    return this._search(this.root, val);
  }

  _search(node, val) {
    if (!node) return false;
    if (val === node.val) return true;
    return val < node.val
      ? this._search(node.left, val)
      : this._search(node.right, val);
  }

  inorder() {
    const result = [];
    this._inorder(this.root, result);
    return result;
  }

  _inorder(node, acc) {
    if (node) {
      this._inorder(node.left, acc);
      acc.push(node.val);
      this._inorder(node.right, acc);
    }
  }
}
''',
            difficulty="medium",
            tags=["data-structure", "bst"],
        )

        self.add_example(
            language="javascript",
            problem="Implement Dijkstra's shortest-path algorithm for a weighted graph.",
            solution='''\
/**
 * Dijkstra's algorithm returning shortest distances from start.
 * @param {Object.<number, [number, number][]>} graph  node → [[neighbour, weight], ...]
 * @param {number} start
 * @returns {Object.<number, number>} node → shortest distance
 */
function dijkstra(graph, start) {
  const dist = { [start]: 0 };
  const visited = new Set();
  const pq = [[0, start]]; // [distance, node]

  while (pq.length > 0) {
    pq.sort((a, b) => a[0] - b[0]);
    const [d, u] = pq.shift();

    if (visited.has(u)) continue;
    visited.add(u);

    for (const [v, w] of (graph[u] || [])) {
      const nd = d + w;
      if (nd < (dist[v] ?? Infinity)) {
        dist[v] = nd;
        pq.push([nd, v]);
      }
    }
  }

  return dist;
}
''',
            difficulty="hard",
            tags=["graph", "shortest-path"],
        )

        self.add_example(
            language="javascript",
            problem="Implement an LRU cache with get and put methods.",
            solution='''\
class LRUCache {
  /**
   * @param {number} capacity
   */
  constructor(capacity) {
    if (capacity <= 0) throw new Error("Capacity must be positive");
    this.capacity = capacity;
    this.cache = new Map();
  }

  /**
   * @param {number} key
   * @returns {number|null}
   */
  get(key) {
    if (!this.cache.has(key)) return null;
    const value = this.cache.get(key);
    // Move to end (most recently used)
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  /**
   * @param {number} key
   * @param {number} value
   */
  put(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }
    this.cache.set(key, value);
    if (this.cache.size > this.capacity) {
      // Delete the least recently used (first entry)
      const lruKey = this.cache.keys().next().value;
      this.cache.delete(lruKey);
    }
  }

  get size() {
    return this.cache.size;
  }
}
''',
            difficulty="hard",
            tags=["data-structure", "cache"],
        )

        logger.info("Added 10 JavaScript examples.")

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------

    @classmethod
    def build_default(cls) -> "SpecializedDataset":
        """Create a dataset pre-populated with all built-in and collected examples."""
        ds = cls()
        ds.add_python_examples()
        ds.add_javascript_examples()
        
        # Also load from combined_dataset.json if it exists to get HF data
        combined_path = Path(__file__).resolve().parent.parent / "data" / "combined_dataset.json"
        if combined_path.exists():
            logger.info("Loading additional examples from %s", combined_path)
            extra_ds = cls.load_from_json(combined_path)
            # Avoid duplicates by problem
            existing_problems = {ex.problem for ex in ds.examples}
            for ex in extra_ds.examples:
                if ex.problem not in existing_problems:
                    ds.examples.append(ex)
                    existing_problems.add(ex.problem)
        
        return ds

    def save_split(self, data_dir: str | Path) -> None:
        """Save per-language and combined JSON files to *data_dir*."""
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        py = [asdict(ex) for ex in self.get_by_language("python")]
        js = [asdict(ex) for ex in self.get_by_language("javascript")]
        combined = [asdict(ex) for ex in self.examples]

        (data_dir / "python_examples.json").write_text(
            json.dumps(py, indent=2), encoding="utf-8"
        )
        (data_dir / "javascript_examples.json").write_text(
            json.dumps(js, indent=2), encoding="utf-8"
        )
        (data_dir / "combined_dataset.json").write_text(
            json.dumps(combined, indent=2), encoding="utf-8"
        )
        logger.info(
            "Saved split dataset to %s (python=%d, js=%d, combined=%d)",
            data_dir, len(py), len(js), len(combined),
        )
