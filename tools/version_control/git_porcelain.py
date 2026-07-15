"""
Git Porcelain Operations Module

Simulates git porcelain operations: commit, branch, merge, rebase,
cherry-pick, stash, tag, diff, log, status, remote operations,
and conflict resolution using internal data structures.
"""

from __future__ import annotations

import collections
import difflib
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ObjectType(Enum):
    BLOB = "blob"
    TREE = "tree"
    COMMIT = "commit"
    TAG = "tag"


class FileStatus(Enum):
    UNMODIFIED = " "
    ADDED = "A"
    MODIFIED = "M"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UNTRACKED = "??"
    IGNORED = "!!"
    UNMERGED = "U"
    STAGED_ADDED = "A"
    STAGED_MODIFIED = "M"
    STAGED_DELETED = "D"


class MergeState(Enum):
    NONE = "none"
    MERGING = "merging"
    CONFLICTING = "conflicting"
    REVERTING = "reverting"
    CHERRY_PICKING = "cherry_picking"


class RebaseState(Enum):
    NONE = "none"
    REBASING = "rebasing"
    PAUSED = "paused"
    CONFLICTING = "conflicting"


@dataclass
class GitBlob:
    """Represents a file content object."""
    oid: str
    content: str
    size: int = 0

    def __post_init__(self) -> None:
        self.size = len(self.content.encode("utf-8"))


@dataclass
class GitTreeEntry:
    """Entry in a tree object."""
    name: str
    oid: str
    object_type: ObjectType = ObjectType.BLOB
    mode: str = "100644"


@dataclass
class GitTree:
    """Represents a directory tree."""
    oid: str
    entries: Dict[str, GitTreeEntry] = field(default_factory=dict)


@dataclass
class GitCommit:
    """Represents a commit object."""
    oid: str
    tree_oid: str
    parent_oids: List[str] = field(default_factory=list)
    author_name: str = ""
    author_email: str = ""
    author_date: datetime = field(default_factory=datetime.utcnow)
    committer_name: str = ""
    committer_email: str = ""
    committer_date: datetime = field(default_factory=datetime.utcnow)
    message: str = ""
    gpg_signature: Optional[str] = None

    @property
    def is_merge(self) -> bool:
        return len(self.parent_oids) > 1


@dataclass
class GitTag:
    """Represents a tag object."""
    name: str
    oid: str
    target_oid: str
    tagger_name: str = ""
    tagger_email: str = ""
    message: str = ""
    tag_date: datetime = field(default_factory=datetime.utcnow)
    is_annotated: bool = True


@dataclass
class GitBranch:
    """Represents a branch reference."""
    name: str
    target_oid: str
    is_remote: bool = False
    upstream: Optional[str] = None


@dataclass
class StashEntry:
    """Represents a stash entry."""
    oid: str
    message: str
    commit_oid: str
    index_oid: str
    untracked_oid: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DiffEntry:
    """Represents a single file diff."""
    old_path: str
    new_path: str
    old_oid: Optional[str] = None
    new_oid: Optional[str] = None
    status: FileStatus = FileStatus.MODIFIED
    additions: int = 0
    deletions: int = 0
    hunks: List[str] = field(default_factory=list)


@dataclass
class DiffResult:
    """Result of a diff operation."""
    entries: List[DiffEntry] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    raw_output: str = ""

    @property
    def summary(self) -> str:
        files = len(self.entries)
        return f"{files} file(s) changed, {self.total_additions} insertions(+), {self.total_deletions} deletions(-)"


@dataclass
class StatusEntry:
    """Entry in git status output."""
    path: str
    index_status: str = " "
    working_tree_status: str = " "
    renamed_from: Optional[str] = None


@dataclass
class StatusResult:
    """Result of git status."""
    branch: str = "main"
    entries: List[StatusEntry] = field(default_factory=list)
    ahead: int = 0
    behind: int = 0
    staged_count: int = 0
    unstaged_count: int = 0
    untracked_count: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.entries


@dataclass
class LogEntry:
    """Entry in git log."""
    oid: str
    author_name: str
    author_email: str
    author_date: datetime
    committer_name: str
    message: str
    refs: List[str] = field(default_factory=list)


@dataclass
class ConflictEntry:
    """Represents a merge conflict."""
    path: str
    ours_content: str = ""
    theirs_content: str = ""
    base_content: str = ""
    is_resolved: bool = False


@dataclass
class RemoteInfo:
    """Information about a remote."""
    name: str
    url: str
    fetch_url: Optional[str] = None
    push_url: Optional[str] = None
    head_oid: Optional[str] = None


# ---------------------------------------------------------------------------
# Object Store
# ---------------------------------------------------------------------------

class ObjectStore:
    """In-memory git object store."""

    def __init__(self) -> None:
        self._blobs: Dict[str, GitBlob] = {}
        self._trees: Dict[str, GitTree] = {}
        self._commits: Dict[str, GitCommit] = {}
        self._tags: Dict[str, GitTag] = {}
        self._lock = threading.Lock()

    def store_blob(self, content: str) -> str:
        oid = self._hash_object(ObjectType.BLOB, content)
        blob = GitBlob(oid=oid, content=content)
        with self._lock:
            self._blobs[oid] = blob
        return oid

    def store_tree(self, tree: GitTree) -> str:
        if not tree.oid:
            tree.oid = self._hash_tree(tree)
        with self._lock:
            self._trees[tree.oid] = tree
        return tree.oid

    def store_commit(self, commit: GitCommit) -> str:
        if not commit.oid:
            commit.oid = self._hash_commit(commit)
        with self._lock:
            self._commits[commit.oid] = commit
        return commit.oid

    def store_tag(self, tag: GitTag) -> None:
        with self._lock:
            self._tags[tag.name] = tag

    def get_blob(self, oid: str) -> Optional[GitBlob]:
        return self._blobs.get(oid)

    def get_tree(self, oid: str) -> Optional[GitTree]:
        return self._trees.get(oid)

    def get_commit(self, oid: str) -> Optional[GitCommit]:
        return self._commits.get(oid)

    def get_tag(self, name: str) -> Optional[GitTag]:
        return self._tags.get(name)

    def list_tags(self) -> List[GitTag]:
        return list(self._tags.values())

    def _hash_object(self, obj_type: ObjectType, content: str) -> str:
        header = f"{obj_type.value} {len(content.encode('utf-8'))}\0"
        full = header + content
        return hashlib.sha1(full.encode("utf-8")).hexdigest()

    def _hash_tree(self, tree: GitTree) -> str:
        entries_str = ""
        for name, entry in sorted(tree.entries.items()):
            entries_str += f"{entry.mode} {entry.object_type.value} {entry.oid}\t{name}\0"
        header = f"tree {len(entries_str.encode('utf-8'))}\0"
        return hashlib.sha1((header + entries_str).encode("utf-8")).hexdigest()

    def _hash_commit(self, commit: GitCommit) -> str:
        parents = "".join(f"parent {p}\n" for p in commit.parent_oids)
        content = (
            f"tree {commit.tree_oid}\n"
            f"{parents}"
            f"author {commit.author_name} <{commit.author_email}> "
            f"{int(commit.author_date.timestamp())} +0000\n"
            f"committer {commit.committer_name} <{commit.committer_email}> "
            f"{int(commit.committer_date.timestamp())} +0000\n"
            f"\n{commit.message}\n"
        )
        header = f"commit {len(content.encode('utf-8'))}\0"
        return hashlib.sha1((header + content).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Index (Staging Area)
# ---------------------------------------------------------------------------

class Index:
    """Simulates the git index (staging area)."""

    def __init__(self) -> None:
        self._entries: Dict[str, str] = {}  # path -> blob oid
        self._lock = threading.Lock()

    def add(self, path: str, blob_oid: str) -> None:
        with self._lock:
            self._entries[path] = blob_oid

    def remove(self, path: str) -> None:
        with self._lock:
            self._entries.pop(path, None)

    def get(self, path: str) -> Optional[str]:
        return self._entries.get(path)

    def list_entries(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Diff Engine
# ---------------------------------------------------------------------------

class DiffEngine:
    """Computes diffs between file contents and trees."""

    def __init__(self, object_store: ObjectStore) -> None:
        self.object_store = object_store

    def diff_blobs(self, old_content: str, new_content: str) -> Tuple[int, int, List[str]]:
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
        additions = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
        deletions = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
        return additions, deletions, diff

    def diff_files(
        self,
        old_oid: Optional[str],
        new_oid: Optional[str],
        old_path: str = "",
        new_path: str = "",
    ) -> DiffEntry:
        old_content = ""
        new_content = ""
        if old_oid:
            blob = self.object_store.get_blob(old_oid)
            if blob:
                old_content = blob.content
        if new_oid:
            blob = self.object_store.get_blob(new_oid)
            if blob:
                new_content = blob.content

        if not old_oid:
            status = FileStatus.ADDED
        elif not new_oid:
            status = FileStatus.DELETED
        elif old_path != new_path and old_content == new_content:
            status = FileStatus.RENAMED
        else:
            status = FileStatus.MODIFIED

        additions, deletions, hunks = self.diff_blobs(old_content, new_content)
        return DiffEntry(
            old_path=old_path or new_path,
            new_path=new_path or old_path,
            old_oid=old_oid,
            new_oid=new_oid,
            status=status,
            additions=additions,
            deletions=deletions,
            hunks=hunks,
        )

    def diff_trees(
        self,
        old_tree_oid: Optional[str],
        new_tree_oid: Optional[str],
    ) -> DiffResult:
        result = DiffResult()
        old_entries: Dict[str, str] = {}
        new_entries: Dict[str, str] = {}

        if old_tree_oid:
            tree = self.object_store.get_tree(old_tree_oid)
            if tree:
                old_entries = {name: entry.oid for name, entry in tree.entries.items()}

        if new_tree_oid:
            tree = self.object_store.get_tree(new_tree_oid)
            if tree:
                new_entries = {name: entry.oid for name, entry in tree.entries.items()}

        all_paths = sorted(set(list(old_entries.keys()) + list(new_entries.keys())))
        for path in all_paths:
            old_oid = old_entries.get(path)
            new_oid = new_entries.get(path)
            entry = self.diff_files(old_oid, new_oid, path, path)
            result.entries.append(entry)
            result.total_additions += entry.additions
            result.total_deletions += entry.deletions

        return result


# ---------------------------------------------------------------------------
# Conflict Resolver
# ---------------------------------------------------------------------------

class ConflictResolver:
    """Resolves merge conflicts."""

    def __init__(self) -> None:
        self._conflicts: Dict[str, ConflictEntry] = {}

    def detect_conflicts(
        self,
        ours_files: Dict[str, str],
        theirs_files: Dict[str, str],
        base_files: Dict[str, str],
    ) -> Dict[str, ConflictEntry]:
        self._conflicts.clear()
        all_paths = sorted(set(list(ours_files.keys()) + list(theirs_files.keys())))

        for path in all_paths:
            ours_content = ours_files.get(path, "")
            theirs_content = theirs_files.get(path, "")
            base_content = base_files.get(path, "")

            if ours_content != theirs_content:
                ours_changed = ours_content != base_content
                theirs_changed = theirs_content != base_content
                if ours_changed and theirs_changed:
                    self._conflicts[path] = ConflictEntry(
                        path=path,
                        ours_content=ours_content,
                        theirs_content=theirs_content,
                        base_content=base_content,
                    )

        return dict(self._conflicts)

    def resolve_ours(self, path: str) -> Optional[str]:
        conflict = self._conflicts.get(path)
        if conflict:
            conflict.is_resolved = True
            return conflict.ours_content
        return None

    def resolve_theirs(self, path: str) -> Optional[str]:
        conflict = self._conflicts.get(path)
        if conflict:
            conflict.is_resolved = True
            return conflict.theirs_content
        return None

    def resolve_manual(self, path: str, content: str) -> bool:
        conflict = self._conflicts.get(path)
        if conflict:
            conflict.is_resolved = True
            return True
        return False

    def get_conflict(self, path: str) -> Optional[ConflictEntry]:
        return self._conflicts.get(path)

    def get_all_conflicts(self) -> Dict[str, ConflictEntry]:
        return dict(self._conflicts)

    def has_unresolved(self) -> bool:
        return any(not c.is_resolved for c in self._conflicts.values())

    def get_conflict_markers(self, path: str) -> str:
        conflict = self._conflicts.get(path)
        if not conflict:
            return ""
        ours_lines = conflict.ours_content.splitlines()
        theirs_lines = conflict.theirs_content.splitlines()
        lines = ["<<<<<<< OURS"]
        lines.extend(ours_lines)
        lines.append("=======")
        lines.extend(theirs_lines)
        lines.append(">>>>>>> THEIRS")
        return "\n".join(lines)

    def three_way_merge(
        self,
        base_content: str,
        ours_content: str,
        theirs_content: str,
    ) -> Tuple[str, bool]:
        base_lines = base_content.splitlines()
        ours_lines = ours_content.splitlines()
        theirs_lines = theirs_content.splitlines()

        matcher = difflib.SequenceMatcher(None, base_lines, ours_lines)
        ours_ops = matcher.get_opcodes()

        matcher = difflib.SequenceMatcher(None, base_lines, theirs_lines)
        theirs_ops = matcher.get_opcodes()

        ours_changes: Set[int] = set()
        for tag, i1, i2, j1, j2 in ours_ops:
            if tag != "equal":
                ours_changes.update(range(i1, max(i2, i1)))

        theirs_changes: Set[int] = set()
        for tag, i1, i2, j1, j2 in theirs_ops:
            if tag != "equal":
                theirs_changes.update(range(i1, max(i2, i1)))

        overlap = ours_changes & theirs_changes
        if overlap:
            return self._conflict_marker_content(base_content, ours_content, theirs_content), False

        result_lines = list(ours_lines)
        return "\n".join(result_lines), True

    @staticmethod
    def _conflict_marker_content(base: str, ours: str, theirs: str) -> str:
        lines = ["<<<<<<< OURS"]
        lines.extend(ours.splitlines())
        lines.append("=======")
        lines.extend(theirs.splitlines())
        lines.append(">>>>>>> THEIRS")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commit Manager
# ---------------------------------------------------------------------------

class CommitManager:
    """Manages commit operations."""

    def __init__(self, object_store: ObjectStore, index: Index) -> None:
        self.object_store = object_store
        self.index = index

    def create_commit(
        self,
        message: str,
        author_name: str = "User",
        author_email: str = "user@example.com",
        parent_oids: Optional[List[str]] = None,
    ) -> GitCommit:
        entries = self.index.list_entries()
        tree = GitTree()
        for path, blob_oid in entries.items():
            tree.entries[path] = GitTreeEntry(name=path, oid=blob_oid)
        tree_oid = self.object_store.store_tree(tree)

        commit = GitCommit(
            oid="",  # will be set by store_commit
            tree_oid=tree_oid,
            parent_oids=parent_oids or [],
            author_name=author_name,
            author_email=author_email,
            committer_name=author_name,
            committer_email=author_email,
            message=message,
        )
        self.object_store.store_commit(commit)
        self.index.clear()
        return commit

    def amend_commit(
        self,
        original: GitCommit,
        new_message: Optional[str] = None,
    ) -> GitCommit:
        amended = GitCommit(
            oid="",
            tree_oid=original.tree_oid,
            parent_oids=original.parent_oids,
            author_name=original.author_name,
            author_email=original.author_email,
            author_date=original.author_date,
            committer_name=original.committer_name,
            committer_email=original.committer_email,
            message=new_message or original.message,
        )
        self.object_store.store_commit(amended)
        return amended


# ---------------------------------------------------------------------------
# Branch Manager
# ---------------------------------------------------------------------------

class BranchManager:
    """Manages branch operations."""

    def __init__(self, object_store: ObjectStore) -> None:
        self.object_store = object_store
        self._branches: Dict[str, GitBranch] = {}
        self._head: str = "main"
        self._detached_head: Optional[str] = None

    def init_default_branch(self, name: str = "main") -> None:
        self._head = name
        self._branches[name] = GitBranch(name=name, target_oid="")

    def create_branch(self, name: str, start_oid: str, force: bool = False) -> GitBranch:
        if name in self._branches and not force:
            raise ValueError(f"Branch '{name}' already exists")
        branch = GitBranch(name=name, target_oid=start_oid)
        self._branches[name] = branch
        return branch

    def delete_branch(self, name: str, force: bool = False) -> bool:
        if name == self._head:
            raise ValueError(f"Cannot delete the current branch '{name}'")
        branch = self._branches.pop(name, None)
        return branch is not None

    def rename_branch(self, old_name: str, new_name: str) -> bool:
        branch = self._branches.pop(old_name, None)
        if branch is None:
            return False
        branch.name = new_name
        self._branches[new_name] = branch
        if self._head == old_name:
            self._head = new_name
        return True

    def checkout(self, name: str) -> str:
        if name not in self._branches:
            raise ValueError(f"Branch '{name}' does not exist")
        self._head = name
        self._detached_head = None
        return self._branches[name].target_oid

    def checkout_detached(self, oid: str) -> None:
        self._detached_head = oid

    def get_current_branch(self) -> str:
        return self._head

    def get_current_oid(self) -> str:
        if self._detached_head:
            return self._detached_head
        branch = self._branches.get(self._head)
        return branch.target_oid if branch else ""

    def set_branch_target(self, name: str, oid: str) -> None:
        branch = self._branches.get(name)
        if branch:
            branch.target_oid = oid

    def get_branch(self, name: str) -> Optional[GitBranch]:
        return self._branches.get(name)

    def list_branches(self, remote: bool = False) -> List[GitBranch]:
        return [
            b for b in self._branches.values()
            if b.is_remote == remote
        ]

    def list_all_branches(self) -> List[GitBranch]:
        return list(self._branches.values())


# ---------------------------------------------------------------------------
# Merge Engine
# ---------------------------------------------------------------------------

class MergeEngine:
    """Handles merge operations."""

    def __init__(
        self,
        object_store: ObjectStore,
        branch_manager: BranchManager,
        conflict_resolver: ConflictResolver,
    ) -> None:
        self.object_store = object_store
        self.branch_manager = branch_manager
        self.conflict_resolver = conflict_resolver
        self._merge_state: MergeState = MergeState.NONE
        self._merge_head: Optional[str] = None

    def merge(
        self,
        theirs_branch: str,
        message: Optional[str] = None,
    ) -> Tuple[Optional[GitCommit], Dict[str, ConflictEntry]]:
        ours_oid = self.branch_manager.get_current_oid()
        theirs_branch_obj = self.branch_manager.get_branch(theirs_branch)
        if theirs_branch_obj is None:
            raise ValueError(f"Branch '{theirs_branch}' not found")
        theirs_oid = theirs_branch_obj.target_oid

        if ours_oid == theirs_oid:
            return None, {}

        ours_commit = self.object_store.get_commit(ours_oid)
        theirs_commit = self.object_store.get_commit(theirs_oid)

        if ours_commit is None or theirs_commit is None:
            raise ValueError("Cannot find commit objects for merge")

        base_oid = self._find_merge_base(ours_oid, theirs_oid)
        base_files: Dict[str, str] = {}
        if base_oid:
            base_commit = self.object_store.get_commit(base_oid)
            if base_commit:
                base_files = self._get_tree_files(base_commit.tree_oid)

        ours_files = self._get_tree_files(ours_commit.tree_oid)
        theirs_files = self._get_tree_files(theirs_commit.tree_oid)

        conflicts = self.conflict_resolver.detect_conflicts(
            ours_files, theirs_files, base_files
        )

        merged_files = dict(ours_files)
        for path, content in theirs_files.items():
            if path not in conflicts:
                merged_files[path] = content

        if conflicts:
            self._merge_state = MergeState.CONFLICTING
            self._merge_head = theirs_oid
            return None, conflicts

        tree = GitTree()
        for path, content in merged_files.items():
            blob_oid = self.object_store.store_blob(content)
            tree.entries[path] = GitTreeEntry(name=path, oid=blob_oid)
        tree_oid = self.object_store.store_tree(tree)

        merge_msg = message or f"Merge branch '{theirs_branch}'"
        merge_commit = GitCommit(
            oid="",
            tree_oid=tree_oid,
            parent_oids=[ours_oid, theirs_oid],
            author_name=ours_commit.author_name,
            author_email=ours_commit.author_email,
            committer_name=ours_commit.committer_name,
            committer_email=ours_commit.committer_email,
            message=merge_msg,
        )
        self.object_store.store_commit(merge_commit)
        self.branch_manager.set_branch_target(
            self.branch_manager.get_current_branch(), merge_commit.oid
        )
        self._merge_state = MergeState.NONE
        return merge_commit, {}

    def _find_merge_base(self, oid1: str, oid2: str) -> Optional[str]:
        ancestors1 = self._get_all_ancestors(oid1)
        ancestors2 = self._get_all_ancestors(oid2)
        common = ancestors1 & ancestors2
        if not common:
            return None
        return max(common, key=lambda o: self._commit_timestamp(o))

    def _get_all_ancestors(self, oid: str) -> Set[str]:
        visited: Set[str] = set()
        queue = collections.deque([oid])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            commit = self.object_store.get_commit(current)
            if commit:
                for parent in commit.parent_oids:
                    queue.append(parent)
        return visited

    def _commit_timestamp(self, oid: str) -> float:
        commit = self.object_store.get_commit(oid)
        if commit:
            return commit.author_date.timestamp()
        return 0

    def _get_tree_files(self, tree_oid: str) -> Dict[str, str]:
        files: Dict[str, str] = {}
        tree = self.object_store.get_tree(tree_oid)
        if tree:
            for name, entry in tree.entries.items():
                blob = self.object_store.get_blob(entry.oid)
                if blob:
                    files[name] = blob.content
        return files

    @property
    def merge_state(self) -> MergeState:
        return self._merge_state


# ---------------------------------------------------------------------------
# Rebase Engine
# ---------------------------------------------------------------------------

class RebaseEngine:
    """Handles rebase operations."""

    def __init__(
        self,
        object_store: ObjectStore,
        branch_manager: BranchManager,
        commit_manager: CommitManager,
    ) -> None:
        self.object_store = object_store
        self.branch_manager = branch_manager
        self.commit_manager = commit_manager
        self._state: RebaseState = RebaseState.NONE
        self._original_branch: Optional[str] = None
        self._onto_oid: Optional[str] = None
        self._pending: Deque[str] = collections.deque()

    def rebase(self, onto_branch: str) -> List[GitCommit]:
        current_branch = self.branch_manager.get_current_branch()
        current_oid = self.branch_manager.get_current_oid()

        onto_branch_obj = self.branch_manager.get_branch(onto_branch)
        if onto_branch_obj is None:
            raise ValueError(f"Branch '{onto_branch}' not found")

        base_oid = self._find_merge_base(current_oid, onto_branch_obj.target_oid)
        if base_oid is None:
            base_oid = current_oid

        commits = self._get_commits_since(base_oid, current_oid)
        commits.reverse()

        self._state = RebaseState.REBASING
        self._original_branch = current_branch
        self._onto_oid = onto_branch_obj.target_oid
        self._pending = collections.deque(c.oid for c in commits)

        rebased: List[GitCommit] = []
        current_parent = onto_branch_obj.target_oid

        for commit in commits:
            new_commit = self._replay_commit(commit, current_parent)
            rebased.append(new_commit)
            current_parent = new_commit.oid

        self.branch_manager.set_branch_target(current_branch, current_parent)
        self._state = RebaseState.NONE
        return rebased

    def _replay_commit(self, original: GitCommit, new_parent: str) -> GitCommit:
        files = self._get_tree_files(original.tree_oid)
        tree = GitTree()
        for path, content in files.items():
            blob_oid = self.object_store.store_blob(content)
            tree.entries[path] = GitTreeEntry(name=path, oid=blob_oid)
        tree_oid = self.object_store.store_tree(tree)

        new_commit = GitCommit(
            oid="",
            tree_oid=tree_oid,
            parent_oids=[new_parent],
            author_name=original.author_name,
            author_email=original.author_email,
            author_date=original.author_date,
            committer_name=original.committer_name,
            committer_email=original.committer_email,
            message=original.message,
        )
        self.object_store.store_commit(new_commit)
        return new_commit

    def _find_merge_base(self, oid1: str, oid2: str) -> Optional[str]:
        ancestors1: Set[str] = set()
        queue = collections.deque([oid1])
        while queue:
            current = queue.popleft()
            if current in ancestors1:
                continue
            ancestors1.add(current)
            commit = self.object_store.get_commit(current)
            if commit:
                queue.extend(commit.parent_oids)

        queue = collections.deque([oid2])
        visited: Set[str] = set()
        while queue:
            current = queue.popleft()
            if current in ancestors1:
                return current
            if current in visited:
                continue
            visited.add(current)
            commit = self.object_store.get_commit(current)
            if commit:
                queue.extend(commit.parent_oids)
        return None

    def _get_commits_since(self, base_oid: str, tip_oid: str) -> List[GitCommit]:
        commits: List[GitCommit] = []
        current = tip_oid
        while current:
            commit = self.object_store.get_commit(current)
            if commit is None:
                break
            if current == base_oid:
                break
            commits.append(commit)
            current = commit.parent_oids[0] if commit.parent_oids else ""
        return commits

    def _get_tree_files(self, tree_oid: str) -> Dict[str, str]:
        files: Dict[str, str] = {}
        tree = self.object_store.get_tree(tree_oid)
        if tree:
            for name, entry in tree.entries.items():
                blob = self.object_store.get_blob(entry.oid)
                if blob:
                    files[name] = blob.content
        return files

    @property
    def state(self) -> RebaseState:
        return self._state


# ---------------------------------------------------------------------------
# Stash Manager
# ---------------------------------------------------------------------------

class StashManager:
    """Manages git stash operations."""

    def __init__(
        self,
        object_store: ObjectStore,
        index: Index,
        commit_manager: CommitManager,
    ) -> None:
        self.object_store = object_store
        self.index = index
        self.commit_manager = commit_manager
        self._stash_list: List[StashEntry] = []

    def push(self, message: str = "WIP") -> StashEntry:
        staged_entries = self.index.list_entries()
        tree = GitTree()
        for path, blob_oid in staged_entries.items():
            tree.entries[path] = GitTreeEntry(name=path, oid=blob_oid)

        tree_oid = self.object_store.store_tree(tree)
        stash_commit = GitCommit(
            oid="",
            tree_oid=tree_oid,
            parent_oids=[self._get_head_oid()],
            message=f"On {self._get_current_branch()}: {message}",
        )
        self.object_store.store_commit(stash_commit)

        entry = StashEntry(
            oid=str(uuid.uuid4())[:8],
            message=message,
            commit_oid=stash_commit.oid,
            index_oid=tree_oid,
        )
        self._stash_list.insert(0, entry)
        self.index.clear()
        return entry

    def pop(self, index: int = 0) -> Optional[StashEntry]:
        if index < 0 or index >= len(self._stash_list):
            return None
        entry = self._stash_list.pop(index)
        self._restore_stash(entry)
        return entry

    def apply(self, index: int = 0) -> Optional[StashEntry]:
        if index < 0 or index >= len(self._stash_list):
            return None
        entry = self._stash_list[index]
        self._restore_stash(entry)
        return entry

    def drop(self, index: int = 0) -> bool:
        if index < 0 or index >= len(self._stash_list):
            return False
        self._stash_list.pop(index)
        return True

    def clear(self) -> None:
        self._stash_list.clear()

    def list_stash(self) -> List[StashEntry]:
        return list(self._stash_list)

    def _restore_stash(self, entry: StashEntry) -> None:
        commit = self.object_store.get_commit(entry.commit_oid)
        if commit:
            tree = self.object_store.get_tree(commit.tree_oid)
            if tree:
                for name, tree_entry in tree.entries.items():
                    self.index.add(name, tree_entry.oid)

    def _get_head_oid(self) -> str:
        return ""

    def _get_current_branch(self) -> str:
        return "main"


# ---------------------------------------------------------------------------
# Tag Manager
# ---------------------------------------------------------------------------

class TagManager:
    """Manages git tag operations."""

    def __init__(self, object_store: ObjectStore) -> None:
        self.object_store = object_store

    def create_tag(
        self,
        name: str,
        target_oid: str,
        message: str = "",
        tagger_name: str = "User",
        tagger_email: str = "user@example.com",
        annotated: bool = True,
        force: bool = False,
    ) -> GitTag:
        existing = self.object_store.get_tag(name)
        if existing and not force:
            raise ValueError(f"Tag '{name}' already exists")

        tag = GitTag(
            name=name,
            oid=str(uuid.uuid4())[:8],
            target_oid=target_oid,
            tagger_name=tagger_name,
            tagger_email=tagger_email,
            message=message,
            is_annotated=annotated,
        )
        self.object_store.store_tag(tag)
        return tag

    def delete_tag(self, name: str) -> bool:
        existing = self.object_store.get_tag(name)
        if existing:
            del self.object_store._tags[name]
            return True
        return False

    def list_tags(self, pattern: Optional[str] = None) -> List[GitTag]:
        tags = self.object_store.list_tags()
        if pattern:
            regex = re.compile(pattern.replace("*", ".*"))
            tags = [t for t in tags if regex.match(t.name)]
        return sorted(tags, key=lambda t: t.tag_date, reverse=True)

    def get_tag(self, name: str) -> Optional[GitTag]:
        return self.object_store.get_tag(name)


# ---------------------------------------------------------------------------
# Git Repository (Main Facade)
# ---------------------------------------------------------------------------

class GitRepository:
    """Main facade for git porcelain operations."""

    def __init__(self, path: str = ".") -> None:
        self.path = path
        self.object_store = ObjectStore()
        self.index = Index()
        self.commit_manager = CommitManager(self.object_store, self.index)
        self.branch_manager = BranchManager(self.object_store)
        self.conflict_resolver = ConflictResolver()
        self.diff_engine = DiffEngine(self.object_store)
        self.merge_engine = MergeEngine(
            self.object_store, self.branch_manager, self.conflict_resolver
        )
        self.rebase_engine = RebaseEngine(
            self.object_store, self.branch_manager, self.commit_manager
        )
        self.stash_manager = StashManager(
            self.object_store, self.index, self.commit_manager
        )
        self.tag_manager = TagManager(self.object_store)
        self._remotes: Dict[str, RemoteInfo] = {}
        self._head_commit: Optional[GitCommit] = None
        self._working_tree: Dict[str, str] = {}
        self._author_name = "User"
        self._author_email = "user@example.com"
        self._lock = threading.Lock()

        self.branch_manager.init_default_branch("main")

    def init(self) -> None:
        self.branch_manager.init_default_branch("main")

    def add(self, path: str, content: str) -> str:
        blob_oid = self.object_store.store_blob(content)
        self.index.add(path, blob_oid)
        self._working_tree[path] = content
        return blob_oid

    def remove(self, path: str) -> None:
        self.index.remove(path)
        self._working_tree.pop(path, None)

    def commit(self, message: str) -> GitCommit:
        commit = self.commit_manager.create_commit(
            message=message,
            author_name=self._author_name,
            author_email=self._author_email,
            parent_oids=[self.branch_manager.get_current_oid()],
        )
        self._head_commit = commit
        self.branch_manager.set_branch_target(
            self.branch_manager.get_current_branch(), commit.oid
        )
        return commit

    def status(self) -> StatusResult:
        result = StatusResult(branch=self.branch_manager.get_current_branch())
        index_entries = self.index.list_entries()
        head_files = self._get_head_files()

        for path in index_entries:
            if path not in head_files:
                result.entries.append(StatusEntry(path=path, index_status="A"))
                result.staged_count += 1
            elif index_entries[path] != head_files[path]:
                result.entries.append(StatusEntry(path=path, index_status="M"))
                result.staged_count += 1

        for path in self._working_tree:
            if path not in index_entries:
                result.entries.append(StatusEntry(path=path, working_tree_status="??"))
                result.untracked_count += 1
            elif path in head_files and index_entries.get(path) == head_files.get(path):
                wt_blob = self.object_store.store_blob(self._working_tree[path])
                if wt_blob != head_files[path]:
                    result.entries.append(StatusEntry(path=path, working_tree_status="M"))
                    result.unstaged_count += 1

        return result

    def log(self, max_count: int = 20, branch: Optional[str] = None) -> List[LogEntry]:
        if branch:
            branch_obj = self.branch_manager.get_branch(branch)
            if branch_obj is None:
                return []
            start_oid = branch_obj.target_oid
        else:
            start_oid = self.branch_manager.get_current_oid()

        entries: List[LogEntry] = []
        visited: Set[str] = set()
        current = start_oid

        while current and len(entries) < max_count:
            if current in visited:
                break
            visited.add(current)
            commit = self.object_store.get_commit(current)
            if commit is None:
                break

            refs: List[str] = []
            for b in self.branch_manager.list_all_branches():
                if b.target_oid == current:
                    refs.append(b.name)
            for tag in self.tag_manager.list_tags():
                if tag.target_oid == current:
                    refs.append(f"tag: {tag.name}")

            entries.append(LogEntry(
                oid=commit.oid[:8],
                author_name=commit.author_name,
                author_email=commit.author_email,
                author_date=commit.author_date,
                committer_name=commit.committer_name,
                message=commit.message.split("\n")[0],
                refs=refs,
            ))
            current = commit.parent_oids[0] if commit.parent_oids else ""

        return entries

    def diff(self, staged: bool = False) -> DiffResult:
        head_files = self._get_head_files()
        if staged:
            index_entries = self.index.list_entries()
            all_paths = sorted(set(list(head_files.keys()) + list(index_entries.keys())))
            result = DiffResult()
            for path in all_paths:
                old_oid = head_files.get(path)
                new_oid = index_entries.get(path)
                entry = self.diff_engine.diff_files(old_oid, new_oid, path, path)
                result.entries.append(entry)
                result.total_additions += entry.additions
                result.total_deletions += entry.deletions
            return result
        else:
            current_oid = self.branch_manager.get_current_oid()
            current_commit = self.object_store.get_commit(current_oid)
            tree_oid = current_commit.tree_oid if current_commit else None
            return self.diff_engine.diff_trees(tree_oid, tree_oid)

    def branch(self, name: str, start_point: Optional[str] = None) -> GitBranch:
        start_oid = start_point or self.branch_manager.get_current_oid()
        return self.branch_manager.create_branch(name, start_oid)

    def checkout(self, name: str) -> str:
        return self.branch_manager.checkout(name)

    def merge(self, branch: str, message: Optional[str] = None) -> Tuple[Optional[GitCommit], Dict[str, ConflictEntry]]:
        return self.merge_engine.merge(branch, message)

    def rebase(self, onto: str) -> List[GitCommit]:
        return self.rebase_engine.rebase(onto)

    def cherry_pick(self, commit_oid: str) -> Optional[GitCommit]:
        commit = self.object_store.get_commit(commit_oid)
        if commit is None:
            raise ValueError(f"Commit '{commit_oid}' not found")
        files = self._get_tree_files(commit.tree_oid)
        for path, content in files.items():
            self.add(path, content)
        return self.commit(f"Cherry-pick: {commit.message.split(chr(10))[0]}")

    def stash_push(self, message: str = "WIP") -> StashEntry:
        return self.stash_manager.push(message)

    def stash_pop(self, index: int = 0) -> Optional[StashEntry]:
        return self.stash_manager.pop(index)

    def stash_list(self) -> List[StashEntry]:
        return self.stash_manager.list_stash()

    def tag(self, name: str, message: str = "", annotated: bool = True) -> GitTag:
        return self.tag_manager.create_tag(
            name=name,
            target_oid=self.branch_manager.get_current_oid(),
            message=message,
            tagger_name=self._author_name,
            tagger_email=self._author_email,
            annotated=annotated,
        )

    def add_remote(self, name: str, url: str) -> RemoteInfo:
        remote = RemoteInfo(name=name, url=url)
        self._remotes[name] = remote
        return remote

    def remove_remote(self, name: str) -> bool:
        return self._remotes.pop(name, None) is not None

    def list_remotes(self) -> List[RemoteInfo]:
        return list(self._remotes.values())

    def config(self, key: str, value: Optional[str] = None) -> Optional[str]:
        if key == "user.name":
            if value:
                self._author_name = value
            return self._author_name
        elif key == "user.email":
            if value:
                self._author_email = value
            return self._author_email
        return None

    def _get_head_files(self) -> Dict[str, str]:
        current_oid = self.branch_manager.get_current_oid()
        commit = self.object_store.get_commit(current_oid)
        if commit is None:
            return {}
        return self._get_tree_files(commit.tree_oid)

    def _get_tree_files(self, tree_oid: str) -> Dict[str, str]:
        files: Dict[str, str] = {}
        tree = self.object_store.get_tree(tree_oid)
        if tree:
            for name, entry in tree.entries.items():
                blob = self.object_store.get_blob(entry.oid)
                if blob:
                    files[name] = blob.content
        return files
