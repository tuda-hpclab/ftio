"""
This file provides a multiprocessing-safe queue implementation for managing file paths,
ensuring that files are processed exactly once across multiple worker processes.

The FileQueue class uses shared memory data structures to enable safe concurrent access
and includes methods for adding, retrieving, and marking files as processed or failed.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: v0.0.9
Date: Apr 2025

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from __future__ import annotations

from multiprocessing import Manager

from ftio.api.gekkoFs.gekko_helper import get_modification_time


class FileQueue:
    """
    A multiprocessing-safe queue for files and folders, ensuring that each item
    is processed exactly once across multiple worker processes.

    Files are only added if their folder is not already in the queue.
    """

    def __init__(self) -> None:
        """
        Initialize the FileQueue with shared memory data structures.
        """
        m = Manager()
        self._queue = m.list()  # Files/folders to process
        self._ignore = m.list()  # List of (timestamp, filename) tuples
        self._lock = m.Lock()

    def put(self, item: str) -> None:
        """
        Add a file or folder to the queue.

        - Folders are always added if not already present.
        - Files are added only if their folder is not already in the queue.

        Args:
            item: The file or folder path to queue.
        """
        with self._lock:
            if "." in item:  # simple file check: has an extension
                folder = item.rsplit("/", 1)[0] if "/" in item else ""
                if folder not in self._queue and item not in self._queue:
                    self._queue.append(item)
            else:  # folder
                if item not in self._queue:
                    self._queue.append(item)

    def get_next(self) -> str | None:
        """
        Get the next item (file or folder) to process.

        Returns:
            A path string if available, otherwise None.
        """
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None

    def mark_done(self, item: str) -> None:
        """
        Mark an item as successfully processed and remove it from the queue.

        Args:
            item: The path to mark as done.
        """
        with self._lock:
            if item in self._queue:
                self._queue.remove(item)

    def mark_failed(self, item: str) -> None:
        """
        Mark an item as failed and remove it from the queue.

        Args:
            item: The path to mark as failed.
        """
        with self._lock:
            if item in self._queue:
                self._queue.remove(item)

    def in_progress(self, item: str) -> bool:
        """
        Check if an item is currently in the queue.

        Args:
            item: The path to check.

        Returns:
            True if the item exists in the queue, otherwise False.
        """
        with self._lock:
            return item in self._queue

    def put_ignore(self, args, item: str):
        """
        Add a file or folder to the ignore list with the current timestamp.
        """
        timestamp = get_modification_time(args, item)
        with self._lock:
            self._ignore.append((timestamp, item))

    def is_ignored(self, args, itempath: str) -> bool:
        """
        Check if a file is in the ignore list:
          1. First check if the filename is present at all.
          2. If yes, compute the current modification time.
          3. Return True if (timestamp, filepath) is in the ignore list.
        """
        with self._lock:
            # Step 1: quick check if file name exists in ignore list
            itemname = [fname for _, fname in self._ignore]
            if itempath not in itemname:
                return False

        # Step 2: only now get the timestamp (expensive operation)
        timestamp = get_modification_time(args, itempath)

        # Step 3: check full (timestamp, filepath) pair
        with self._lock:
            return (timestamp, itempath) in self._ignore

    def __len__(self) -> int:
        """
        Get the number of items currently in the queue.

        Returns:
            The number of items in the queue.
        """
        return len(self._queue)

    def __str__(self) -> str:
        """
        Get a string representation of the queue.

        Returns:
            A string representation of the queue contents.
        """
        return f"FileQueue(queue size={len(self._queue)}, contents={list(self._queue)})"
