# How to Contribute For Students

> [!note]
> This document provides contribution guidelines for students working on their theses. Please follow these steps
> carefully to ensure smooth collaboration.

- [How to Contribute For Students](#how-to-contribute-for-students)
    - [Workflow Overview](#workflow-overview)
    - [Step-by-Step Instructions](#step-by-step-instructions)
        - [1. Keeping Your Branch Updated](#1-keeping-your-branch-updated)
        - [2. Committing Changes](#2-committing-changes)
        - [3. Working With Issues](#3-working-with-issues)
        - [4. Adding Examples, Tests, and Documentation](#4-adding-examples-tests-and-documentation)
        - [5. Preparing the Final Version](#5-preparing-the-final-version)
        - [6. Submitting Your Work](#6-submitting-your-work)
    - [Language and Conduct](#language-and-conduct)
    - [Best Practices](#best-practices)
    - [Instructions for Adding an Example](#instructions-for-adding-an-example)
    - [Instructions for Adding a Test Case](#instructions-for-adding-a-test-case)
    - [Instructions for Adding Documentation](#instructions-for-adding-documentation)


## Workflow Overview

1. **GitHub Username**:
    - Before starting your work, please create a GitHub account if you don’t have one already and send me your GitHub
      username. This will be necessary to set up your branch and reflect your contributions to the code.

2. **Branch Creation or Forking**:
    - Ahmad will create a branch for your work once your thesis starts. Alternatively, you can **fork** the repository and work on your own fork. Working on a fork is often preferred as it gives you more control over your environment. 
    - If you work on the main repository, Ahmad will set up a branch linked to an issue to track your progress. Our meetings are reserved for content discussion. The discussions in the issue are only related to code errors.
    - You **do not** create branches yourself on the main repository. Also, **do not** work on other student branches.

3. **Creating Issues**:
    - Once your thesis starts, create an issue to describe the feature, bug fix, or enhancement you plan to implement.
      This helps us track contributions and avoids duplicate work. Keep the description abstract and add a few
      checkboxes listing what you want to add. You do not need to explicitly mention the methods. Keep it abstract,
      mentioning the purpose or benefits gained.
    - Go to the **Issues** tab in the [FTIO repository](https://github.com/tuda-hpclab/ftio).
    - Click **New Issue** and provide a clear title and description.
    - Label the issue appropriately as `feature` and include call it `feature...`.
    - Once you push commits, some of them should address the issue.
    - You should regularly update the issue (at least every few weeks).

4. **Development Workflow**:
    - Work only on the branch assigned to you or on your fork.
    - Regularly pull updates from the `development` branch of the main repository and merge them into your branch to
      stay up-to-date (at least every two weeks).
    - Build FTIO with the debug flag so that changes made in the directory are directly visible to the command line call
      without reinstalling FTIO. For that call:

      ```bash
      cd <ftio_repo>
      make debug #creates a virtual environment and install FTIO
      # or
      pip install -e .
      ```
    - follow the [best practice guidelines](#best-practices) to ensure code compatibility and a smooth experience for
      all developers

5. **Merging Restrictions**:
    - You are **not allowed** to merge into the `development` or `main` branches.

6. **Adding Examples, Tests, and Documentation**
    - Any new feature or significant change **should include**:
        - An [example](#instructions-for-adding-an-example) (or explanation in the doc on how to use the feature)
        - A [test case](#instructions-for-adding-a-test-case)
        - A [Documentation](#instructions-for-adding-documentation)
    - Examples should be minimal, reproducible, and placed in `examples/`. Alternatively, the documentation shows how to use the new feature
    - Test cases must pass and follow existing structures.
    - Documentation should be clear and concise.

7. **Final Submission**:
   - Submit your work as documented [here](#6-submitting-your-work)
  
---

## Step-by-Step Instructions

### 1. Keeping Your Branch Updated

- Periodically update your branch with changes from the main `development` branch:
  
  **If you work on the main repository:**
  ```bash
  git checkout development
  git pull origin development
  git checkout your-branch
  
  # Either merge or rebase with development
  git merge development
  # Or rebase for a linear history
  git rebase development
  ```
  
  **If you work on a fork:**
  ```bash
  # First, add the main repo as a remote (one-time setup)
  git remote add upstream https://github.com/tuda-hpclab/ftio.git
  git fetch upstream
  git checkout your-branch
  git merge upstream/development
  ```
- Resolve any merge conflicts promptly and test your work.

### 2. Committing Changes

- Make frequent commits with clear and descriptive messages. Ideally, once you are finished working on an aspect, you
  create a commit for it. Before committing, always check and fix the style:

```bash
 cd <ftio_repo>
 make check_style
```

Afterward, stage the files you want to push

```bash
  git add file1.py ./some_location/file2.py
  ```

  ```bash
  git commit -m "FTIO: Add feature X to improve performance"
  ```

Afterwards, push your changes from *your* local branch to *your* remote:

  ```bash
  # You are on your-branch (check using git branch -a)
  git push
  ```

> [!note]
> Avoid using overly short or overly descriptive commit messages, such as 'update' or 'code cleaned'.

> [!note]
> Always check that your commits align with the style used. For that, call 'make check_style' in the root directory of
> the repo. An easier, more automated way is explained [here](#commit-with-style).

- You are **not allowed** to merge into the `development` or `main` branches.

### Commit With Style

You can make things even easier regarding the style by combining this with a Git hook. For that
Create a file called `pre-commit` under the root directory's .git folder (`.git/hooks/pre-commit`).

```bash
cd <ftio_repo>
touch .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Now add the following content to it:

```bash
#!/bin/bash

echo "Running black formatter..."
black . 
isort .
```

That's it. Now every time you commit, the code is formatted correctly.

### 3. Working With Issues

- **Link your work**: Link commits to your assigned issue in the main repository.
- **If working on a fork**: To link your fork's commits to an issue in the main repository (`tuda-hpclab/ftio`), use the full repository reference in your commit messages or pull request descriptions.
  - **Commit messages**: Include `tuda-hpclab/ftio#<issue_number>` in your commit message. For example:
    ```bash
    git commit -m "Add feature X (ref: tuda-hpclab/ftio#123)"
    ```
  - **Pull Request descriptions**: Use keywords like `Closes tuda-hpclab/ftio#123` or `Fixes tuda-hpclab/ftio#123` to automatically link and eventually close the issue.
- **Update issues regularly**: Post updates or ask code-related questions directly in the issue on the main repository.
- **Use issue discussion**: Use issue discussion only for code-related questions. For content discussion, wait for the scheduled meetings.

### 4. Adding Examples, Tests, and Documentation

- Add an example, test case, and documentation for each significant change.
- Follow section instructions for:
  - [Examples](#instructions-for-adding-an-example)
  - [Tests](#instructions-for-adding-a-test-case)
  - [Documentation](#instructions-for-adding-documentation)

### 5. Preparing the Final Version

- Ensure all tests pass, style checks succeed, examples and documentation are complete.
- Clean up unused code or debug output.

### 6. Submitting Your Work
 - Make sure all above points are addressed (especially **4** and **5**)
 - **Pull Request Deadline**: You must create a **pull request (PR)** to merge your work into the `development` branch **at least two weeks before the end of your thesis**. This allows for early feedback and review. You can (and should) continue to update the PR with new commits until the final submission of your thesis.
 - When you are done with the thesis, tag your final version:
     ```bash
     git tag -a BA_time_window_adaptation -m "Bachelor Thesis: implementation and evaluation of time window adaptation"
     git push --tags
     ```
 - Include a summary of your work in the PR and link it to your issue for reference.
 - Don’t forget to add yourself to the [list of contributors](/docs/contributing.md#list-of-contributors).


---

## Language and Conduct

1. **Appropriate Language**:
    - Use professional, respectful, and clear language in all commit messages, comments, and documentation.
    - Avoid using slang, jokes, or informal phrases that could be misinterpreted or deemed inappropriate.

2. **Avoid Bad Language**:
    - Refrain from using any offensive, vulgar, or discriminatory language in any form. This applies to commit messages,
      comments, documentation, or communication within the team.

3. **Be Respectful**:
    - Show courtesy when discussing issues, asking questions, or providing feedback. Collaborative communication is key
      to the success of the project.

4. **Constructive Feedback**:
    - Provide helpful suggestions or feedback without criticism that could discourage others.

5. **Gender-Neutral and Inclusive Language**:
    - Ensure that all language used in the project, including commit messages, documentation, and communication, is
      gender-neutral and inclusive. Avoid using gendered pronouns or assumptions, and instead use terms that are
      respectful and inclusive of all genders. This helps create a welcoming environment for everyone involved in the
      project.

---

## Best Practices

- **External dependencies**: Some features in the project rely on optional external dependencies (e.g., `fastdtw`,
  `dash`, `dash-extensions`), which are not essential, but provide an optimized version or additional functionalities. If
  these dependencies are not available, the code should fall back and continue to function without those specific
  features as described [here]
- **Stay Updated**: Regularly pull changes from `development` to avoid large merge conflicts. Also, keep the issue
  updated.
- **Communicate**: Reach out if you encounter issues or need clarification.
- **Test Thoroughly**: Ensure your work doesn’t break existing functionality. Do **not** rename or reformat entire
  documents, except if you created them from scratch. Regularly test your code with
  your [test case](/docs/students_contribute.md#instructions-for-adding-a-test-case).
- **Document Changes**: Write clear comments and update related documentation as needed.

---

## Instructions for External Dependencies:

Some features in the project rely on optional external dependencies (e.g., `fastdtw`). If these dependencies are not
available, and if they are not essential, the code should fall back and continue to function without those specific
features.

Example of how to handle optional dependencies:

```python
import numpy as np
import importlib.util
from scipy.spatial.distance import euclidean

# Check if fastdtw is available
FASTDTW_AVAILABLE = importlib.util.find_spec("fastdtw") is not None
if FASTDTW_AVAILABLE:
    from fastdtw import fastdtw


## Call DTW function
def fdtw(s1, s2):
    if FASTDTW_AVAILABLE:
        return fastdtw(s1, s2, dist=euclidean)
    else:
        return fill_dtw_cost_matrix(s1, s2)


## Fill DTW Cost Matrix using NumPy
def fill_dtw_cost_matrix(s1, s2):
    ...
```

> [!note]
> External dependencies should be avoided as much as possible, as each additional dependency introduces a potential risk
> for the code to break. Only include dependencies that are essential for the core functionality of the project.
> Optional
> dependencies should be handled in a way that the code can continue functioning without them, using fallbacks where
> possible.

## Creating New Files and Modules

To keep the codebase maintainable and collaboration-friendly, we recommend organizing your work into **cohesive modules
** rather than placing everything into a single file or a monolithic script.

### ✅ Why modularize?

- **Avoid merge conflicts**: Isolating related functionality into separate files reduces the chances of developers
  working on the same file at the same time.
- **Improve readability**: Smaller, focused modules are easier to read, understand, and review.
- **Enhance reusability**: Modular code is easier to reuse across different parts of the project.
- **Enable testing**: Individual modules and their functions can be unit tested more effectively.

---

### ⚠️ But don’t go overboard

While modularization is good, **creating too many small or overly granular files** can:

- Make the project harder to navigate.
- Introduce unnecessary complexity in the import structure.
- Obscure the overall logic of the system.

**Guideline**: Group logically related functions or classes into a single module. Avoid creating new files for each
utility or tiny helper unless it serves a clear organizational purpose.

---

### 🧾 Module Documentation and Licensing

Every new module should start with a module-level docstring to explain its purpose, authorship, and license. Below is a
template you should use:

```python
"""
Example Description: 
This module provides helper functions for setting up and managing the JIT environment.
It includes utilities for checking ports, parsing options, allocating resources,
handling signals, and managing components like FTIO, GekkoFS, and Cargo.

Author: Your Name  
Copyright (c) 2024-2026 TU Darmstadt, Germany  
Date: <Month Year>

Licensed under the BSD 3-Clause License.  
For more information, see the LICENSE file in the project root:
https://github.com/tuda-hpclab/ftio/blob/main/LICENSE
"""

```

---

## Instructions for Adding an Example

To demonstrate how to use `FTIO` with you new feature, you should add a relevant example under the `examples` directory:

1. **Create a new example script** in the `examples` folder.
2. **Ensure the example is clear**, easy to understand, and includes proper usage of `FTIO`.
3. **Push and commit** your changes:

    ```bash
    git add examples/your_example.py
    git commit -m "FTIO: Add example usage of feature XXX"
    ```

---

## Instructions for Adding a Test Case

To add a test case for verifying your changes, follow these steps:

1. **Write a new test script** in the `test` directory to check for the desired functionality of `FTIO`.
2. **Ensure the test is clear** and isolates the tested functionality.
3. **Push and commit** your changes:

    ```bash
    git add test/test_example.py
    git commit -m "Add test case for FTIO read/write functionality"
    ```

4. **Regularly test your testcase**:

    ```bash
    cd <ftio_repo>
    make test
    ```
5. **Regularly check that you code is conform with the style**:
    ```bash
    make check_style
   ```

---

## Instructions for Adding Documentation

To ensure proper documentation for your work, follow these steps:

1. **Write a new documentation file** or update an existing one in the `docs` directory.
2. **Include relevant details**, such as how to use the example, the purpose of the test cases, and any other important
   information.
3. **Push and commit** your changes:

    ```bash
    git add docs/example_usage.md
    git commit -m "FTIO: Add documentation for feature XXX"
    ```

4. If you made changes to the command line arguments, please update the usage section in the [readme](/README.md#usage).

---

We sincerely thank our [contributors](../CONTRIBUTORS.md) for their valuable contributions!

Thanks a lot for your contribution! I look forward to seeing the progress we will make together. Let's make this a great
experience! 🚀🚀
