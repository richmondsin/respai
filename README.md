# Responsible AI Project

The **Responsible AI** project aims to provide tools, code, and guidelines that help developers create AI systems in a responsible and ethical manner. The core goal is to ensure that AI systems adhere to principles of fairness, and accountability. 

## Project Scope

This project involves two primary tasks:

### 1. **Evaluation of LionGuard Model**
   - Assess the performance and effectiveness of the LionGuard model in detecting unsafe content within a specified context.

### 2. **Development of Relevant Prompt Detection Tool**
   - Create a tool that filters out off-topic queries before reaching an LLM, ensuring that only relevant prompts are processed, improving the efficiency and resource usage of the LLM-powered chatbots.

---

## Project Structure

This repository contains different files and directories that correspond to various sections and tasks within the Responsible AI project. Below is a description of the key components:

### **Main Sections and Files**

- **`off_topic_detector/`**: Contains the code for the off-topic prompt detection tool.
- **`section1.md`**: Contains answers and explanations for Section 1 of the project.
- **`section1q2.ipynb`** / **`section1q2.py`**: Code and notebook files for answering Question 2 of Section 1.
- **`section2.md`**: Contains answers and explanations for Section 2 of the project.
- **`section2q1.ipynb`** / **`section2q1.py`**: Code and notebook files for answering Question 1 of Section 2.
- **`venv.html`**: Detailed instructions on how to set up a virtual environment. You can refer to this file to ensure your development environment is properly isolated.

---

## LionGuard Model for SGHateCheck

The repository contains files to run and evaluate the **LionGuard** model on the SGHateCheck dataset. This model focuses on detecting harmful or inappropriate content based on specific guidelines.

### Key Files:
- **`config.json`**: Configuration file that contains the necessary settings for the LionGuard model.
- **`inference.py`**: Python script providing inference functionality for running LionGuard on the SGHateCheck dataset.
- **`sghatecheckdata/`**: Folder containing the SGHateCheck dataset, which includes various test cases designed to evaluate LionGuard’s performance.

---

## Requirements

Before running the code, it's recommended to create a virtual environment. You can find the detailed instructions in the [virtual environment setup guide](venv.html).

---

(**Proudly written by me for use in my teaching class for DSA3101: Data Science in Practice**)

### Install dependencies:

Once your virtual environment is set up and activated, install the necessary dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```