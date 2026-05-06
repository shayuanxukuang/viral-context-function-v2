# GitHub Upload Guide

Use this directory as the GitHub repository root:

```text
artifacts/github_release/viral-context-function-v2_20260506
```

Do not initialize Git from the full `<LOCAL_WORKSPACE>` workspace. The full workspace contains raw data, runs, databases, server sync files, and generated artifacts that should not be committed.

## 1. Create The Repository On GitHub

In GitHub:

1. Click **New repository**.
2. Repository name: `viral-context-function-v2` or `virufunc-v2`.
3. Choose private first if the manuscript is still under review.
4. Do not add a README, `.gitignore`, or license on GitHub; this folder already contains the local files.

## 2. Initialize And Push

From PowerShell:

```powershell
cd <LOCAL_WORKSPACE>\artifacts\github_release\viral-context-function-v2_20260506
git init
git add .
git commit -m "Initial ViruFunc V2 reproducibility release"
git branch -M main
git remote add origin https://github.com/<YOUR_USER_OR_ORG>/<REPO_NAME>.git
git push -u origin main
```

Replace `<YOUR_USER_OR_ORG>` and `<REPO_NAME>` with the actual GitHub path.

## 3. Keep Manuscript Files Out Of The Public Repository

This public repository is code/data-source only. Do not upload these files until the authors intentionally release the article package:

- `manuscript/`
- `main.tex`
- `main.pdf`
- `supplement.tex`
- `supplement.pdf`
- article figure PDFs/PNGs
- full supplementary package zip if it includes article figures or draft text

## 4. Final Public-Release Checks

Before making the repository public:

- confirm `LICENSE` is MIT;
- check that no raw data, checkpoints, pLM embeddings, predicted PDB archives, or Foldseek databases were added;
- check that no manuscript text, compiled manuscript PDFs, or article figures were added;
- confirm manuscript wording uses `prioritized`, `supported`, `ambiguous`, and `hypothesis`, not experimental validation language.
