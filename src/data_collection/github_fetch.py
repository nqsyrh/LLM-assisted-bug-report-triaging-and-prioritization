import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

# Load secrets from the .env file
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}


def fetch_issues(owner, repo, state="closed", label="bug", max_pages=10):
    """
    Fetches bug-labeled issues from a GitHub repo.
    Example: owner='microsoft', repo='vscode'
    """
    all_issues = []
    page = 1

    while page <= max_pages:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": state,
            "labels": label,
            "per_page": 100,
            "page": page
        }
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        data = response.json()
        if not data:
            break  # no more issues to fetch

        for issue in data:
            # Skip pull requests (GitHub API mixes them in with issues)
            if "pull_request" in issue:
                continue

            all_issues.append({
                "id": issue["id"],
                "title": issue["title"],
                "body": issue["body"],
                "labels": [l["name"] for l in issue["labels"]],
                "state": issue["state"],
                "created_at": issue["created_at"],
                "closed_at": issue["closed_at"],
                "comments": issue["comments"],
                "repo": f"{owner}/{repo}"
            })

        print(f"Page {page}: collected {len(data)} issues, total so far: {len(all_issues)}")
        page += 1
        time.sleep(1)  # be polite to the API, avoid rate limits

    return all_issues


if __name__ == "__main__":
    if not TOKEN:
        print("ERROR: No GITHUB_TOKEN found.")
        print("Check that your .env file exists and contains a line like:")
        print("GITHUB_TOKEN=your_token_here")
        exit()

    # Example repos to pull bug reports from — change these anytime
    repos = [
        ("microsoft", "vscode"),
        ("facebook", "react"),
        ("kubernetes", "kubernetes")
    ]

    all_data = []
    for owner, repo in repos:
        print(f"\nFetching issues from {owner}/{repo}...")
        issues = fetch_issues(owner, repo, max_pages=5)  # 5 pages ~ 500 issues per repo
        all_data.extend(issues)

    df = pd.DataFrame(all_data)
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/github_bug_reports.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDone! Saved {len(df)} bug reports to {output_path}")
