#!/bin/bash

# Step 0: Load the .env file to get the GitHub token and username
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found! Please create a .env file with your GITHUB_TOKEN and GITHUB_USERNAME."
  exit 1
fi

# Check if GITHUB_TOKEN and GITHUB_USERNAME are set
if [[ -z "$GITHUB_TOKEN" || -z "$GITHUB_USERNAME" ]]; then
  echo "GITHUB_TOKEN or GITHUB_USERNAME not set in .env file. Please add them."
  exit 1
fi

# Step 1: Create a new Python virtual environment with user input
read -p "Enter the name for your virtual environment: " venv_name
python3 -m venv "$venv_name"

# Step 2: Activate the virtual environment
source "$venv_name/bin/activate"

# Step 3: Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Step 4: Initialize a Git repository
git init

# Step 5: Create a .gitignore file to ignore the virtual environment folder
echo "$venv_name/" > .gitignore
echo ".env" > .gitignore
echo ".gitignore created with virtual environment exclusion."

# Step 6: Connect to GitHub and create a new public repository
read -p "Enter the name for your GitHub repository: " repo_name

# Use GitHub API to create the repository as public
create_repo_response=$(curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
  -d "{\"name\": \"$repo_name\", \"private\": false}" \
  https://api.github.com/user/repos)

# Check if the repository creation was successful
if echo "$create_repo_response" | grep -q '"full_name"'; then
  echo "Public repository '$repo_name' created successfully on GitHub."
else
  echo "Failed to create repository. Please check your token and permissions."
  exit 1
fi

# Step 7: Add files to the repository, commit, and push to GitHub
git add .
git commit -m "Initial commit"
git remote add origin "https://github.com/$GITHUB_USERNAME/$repo_name.git"
git branch -M main
git push -u origin main

echo "Project pushed to GitHub repository '$repo_name'."
echo "Setup complete!"
