# GitHub publication

Create an empty repository in the intended GitHub account and upload the contents of this folder, including hidden `.github` files. Alternatively:

```sh
git init
git add .
git commit -m "Prepare TWINOVA research prototype"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace YOUR_REPOSITORY_URL with your repository URL. No remote account or repository has been modified by preparing this package.

Add the repository URL to the project submission only after checking the account, visibility, release and competition rules. Do not claim GitHub CI has passed until a real workflow run completes. Local evidence is in `validation/`.

Set the code license only after confirming contributor and third-party rights. No new license is assigned by this cleanup. Retain source notices and licenses. Model/data origin remains documented for scientific reproducibility.
