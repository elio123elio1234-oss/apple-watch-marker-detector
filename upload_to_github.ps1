# GitHub Upload Script - Manual Method
# =====================================
# Run this script to upload your project to GitHub

Write-Host "🚀 Apple Watch Marker Detector - GitHub Upload Script" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
$currentDir = Get-Location
if ($currentDir.Path -notlike "*Marker Detector*") {
    Write-Host "⚠️  Not in Marker Detector directory. Navigating..." -ForegroundColor Yellow
    Set-Location "c:\Users\elio1\Desktop\Marker Detector"
}

# Check git status
Write-Host "📊 Checking Git status..." -ForegroundColor Green
git status --short

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "STEP 1: Create GitHub Repository" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Open: https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: apple-watch-marker-detector" -ForegroundColor White
Write-Host "3. Choose Public or Private" -ForegroundColor White
Write-Host "4. DON'T check 'Add README' (we already have one)" -ForegroundColor White
Write-Host "5. Click 'Create repository'" -ForegroundColor White
Write-Host ""

# Open GitHub new repo page
$openBrowser = Read-Host "Open GitHub new repository page now? (Y/n)"
if ($openBrowser -ne 'n') {
    Start-Process "https://github.com/new"
    Write-Host "✅ Opening browser..." -ForegroundColor Green
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "STEP 2: Enter Your GitHub Details" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$username = Read-Host "Enter your GitHub username"
$repoName = Read-Host "Enter repository name (default: apple-watch-marker-detector)"

if ([string]::IsNullOrWhiteSpace($repoName)) {
    $repoName = "apple-watch-marker-detector"
}

$repoUrl = "https://github.com/$username/$repoName.git"

Write-Host ""
Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
Write-Host ""

$confirm = Read-Host "Is this correct? (Y/n)"
if ($confirm -eq 'n') {
    Write-Host "❌ Cancelled. Please run the script again." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "STEP 3: Connecting to GitHub..." -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# Remove existing remote if it exists
Write-Host "🔗 Removing old remote (if exists)..." -ForegroundColor Green
git remote remove origin 2>$null

# Add new remote
Write-Host "🔗 Adding remote: $repoUrl" -ForegroundColor Green
git remote add origin $repoUrl

# Verify remote
$remote = git remote -v
Write-Host "✅ Remote configured:" -ForegroundColor Green
Write-Host $remote
Write-Host ""

# Rename branch to main
Write-Host "🌿 Renaming branch to 'main'..." -ForegroundColor Green
git branch -M main

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "STEP 4: Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "ℹ️  You may be prompted for GitHub credentials:" -ForegroundColor Cyan
Write-Host "   - Username: $username" -ForegroundColor White
Write-Host "   - Password: Use Personal Access Token (not your password!)" -ForegroundColor White
Write-Host ""
Write-Host "   Get token from: https://github.com/settings/tokens" -ForegroundColor White
Write-Host "   Required scope: 'repo' (full control)" -ForegroundColor White
Write-Host ""

$push = Read-Host "Ready to push? (Y/n)"
if ($push -eq 'n') {
    Write-Host "⏸️  Push cancelled. You can push manually later with:" -ForegroundColor Yellow
    Write-Host "   git push -u origin main" -ForegroundColor White
    exit
}

Write-Host ""
Write-Host "📤 Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host "✅ SUCCESS! Project uploaded to GitHub!" -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Repository URL:" -ForegroundColor Cyan
    Write-Host "   https://github.com/$username/$repoName" -ForegroundColor White
    Write-Host ""
    
    $openRepo = Read-Host "Open repository in browser? (Y/n)"
    if ($openRepo -ne 'n') {
        Start-Process "https://github.com/$username/$repoName"
    }
    
    Write-Host ""
    Write-Host "📝 Next steps:" -ForegroundColor Yellow
    Write-Host "   • Share: Send the URL to others" -ForegroundColor White
    Write-Host "   • Clone: git clone https://github.com/$username/$repoName.git" -ForegroundColor White
    Write-Host "   • Update: git add . && git commit -m 'message' && git push" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
    Write-Host "❌ Push failed. Common issues:" -ForegroundColor Red
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Red
    Write-Host ""
    Write-Host "1. Repository doesn't exist on GitHub yet" -ForegroundColor Yellow
    Write-Host "   → Create it at: https://github.com/new" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Wrong username or repository name" -ForegroundColor Yellow
    Write-Host "   → Check: https://github.com/$username/$repoName" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Authentication failed" -ForegroundColor Yellow
    Write-Host "   → Use Personal Access Token as password" -ForegroundColor White
    Write-Host "   → Get from: https://github.com/settings/tokens" -ForegroundColor White
    Write-Host ""
    Write-Host "Try again with: git push -u origin main" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
