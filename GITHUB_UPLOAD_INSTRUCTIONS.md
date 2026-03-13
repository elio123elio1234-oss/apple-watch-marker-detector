# 🚀 העלאה לגיטהאב - הוראות שלב אחר שלב

## המצב הנוכחי ✅
הכל מוכן! כבר עשיתי:
- ✅ `git init` - אתחול repository
- ✅ `git add .` - הוספת כל הקבצים
- ✅ `git commit` - commit ראשון עם כל הפרויקט
- ✅ התקנת GitHub CLI (מחכה לרענון)

---

## שיטה 1: דרך הדפדפן (הכי פשוט!) 🌐

### 1️⃣ צור repository חדש ב-GitHub
1. לך ל-https://github.com/new
2. **Repository name:** `apple-watch-marker-detector` (או כל שם שתרצה)
3. **Description:** `Real-time 6DOF pose tracking using dual-color SplitScreen marker on Apple Watch OLED display`
4. בחר **Public** או **Private** (לפי העדפתך)
5. **אל תסמן** "Add README" (כבר יש לנו!)
6. לחץ **"Create repository"**

### 2️⃣ חבר את הפרויקט המקומי ל-GitHub
GitHub יציג לך הוראות. העתק את השורות האלה (עם ה-URL שלך!) והדבק ב-PowerShell:

```powershell
cd "c:\Users\elio1\Desktop\Marker Detector"
git remote add origin https://github.com/YOUR_USERNAME/apple-watch-marker-detector.git
git branch -M main
git push -u origin main
```

**החלף `YOUR_USERNAME` בשם המשתמש שלך ב-GitHub!**

### 3️⃣ זהו! ✨
הפרויקט יועלה תוך שניות. כל הקבצים יופיעו ב-GitHub.

---

## שיטה 2: עם GitHub CLI (מהיר יותר!) ⚡

אם אתה רוצה שהכל יקרה אוטומטית:

### 1️⃣ פתח PowerShell **חדש** (חובה! כדי לטעון את gh)

### 2️⃣ התחבר ל-GitHub
```powershell
gh auth login
```
בחר:
- **GitHub.com**
- **HTTPS**
- **Login with a web browser**
- העתק את הקוד שנותן לך והדבק בדפדפן

### 3️⃣ צור repository והעלה הכל בפקודה אחת!
```powershell
cd "c:\Users\elio1\Desktop\Marker Detector"
gh repo create apple-watch-marker-detector --public --source=. --remote=origin --push
```

או אם תרצה **Private**:
```powershell
gh repo create apple-watch-marker-detector --private --source=. --remote=origin --push
```

### 4️⃣ זהו! פתח את ה-repo בדפדפן:
```powershell
gh repo view --web
```

---

## הקבצים שיעלו 📦

```
apple-watch-marker-detector/
├── .gitignore                      # חריגות Git
├── README.md                       # תיאור הפרויקט
├── requirements.txt                # תלויות Python
├── generate_marker.py              # יצירת סמנים
├── detect_marker.py                # זיהוי בזמן אמת
├── test_pipeline.py                # חבילת בדיקות
├── marker.png                      # דוגמת סמן
├── TECHNICAL_DOCUMENTATION.md      # תיעוד אנגלית
└── תיעוד_טכני.md                  # תיעוד עברית
```

**סה"כ:** 9 קבצים, 2859 שורות קוד! 🎉

---

## אם משהו השתבש 🔧

### בעיה: "remote origin already exists"
```powershell
cd "c:\Users\elio1\Desktop\Marker Detector"
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### בעיה: GitHub מבקש אימות
1. לך ל-https://github.com/settings/tokens
2. **Generate new token (classic)**
3. סמן `repo` (full control)
4. העתק את ה-token
5. כשGit מבקש סיסמה, הדבק את ה-token

### בעיה: סניף נקרא `master` במקום `main`
```powershell
git branch -M main
```

---

## מה הלאה? 🚀

אחרי שהעלית ל-GitHub, אפשר:
- **לשתף:** תן את ה-URL לחברים/קולגות
- **Clone:** `git clone https://github.com/YOUR_USERNAME/REPO_NAME.git`
- **עדכונים עתידיים:**
  ```powershell
  cd "c:\Users\elio1\Desktop\Marker Detector"
  git add .
  git commit -m "תיאור השינויים"
  git push
  ```

---

**אם צריך עזרה - פשוט קרא לי! 🤖**
