# Deploying Word Familiarity API on Render

Step-by-step guide to host the Word Familiarity API on [Render](https://render.com).

---

## Prerequisites

- [Render](https://render.com) account (free to sign up)
- Code in a **GitHub** or **GitLab** repository
- OpenAI API key

---

## Step 1: Push Your Code to GitHub

If not already done:

```bash
cd /Volumes/Data/Projects/ella/familiarity-model
git add .
git commit -m "Add deployment config"
git remote add origin https://github.com/YOUR_USERNAME/familiarity-model.git
git push -u origin main
```

**Important:** Ensure `.env` is in `.gitignore` — never push API keys to Git.

---

## Step 2: Create a Web Service on Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub/GitLab account if prompted
4. Select your **familiarity-model** repository
5. Click **Connect**

---

## Step 3: Configure the Service

| Setting | Value |
|---------|-------|
| **Name** | `word-familiarity-api` (or your choice) |
| **Region** | Choose closest to your users |
| **Branch** | `main` |
| **Runtime** | **Python 3** |
| **Build Command** | `poetry install --no-dev` |
| **Start Command** | `poetry run python main.py` |

---

## Step 4: Instance Size (Important)

The app uses **Stanza + PyTorch** and needs **at least 2 GB RAM**. Render's free tier (512 MB) will not work.

| Instance | RAM | Cost | Recommendation |
|----------|-----|------|----------------|
| Free | 512 MB | $0 | ❌ Too small |
| Standard | 2 GB | $25/mo | ⚠️ Minimum, may be slow |
| **Pro** | **4 GB** | **$85/mo** | ✅ **Recommended** |
| Pro Plus | 8 GB | $175/mo | For high traffic |

Select **Pro** (4 GB) or higher in the instance type dropdown.

---

## Step 5: Environment Variables

In the **Environment** section, add:

| Key | Value | Secret? |
|-----|-------|---------|
| `PORT` | `8080` | No (Render sets this automatically, but your app reads it) |
| `OPENAI_API_KEY` | `sk-your-actual-key` | **Yes** ✓ |

Render sets `PORT` automatically — your `main.py` already reads it. You can omit `PORT` if you prefer; the default is 8080.

---

## Step 6: Deploy

1. Click **Create Web Service**
2. Render will clone your repo, run `poetry install`, and start the app
3. First deploy takes **5–15 minutes** (downloading Stanza models)
4. Watch the logs for progress

---

## Step 7: Get Your URL

Once deployed, Render gives you a URL like:

```
https://word-familiarity-api.onrender.com
```

Test it:

```bash
curl https://word-familiarity-api.onrender.com/health
# {"status":"healthy"}
```

---

## Optional: render.yaml (Infrastructure as Code)

Create `render.yaml` in your project root for repeatable deploys:

```yaml
services:
  - type: web
    name: word-familiarity-api
    runtime: python
    plan: pro  # or starter, standard, pro, pro plus

    buildCommand: poetry install --no-dev
    startCommand: poetry run python main.py

    envVars:
      - key: OPENAI_API_KEY
        sync: false  # Set manually in dashboard
```

Then create the service via **Blueprint** (New + → Blueprint) and point it at your repo.

---

## Custom Domain

1. Go to your service → **Settings** → **Custom Domains**
2. Add your domain (e.g. `api.yourdomain.com`)
3. Add the CNAME record Render provides to your DNS
4. HTTPS is automatic

---

## Troubleshooting

### Build fails: "poetry: command not found"
- Ensure **Runtime** is **Python 3** (not Docker)
- Render supports Poetry natively for Python runtime

### Out of memory / OOM killed
- Upgrade to **Pro** (4 GB) or **Pro Plus** (8 GB)
- Stanza loads multiple language models at startup

### Slow first request
- Render may spin down free/low-tier instances after inactivity
- Paid instances stay warm; first request after idle can be slower

### Stanza download timeout
- Build can take 10+ minutes. If it times out, try:
  - Use a **Pre-deploy command** to pre-download models (advanced)
  - Or ensure `poetry.lock` is committed for faster installs

---

## Cost Summary

| Plan | Monthly Cost |
|------|--------------|
| Standard (2 GB) | $25 |
| **Pro (4 GB)** | **$85** |
| Pro Plus (8 GB) | $175 |

Plus any bandwidth over 100 GB/month on free tier.
