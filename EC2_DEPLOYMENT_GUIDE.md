# Deploying Word Familiarity API on AWS EC2

Step-by-step guide to host the Word Familiarity API on an Amazon EC2 instance.

---

## Prerequisites

- AWS account
- SSH key pair (or create one during EC2 launch)
- Your code in a Git repository (GitHub, GitLab, etc.) or a way to transfer files

---

## Step 1: Launch an EC2 Instance

### 1.1 Choose Instance Type

The app uses **Stanza NLP models** and **PyTorch**, which are memory-intensive. Recommended:

| Instance Type | vCPUs | RAM | Use Case |
|---------------|-------|-----|----------|
| **t3.medium** | 2 | 4 GB | Light usage, testing |
| **t3.large** | 2 | 8 GB | **Recommended for production** |
| **t3.xlarge** | 4 | 16 GB | High traffic |

### 1.2 Launch Steps

1. Go to **AWS Console** → **EC2** → **Launch Instance**
2. **Name:** `word-familiarity-api` (or your choice)
3. **AMI:** Amazon Linux 2023 (or Ubuntu 22.04 LTS)
4. **Instance type:** `t3.large` (recommended)
5. **Key pair:** Create new or select existing (download `.pem` if new)
6. **Network settings:**
   - Create security group (or use existing)
   - Allow **SSH (22)** from your IP
   - Allow **HTTP (80)** and **HTTPS (443)** from `0.0.0.0/0` (or restrict to your clients)
   - Allow **8080** from `0.0.0.0/0` if you plan to access the API directly (or use Nginx on 80)
7. **Storage:** 30 GB minimum (Stanza models + datasets)
8. Click **Launch instance**

---

## Step 2: Connect to Your Instance

```bash
# Replace with your key path and instance public IP/DNS
chmod 400 ~/path/to/your-key.pem
ssh -i ~/path/to/your-key.pem ec2-user@<PUBLIC_IP>
```

For Ubuntu AMI, use `ubuntu` instead of `ec2-user`:
```bash
ssh -i ~/path/to/your-key.pem ubuntu@<PUBLIC_IP>
```

---

## Step 3: Install System Dependencies

### Amazon Linux 2023

```bash
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-pip git
```

### Ubuntu 22.04

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip git
```

### Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 4: Deploy Your Application

### Option A: Clone from Git

```bash
cd ~
git clone https://github.com/YOUR_ORG/familiarity-model.git
cd familiarity-model
```

### Option B: Transfer via SCP

From your local machine:

```bash
scp -i ~/path/to/your-key.pem -r /Volumes/Data/Projects/ella/familiarity-model ec2-user@<PUBLIC_IP>:~/
```

Then on the EC2 instance:

```bash
cd ~/familiarity-model
```

---

## Step 5: Set Up Python Environment

```bash
cd ~/familiarity-model  # or your project path

# Create virtual environment and install dependencies
poetry install --no-dev

# Activate the environment (or use poetry run)
poetry shell
```

---

## Step 6: Configure Environment Variables

```bash
# Create .env file (never commit this!)
nano .env
```

Add:

```
PORT=8080
OPENAI_API_KEY=sk-your-actual-openai-api-key
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter`).

**Security:** Ensure `.env` is in `.gitignore` and has restricted permissions:

```bash
chmod 600 .env
```

---

## Step 7: Download Stanza Models (First Run)

Stanza downloads language models on first use. Pre-download them:

```bash
cd ~/familiarity-model
poetry run python -c "
from core.tokenizer import tokenizer
tokenizer.preload_all_pipelines()
print('Stanza pipelines preloaded')
"
```

This may take several minutes. Supported canonical codes include `eng`, `ita`, `spa`, `fra`, `deu`, `por`, `nld`, `pol`, `rus`, `jpn`, `kor`, `cmn`, `arb`, `heb` (see `GET /languages` and `locale_aliases` on the running API).

---

## Step 8: Test the Application

```bash
cd ~/familiarity-model
poetry run python main.py
```

In another terminal (or from your local machine):

```bash
curl http://<PUBLIC_IP>:8080/health
```

Expected: `{"status":"healthy"}`

Stop the server with `Ctrl+C` when done testing.

---

## Step 9: Run as a System Service (Production)

Create a systemd unit so the API starts on boot and restarts on failure.

```bash
sudo nano /etc/systemd/system/word-familiarity-api.service
```

Paste (adjust paths if needed):

```ini
[Unit]
Description=Word Familiarity API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/familiarity-model
Environment="PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/ec2-user/familiarity-model/.env
ExecStart=/home/ec2-user/.local/bin/poetry run python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**For Ubuntu**, replace `ec2-user` with `ubuntu` and `/home/ec2-user` with `/home/ubuntu`.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable word-familiarity-api
sudo systemctl start word-familiarity-api
sudo systemctl status word-familiarity-api
```

View logs:

```bash
sudo journalctl -u word-familiarity-api -f
```

---

## Step 10: (Optional) Nginx Reverse Proxy + SSL

To serve on port 80/443 and add HTTPS:

### Install Nginx

```bash
# Amazon Linux
sudo dnf install -y nginx

# Ubuntu
sudo apt install -y nginx
```

### Configure Nginx

```bash
sudo nano /etc/nginx/conf.d/word-familiarity.conf
```

```nginx
server {
    listen 80;
    server_name your-domain.com;  # or your EC2 public IP/DNS

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
    }
}
```

```bash
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx
```

### Add SSL with Let's Encrypt (if you have a domain)

```bash
# Amazon Linux
sudo dnf install -y certbot python3-certbot-nginx

# Ubuntu
sudo apt install -y certbot python3-certbot-nginx

sudo certbot --nginx -d your-domain.com
```

---

## Step 11: Update Security Group

Ensure your security group allows:

- **Port 22** (SSH): Your IP only
- **Port 80** (HTTP): `0.0.0.0/0` (or restricted)
- **Port 443** (HTTPS): `0.0.0.0/0` (if using SSL)
- **Port 8080**: Only if you skip Nginx and access the API directly

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start service | `sudo systemctl start word-familiarity-api` |
| Stop service | `sudo systemctl stop word-familiarity-api` |
| Restart service | `sudo systemctl restart word-familiarity-api` |
| View logs | `sudo journalctl -u word-familiarity-api -f` |
| Check status | `sudo systemctl status word-familiarity-api` |

---

## Troubleshooting

### Out of memory
- Upgrade to `t3.large` or `t3.xlarge`
- Stanza models use ~2–4 GB RAM when loaded

### Stanza download fails
- Check internet connectivity
- Run `poetry run python -c "import stanza; stanza.download('en')"` manually

### OpenAI API errors
- Verify `OPENAI_API_KEY` in `.env`
- Check OpenAI account credits and rate limits

### Port already in use
- `sudo lsof -i :8080` to find process
- Change `PORT` in `.env` if needed

---

## Cost Estimate (Approximate)

| Resource | Monthly Cost (us-east-1) |
|----------|--------------------------|
| t3.large (on-demand) | ~$60 |
| 30 GB EBS | ~$3 |
| Data transfer | Variable |
| **Total** | **~$65+/month** |

Use **Spot Instances** or **Reserved Instances** for lower cost.
