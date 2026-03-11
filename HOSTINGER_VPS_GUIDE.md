# Deploying on Hostinger VPS

**Yes, you can run the Word Familiarity API on Hostinger VPS.** A VPS is a virtual server — you get root access and full control, similar to EC2.

---

## Hostinger VPS vs EC2

| Aspect | Hostinger VPS | AWS EC2 |
|--------|---------------|---------|
| **What you get** | Virtual server, root access | Virtual server, root access |
| **Deployment** | Same process: SSH → install → run | Same process |
| **Ease** | Similar | Similar |
| **Pricing** | Often cheaper (~$4–20/mo) | ~$65+/mo for t3.large |
| **Control panel** | hPanel (Hostinger) | AWS Console |

---

## Use the EC2 Guide

The steps are **almost identical**. Follow **[EC2_DEPLOYMENT_GUIDE.md](./EC2_DEPLOYMENT_GUIDE.md)** and adapt:

### Hostinger-Specific Changes

1. **SSH access**
   - Get your VPS IP and root password from Hostinger hPanel
   - Or use an SSH key if you've set one up
   ```bash
   ssh root@YOUR_VPS_IP
   # or
   ssh -i your-key.pem root@YOUR_VPS_IP
   ```

2. **OS**
   - Hostinger VPS often uses **Ubuntu** or **CentOS**
   - For Ubuntu: follow the Ubuntu sections in the EC2 guide
   - For CentOS/Rocky: use `dnf` instead of `apt` (similar to Amazon Linux)

3. **User**
   - EC2 uses `ec2-user` or `ubuntu`
   - Hostinger often uses `root` — replace `ec2-user` with `root` and `/home/ec2-user` with `/root` in the systemd service

4. **Security group / firewall**
   - In hPanel: **VPS** → **Firewall** (or use `ufw` / `firewalld` on the server)
   - Open: 22 (SSH), 80 (HTTP), 443 (HTTPS), 8080 (if needed)

5. **Instance size**
   - Choose a plan with **at least 4 GB RAM** (Stanza + PyTorch need it)
   - Hostinger VPS plans: KVM 2 (~4 GB) or higher recommended

---

## Quick Summary

1. Buy a Hostinger VPS (4 GB+ RAM)
2. SSH in as `root`
3. Install Python 3.10+, Poetry, Git
4. Clone your repo or upload files
5. `poetry install --no-dev`
6. Create `.env` with `OPENAI_API_KEY`
7. Run the Stanza preload script
8. Set up systemd service (use `root` and `/root` in paths)
9. Optionally add Nginx + SSL

Same workflow as EC2 — just different provider and credentials.
