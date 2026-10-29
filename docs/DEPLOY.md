# Deploying Cocreate for free

Cocreate is one Docker image (FastAPI serving the built React app). Any host that runs a container with ~512 MB RAM
works; the recipe below is Render's free tier, which deploys straight from the public GitHub repo.

## Render (free tier)

1. Create an account at https://render.com (GitHub login is fine).
2. **New → Blueprint**. Either connect the GitHub repo `RishikeshAluguvelli/Cocreate`, or choose *Public Git
   repository* and paste `https://github.com/RishikeshAluguvelli/Cocreate`.
3. Pick the branch to deploy (`deploy`, or `main` once these changes are merged). Render reads `render.yaml`: a Docker
   web service on the free plan with the health check on `/api/health` and the variables below already set.
4. When asked, enter **`GEMINI_API_KEY`** (the only secret). Click *Apply*. The first build takes 5 to 8 minutes
   (npm build + Python dependencies); later builds are faster.
5. Open the service URL (`https://cocreate-xxxx.onrender.com`). You land in a private guest workspace. Open
   *Aerials · remove the garbage bins*, select Photo 1 with *Only Photo 1*, send "Remove the garbage bins": the first
   request also warms the model path.
6. Optional: set `PUBLIC_BASE_URL` to the service URL (used in password-reset links). Everything else is in `render.yaml`.

### What the free tier means

| | |
|---|---|
| Instance | 512 MB RAM, shared CPU. `EDIT_CONCURRENCY=1` keeps peak memory near 330 MB; a 9-photo request runs its edits one after another. CPU steps (masks, compositing) take a few seconds each here instead of under a second on a laptop. |
| Sleep | The instance sleeps after 15 minutes without traffic; the next visit waits roughly a minute for it to wake. Open the URL a couple of minutes before a demo. |
| Disk | Ephemeral: shoots, versions and accounts disappear on redeploys and restarts. Demo shoots rebuild from the dataset in seconds. A persistent disk is a paid add-on. |
| Hours | 750 free instance hours a month, enough for one service running all month. |

### Verifying a deployment

```bash
BASE=https://cocreate-xxxx.onrender.com
curl -s $BASE/api/health                      # {"ok":true,"planner":true,...,"auth_mode":"guest"}
curl -s -c c.txt $BASE/api/auth/me            # a guest user; the cookie is set
curl -s -b c.txt -X POST $BASE/api/shoots/demo/bins | head -c 200   # a demo shoot is created for that guest
```
Then in the browser: the home page shows the workspace card, a demo opens, the Colour tool previews a region in
under two seconds, and a chat request streams status lines and finishes with a new version.

## Alternatives

- **Koyeb** free tier (one web service, 512 MB, scales to zero): *Create service → Docker → GitHub repo*, port 8000,
  the same environment variables as `render.yaml`.
- **Google Cloud Run**: the most capable free tier (2 vCPU, 2 GB while serving, scale to zero), but it needs a Google
  Cloud project with billing enabled. `gcloud run deploy cocreate --source . --port 8000 --memory 2Gi --timeout 900
  --set-env-vars AUTH_MODE=guest,COOKIE_SECURE=true --set-secrets GEMINI_API_KEY=...`.
- **A laptop for a live demo**: `./dev.sh` plus `cloudflared tunnel --url http://localhost:8000` gives a public URL in
  seconds with no account, as long as the laptop stays online.

Hugging Face Spaces are not an option any more: Docker Spaces need a PRO subscription.

## Running as a container anywhere

```bash
docker build -t cocreate .
docker run -p 8000:8000 -e GEMINI_API_KEY=... -e AUTH_MODE=guest -e COOKIE_SECURE=true -e EDIT_CONCURRENCY=1 cocreate
```
The image runs as an unprivileged user (uid 1000) and writes only under `/srv/data`; mount a volume there to keep
shoots and accounts across restarts.
